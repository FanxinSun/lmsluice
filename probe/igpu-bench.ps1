<#
  igpu-bench.ps1 - Measure what an integrated GPU can actually do for LLM inference.

  Uses only what ships with Windows: Direct3D 11 compute (d3d11.dll),
  the runtime HLSL compiler (d3dcompiler_47.dll) and DXGI. No installs.

  Two kernels:
    BANDWIDTH - streaming coalesced reads. This is the number that decides
                token-generation speed, because decode is memory bound.
    FMA       - dependent-free fused multiply-add chains in registers.
                This decides prompt-processing (prefill) speed.

  Usage:
    powershell -ExecutionPolicy Bypass -File .\igpu-bench.ps1
    powershell -ExecutionPolicy Bypass -File .\igpu-bench.ps1 -SizeMB 256 -Reps 100
    powershell -ExecutionPolicy Bypass -File .\igpu-bench.ps1 -Adapter 1
#>
[CmdletBinding()]
param(
    [int]   $Adapter  = -1,      # -1 = every hardware adapter
    [int]   $SizeMB   = 128,     # streaming buffer size
    [int]   $Reps     = 50,      # bandwidth dispatches per timing run
    [int]   $FmaReps  = 10,      # FMA dispatches per timing run
    [switch]$IncludeSoftware     # also bench the WARP software rasterizer
)

$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------
# Native interop. Declared once per session.
# --------------------------------------------------------------------------
if (-not ('IGpuBench' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

public class AdapterInfo {
    public int    Index;
    public string Description;
    public uint   VendorId;
    public uint   DeviceId;
    public ulong  DedicatedVideoMemory;
    public ulong  SharedSystemMemory;
    public bool   IsSoftware;
}

public class BenchResult {
    public double ReadGBs;
    public double GFlops;
    public int    FeatureLevel;
    public string Error;
}

public static class IGpuBench
{
    // ---- entry points -----------------------------------------------------
    [DllImport("dxgi.dll")]
    static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr ppFactory);

    [DllImport("d3d11.dll")]
    static extern int D3D11CreateDevice(
        IntPtr pAdapter, int driverType, IntPtr software, uint flags,
        IntPtr pFeatureLevels, uint featureLevels, uint sdkVersion,
        out IntPtr ppDevice, out int pFeatureLevel, out IntPtr ppContext);

    [DllImport("d3dcompiler_47.dll", CharSet = CharSet.Ansi)]
    static extern int D3DCompile(
        byte[] pSrcData, IntPtr srcDataSize, string pSourceName,
        IntPtr pDefines, IntPtr pInclude, string pEntrypoint, string pTarget,
        uint flags1, uint flags2, out IntPtr ppCode, out IntPtr ppErrorMsgs);

    // ---- vtable plumbing --------------------------------------------------
    // Declaring the full ID3D11DeviceContext interface would mean 115 methods
    // in exact order; reading single slots off the vtable is far less to get
    // wrong. Slot numbers below come from d3d11.h / dxgi.h.
    static IntPtr Slot(IntPtr obj, int slot) {
        IntPtr vtbl = Marshal.ReadIntPtr(obj);
        return Marshal.ReadIntPtr(vtbl, slot * IntPtr.Size);
    }
    static object Fn(IntPtr obj, int slot, Type t) {
        return Marshal.GetDelegateForFunctionPointer(Slot(obj, slot), t);
    }

    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate uint   DRelease(IntPtr self);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DEnumAdapters1(IntPtr self, uint i, out IntPtr adapter);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DGetDesc1(IntPtr self, IntPtr desc);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate IntPtr DBlobPtr(IntPtr self);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate IntPtr DBlobSize(IntPtr self);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DCreateBuffer(IntPtr self, ref BufferDesc d, IntPtr init, out IntPtr buf);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DCreateUav(IntPtr self, IntPtr res, ref UavDesc d, out IntPtr uav);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DCreateCs(IntPtr self, IntPtr code, IntPtr len, IntPtr linkage, out IntPtr cs);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void   DCsSetShader(IntPtr self, IntPtr cs, IntPtr inst, uint n);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void   DCsSetUavs(IntPtr self, uint start, uint n, IntPtr[] uavs, uint[] counts);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void   DDispatch(IntPtr self, uint x, uint y, uint z);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void   DCopyResource(IntPtr self, IntPtr dst, IntPtr src);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int    DMap(IntPtr self, IntPtr res, uint sub, uint type, uint flags, out MappedSub m);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void   DUnmap(IntPtr self, IntPtr res, uint sub);

    [StructLayout(LayoutKind.Sequential)]
    struct BufferDesc { public uint ByteWidth, Usage, BindFlags, CpuAccessFlags, MiscFlags, StructureByteStride; }
    [StructLayout(LayoutKind.Sequential)]
    struct UavDesc { public uint Format, ViewDimension, FirstElement, NumElements, Flags; }
    [StructLayout(LayoutKind.Sequential)]
    struct MappedSub { public IntPtr pData; public uint RowPitch, DepthPitch; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct AdapterDesc1 {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Description;
        public uint VendorId, DeviceId, SubSysId, Revision;
        public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public long AdapterLuid;
        public uint Flags;
    }

    // ID3D11Device               : CreateBuffer 3, CreateUnorderedAccessView 8, CreateComputeShader 18
    // ID3D11DeviceContext        : Map 14, Unmap 15, Dispatch 41, CopyResource 47,
    //                              CSSetUnorderedAccessViews 68, CSSetShader 69
    // IDXGIFactory1              : EnumAdapters1 12
    // IDXGIAdapter1              : GetDesc1 10
    const int GROUP_SIZE    = 256;
    const int GROUPS        = 1024;
    const int TOTAL_THREADS = GROUP_SIZE * GROUPS;   // 262144
    const int FMA_ITERS     = 2048;

    static void Rel(IntPtr p) {
        if (p != IntPtr.Zero) ((DRelease)Fn(p, 2, typeof(DRelease)))(p);
    }
    static void Check(int hr, string what) {
        if (hr < 0) throw new Exception(what + " failed: 0x" + hr.ToString("X8"));
    }

    public static AdapterInfo[] Enumerate() {
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387"); // IDXGIFactory1
        IntPtr factory;
        Check(CreateDXGIFactory1(ref iid, out factory), "CreateDXGIFactory1");
        var list = new System.Collections.Generic.List<AdapterInfo>();
        try {
            var enumAd = (DEnumAdapters1)Fn(factory, 12, typeof(DEnumAdapters1));
            IntPtr buf = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(AdapterDesc1)));
            try {
                for (uint i = 0; ; i++) {
                    IntPtr ad;
                    if (enumAd(factory, i, out ad) < 0 || ad == IntPtr.Zero) break;
                    try {
                        Check(((DGetDesc1)Fn(ad, 10, typeof(DGetDesc1)))(ad, buf), "GetDesc1");
                        var d = (AdapterDesc1)Marshal.PtrToStructure(buf, typeof(AdapterDesc1));
                        var a = new AdapterInfo();
                        a.Index = (int)i;
                        a.Description = d.Description;
                        a.VendorId = d.VendorId;
                        a.DeviceId = d.DeviceId;
                        a.DedicatedVideoMemory = d.DedicatedVideoMemory.ToUInt64();
                        a.SharedSystemMemory = d.SharedSystemMemory.ToUInt64();
                        a.IsSoftware = (d.Flags & 2) != 0;
                        list.Add(a);
                    } finally { Rel(ad); }
                }
            } finally { Marshal.FreeHGlobal(buf); }
        } finally { Rel(factory); }
        return list.ToArray();
    }

    static IntPtr Compile(string hlsl) {
        byte[] src = System.Text.Encoding.ASCII.GetBytes(hlsl);
        IntPtr code, err;
        int hr = D3DCompile(src, (IntPtr)src.Length, "cs.hlsl", IntPtr.Zero, IntPtr.Zero,
                            "main", "cs_5_0", 1u << 15 /* O3 */, 0, out code, out err);
        if (hr < 0) {
            string msg = "unknown";
            if (err != IntPtr.Zero) {
                msg = Marshal.PtrToStringAnsi(((DBlobPtr)Fn(err, 3, typeof(DBlobPtr)))(err));
                Rel(err);
            }
            throw new Exception("HLSL compile failed: " + msg);
        }
        if (err != IntPtr.Zero) Rel(err);
        return code;
    }

    static string BandwidthHlsl(int iters) {
        return
        "RWStructuredBuffer<float4> Src : register(u0);\n" +
        "RWStructuredBuffer<float4> Dst : register(u1);\n" +
        "[numthreads(" + GROUP_SIZE + ",1,1)]\n" +
        "void main(uint3 gid : SV_DispatchThreadID) {\n" +
        "  const uint stride = " + TOTAL_THREADS + "u;\n" +
        "  uint idx = gid.x;\n" +
        "  float4 a=0, b=0, c=0, d=0;\n" +
        "  [loop] for (uint k = 0; k < " + iters + "u; k += 4) {\n" +
        "    a += Src[idx + (k+0)*stride];\n" +
        "    b += Src[idx + (k+1)*stride];\n" +
        "    c += Src[idx + (k+2)*stride];\n" +
        "    d += Src[idx + (k+3)*stride];\n" +
        "  }\n" +
        "  Dst[idx] = a+b+c+d;\n" +
        "}\n";
    }

    static string FmaHlsl() {
        return
        "RWStructuredBuffer<float4> Dst : register(u1);\n" +
        "[numthreads(" + GROUP_SIZE + ",1,1)]\n" +
        "void main(uint3 gid : SV_DispatchThreadID) {\n" +
        "  float f = (float)gid.x;\n" +
        "  float4 a = float4(f, f+1, f+2, f+3);\n" +
        "  float4 b = a*0.5  + 1.0;\n" +
        "  float4 c = a*0.25 + 2.0;\n" +
        "  float4 d = a*0.125+ 3.0;\n" +
        "  float4 m = float4(1.0001, 0.9999, 1.0002, 0.9998);\n" +
        "  float4 n = float4(0.0001, 0.0002, 0.0003, 0.0004);\n" +
        "  [loop] for (uint k = 0; k < " + FMA_ITERS + "u; k++) {\n" +
        "    a = a*m+n; b = b*m+n; c = c*m+n; d = d*m+n;\n" +
        "    a = a*m+n; b = b*m+n; c = c*m+n; d = d*m+n;\n" +
        "  }\n" +
        "  Dst[gid.x] = a+b+c+d;\n" +
        "}\n";
    }


    // CPU baseline. On an integrated GPU both engines share one DDR bus, so
    // this is the number the iGPU has to beat to be worth using at all.
    public static BenchResult RunCpu(int sizeMB, int reps) {
        var r = new BenchResult();
        try {
            // .NET caps a single object at 2 GB, so the CPU array tops out at
            // 1 GB. That is already well clear of any cache, so the steady
            // state number is unaffected.
            if (sizeMB > 1024) sizeMB = 1024;
            long bytes = (long)sizeMB * 1024 * 1024;
            int n = (int)(bytes / 4);
            float[] buf = new float[n];
            for (int i = 0; i < n; i++) buf[i] = 1.0f;
            int threads = Environment.ProcessorCount;
            int chunk = n / threads;
            double[] sink = new double[threads];
            System.Threading.Tasks.Parallel.For(0, threads, delegate(int t) {
                int s = t * chunk, e = (t == threads - 1) ? n : s + chunk;
                float a = 0, b = 0, c = 0, d = 0;
                for (int i = s; i + 3 < e; i += 4) { a += buf[i]; b += buf[i+1]; c += buf[i+2]; d += buf[i+3]; }
                sink[t] = a + b + c + d;
            });
            var sw = Stopwatch.StartNew();
            for (int rep = 0; rep < reps; rep++) {
                System.Threading.Tasks.Parallel.For(0, threads, delegate(int t) {
                    int s = t * chunk, e = (t == threads - 1) ? n : s + chunk;
                    float a = 0, b = 0, c = 0, d = 0;
                    for (int i = s; i + 3 < e; i += 4) { a += buf[i]; b += buf[i+1]; c += buf[i+2]; d += buf[i+3]; }
                    sink[t] = a + b + c + d;
                });
            }
            sw.Stop();
            r.ReadGBs = (double)bytes * reps / sw.Elapsed.TotalSeconds / 1e9;
            if (sink[0] == 1e30) r.GFlops = 1;
        } catch (Exception e) { r.Error = e.Message; }
        return r;
    }

    public static BenchResult Run(int adapterIndex, int sizeMB, int reps, int fmaReps) {
        var r = new BenchResult();
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        IntPtr factory = IntPtr.Zero, adapter = IntPtr.Zero, dev = IntPtr.Zero, ctx = IntPtr.Zero;
        IntPtr src = IntPtr.Zero, dst = IntPtr.Zero, stage = IntPtr.Zero;
        IntPtr uavSrc = IntPtr.Zero, uavDst = IntPtr.Zero;
        IntPtr bwBlob = IntPtr.Zero, fmaBlob = IntPtr.Zero, bwCs = IntPtr.Zero, fmaCs = IntPtr.Zero;
        try {
            Check(CreateDXGIFactory1(ref iid, out factory), "CreateDXGIFactory1");
            Check(((DEnumAdapters1)Fn(factory, 12, typeof(DEnumAdapters1)))(factory, (uint)adapterIndex, out adapter), "EnumAdapters1");

            int fl;
            Check(D3D11CreateDevice(adapter, 0 /* UNKNOWN */, IntPtr.Zero, 0,
                                    IntPtr.Zero, 0, 7 /* D3D11_SDK_VERSION */,
                                    out dev, out fl, out ctx), "D3D11CreateDevice");
            r.FeatureLevel = fl;

            long srcBytes  = (long)sizeMB * 1024 * 1024;
            int  srcElems  = (int)(srcBytes / 16);
            int  iters     = srcElems / TOTAL_THREADS;
            if (iters < 4 || (iters % 4) != 0) throw new Exception("SizeMB must be a multiple of 16");

            var createBuf = (DCreateBuffer)Fn(dev, 3, typeof(DCreateBuffer));
            var createUav = (DCreateUav)Fn(dev, 8, typeof(DCreateUav));
            var createCs  = (DCreateCs)Fn(dev, 18, typeof(DCreateCs));

            BufferDesc bd = new BufferDesc();
            bd.Usage = 0; bd.BindFlags = 0x80; bd.MiscFlags = 0x40; bd.StructureByteStride = 16;
            bd.ByteWidth = (uint)srcBytes;
            Check(createBuf(dev, ref bd, IntPtr.Zero, out src), "CreateBuffer(src)");
            bd.ByteWidth = (uint)(TOTAL_THREADS * 16);
            Check(createBuf(dev, ref bd, IntPtr.Zero, out dst), "CreateBuffer(dst)");

            BufferDesc sd = new BufferDesc();
            sd.ByteWidth = (uint)(TOTAL_THREADS * 16);
            sd.Usage = 3 /* STAGING */; sd.CpuAccessFlags = 0x20000 /* READ */;
            Check(createBuf(dev, ref sd, IntPtr.Zero, out stage), "CreateBuffer(stage)");

            UavDesc ud = new UavDesc();
            ud.Format = 0; ud.ViewDimension = 1; ud.FirstElement = 0; ud.Flags = 0;
            ud.NumElements = (uint)srcElems;
            Check(createUav(dev, src, ref ud, out uavSrc), "CreateUAV(src)");
            ud.NumElements = (uint)TOTAL_THREADS;
            Check(createUav(dev, dst, ref ud, out uavDst), "CreateUAV(dst)");

            bwBlob  = Compile(BandwidthHlsl(iters));
            fmaBlob = Compile(FmaHlsl());
            var bp = (DBlobPtr)Fn(bwBlob, 3, typeof(DBlobPtr));
            var bs = (DBlobSize)Fn(bwBlob, 4, typeof(DBlobSize));
            Check(createCs(dev, bp(bwBlob), bs(bwBlob), IntPtr.Zero, out bwCs), "CreateComputeShader(bw)");
            var fp = (DBlobPtr)Fn(fmaBlob, 3, typeof(DBlobPtr));
            var fs = (DBlobSize)Fn(fmaBlob, 4, typeof(DBlobSize));
            Check(createCs(dev, fp(fmaBlob), fs(fmaBlob), IntPtr.Zero, out fmaCs), "CreateComputeShader(fma)");

            var setShader = (DCsSetShader)Fn(ctx, 69, typeof(DCsSetShader));
            var setUavs   = (DCsSetUavs)Fn(ctx, 68, typeof(DCsSetUavs));
            var dispatch  = (DDispatch)Fn(ctx, 41, typeof(DDispatch));
            var copyRes   = (DCopyResource)Fn(ctx, 47, typeof(DCopyResource));
            var map       = (DMap)Fn(ctx, 14, typeof(DMap));
            var unmap     = (DUnmap)Fn(ctx, 15, typeof(DUnmap));

            IntPtr[] uavs = new IntPtr[] { uavSrc, uavDst };
            uint[] counts = new uint[] { 0xFFFFFFFFu, 0xFFFFFFFFu };
            setUavs(ctx, 0, 2, uavs, counts);

            // Mapping the staging copy blocks until the GPU has drained, which
            // is all the synchronisation this needs.
            Action drain = delegate {
                copyRes(ctx, stage, dst);
                MappedSub ms;
                Check(map(ctx, stage, 0, 1 /* MAP_READ */, 0, out ms), "Map");
                unmap(ctx, stage, 0);
            };

            var sw = new Stopwatch();

            setShader(ctx, bwCs, IntPtr.Zero, 0);
            for (int i = 0; i < 3; i++) dispatch(ctx, GROUPS, 1, 1);
            drain();
            sw.Restart();
            for (int i = 0; i < reps; i++) dispatch(ctx, GROUPS, 1, 1);
            drain();
            sw.Stop();
            double secs = sw.Elapsed.TotalSeconds;
            r.ReadGBs = (double)srcBytes * reps / secs / 1e9;

            setShader(ctx, fmaCs, IntPtr.Zero, 0);
            dispatch(ctx, GROUPS, 1, 1);
            drain();
            sw.Restart();
            for (int i = 0; i < fmaReps; i++) dispatch(ctx, GROUPS, 1, 1);
            drain();
            sw.Stop();
            secs = sw.Elapsed.TotalSeconds;
            // 8 vector FMA per loop iteration * 4 lanes * 2 flops
            double flops = (double)TOTAL_THREADS * FMA_ITERS * 64.0 * fmaReps;
            r.GFlops = flops / secs / 1e9;
        } catch (Exception e) {
            r.Error = e.Message;
        } finally {
            Rel(uavSrc); Rel(uavDst); Rel(stage); Rel(dst); Rel(src);
            Rel(bwCs); Rel(fmaCs); Rel(bwBlob); Rel(fmaBlob);
            Rel(ctx); Rel(dev); Rel(adapter); Rel(factory);
        }
        return r;
    }
}
'@
}

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
function Get-VendorName([uint32]$id) {
    switch ($id) {
        0x1002  { 'AMD' }     0x10DE { 'NVIDIA' }  0x8086 { 'Intel' }
        0x1414  { 'Microsoft' } default { ('0x{0:X4}' -f $id) }
    }
}

$adapters = [IGpuBench]::Enumerate()

Write-Host ''
Write-Host 'Adapters' -ForegroundColor Cyan
Write-Host '--------'
foreach ($a in $adapters) {
    $tag = if ($a.IsSoftware) { ' [software]' } else { '' }
    '{0}: {1,-34} {2,-6} dev 0x{3:X4}  vram {4,6:N0} MB  shared {5,6:N0} MB{6}' -f `
        $a.Index, $a.Description.Trim(), (Get-VendorName $a.VendorId), $a.DeviceId,
        ($a.DedicatedVideoMemory / 1MB), ($a.SharedSystemMemory / 1MB), $tag | Write-Host
}

$targets = if ($Adapter -ge 0) { $adapters | Where-Object { $_.Index -eq $Adapter } }
           else { $adapters | Where-Object { $IncludeSoftware -or -not $_.IsSoftware } }

if (-not $targets) { Write-Host "`nNo matching adapter." -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host ("Benchmarking  buffer {0} MB, {1} bandwidth reps, {2} FMA reps" -f $SizeMB, $Reps, $FmaReps) -ForegroundColor Cyan
Write-Host '------------------------------------------------------------'

$rows = @()
foreach ($a in $targets) {
    Write-Host ("  {0} ..." -f $a.Description.Trim()) -NoNewline
    $r = [IGpuBench]::Run($a.Index, $SizeMB, $Reps, $FmaReps)
    if ($r.Error) {
        Write-Host (" FAILED: {0}" -f $r.Error) -ForegroundColor Red
        continue
    }
    Write-Host ' ok'
    $rows += [pscustomobject]@{
        Adapter        = $a.Description.Trim()
        'Read GB/s'    = [math]::Round($r.ReadGBs, 1)
        'FP32 GFLOP/s' = [math]::Round($r.GFlops, 0)
        'FeatureLevel' = ('{0:X}' -f $r.FeatureLevel)
    }
}

Write-Host ("  CPU ({0} threads, all cores) ..." -f [Environment]::ProcessorCount) -NoNewline
$c = [IGpuBench]::RunCpu($SizeMB, $Reps)
if ($c.Error) { Write-Host (" FAILED: {0}" -f $c.Error) -ForegroundColor Red }
else {
    Write-Host ' ok'
    $rows += [pscustomobject]@{
        Adapter        = 'CPU (all cores)'
        'Read GB/s'    = [math]::Round($c.ReadGBs, 1)
        'FP32 GFLOP/s' = $null
        'FeatureLevel' = '-'
    }
}

if (-not $rows) { exit 1 }
Write-Host ''
$rows | Format-Table -AutoSize

# --------------------------------------------------------------------------
# What the bandwidth number means for LLM decode.
# Token generation streams the whole active weight set once per token, so
# tok/s has a hard ceiling of  bandwidth / model bytes.  Real runtimes reach
# roughly 70% of a pure streaming read.
# --------------------------------------------------------------------------
$models = @(
    @{ Name = 'Llama 3.1 8B      Q4_K_M'; GB =  4.9 }
    @{ Name = 'Qwen 14B          Q4_K_M'; GB =  8.6 }
    @{ Name = 'Qwen 32B          Q4_K_M'; GB = 19.8 }
    @{ Name = 'Llama 70B         Q4_K_M'; GB = 42.5 }
    @{ Name = 'gpt-oss-20b  MoE (active)'; GB =  2.6 }
    @{ Name = 'gpt-oss-120b MoE (active)'; GB =  3.6 }
)

Write-Host 'Projected decode ceiling (tok/s, at 70% of measured read bandwidth)' -ForegroundColor Cyan
Write-Host '-------------------------------------------------------------------'

$proj = foreach ($m in $models) {
    $row = [ordered]@{ Model = $m.Name; 'Weights GB' = $m.GB }
    foreach ($r in $rows) {
        $tps = ($r.'Read GB/s' * 0.70) / $m.GB
        $row[$r.Adapter] = [math]::Round($tps, 1)
    }
    [pscustomobject]$row
}
$proj | Format-Table -AutoSize

Write-Host 'Note: MoE rows count only the active experts, but the full model must'
Write-Host '      still fit in memory. Prefill speed tracks the GFLOP/s column, not'
Write-Host '      bandwidth, so long prompts degrade far worse than these numbers.'
Write-Host ''
