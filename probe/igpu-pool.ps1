<#
.SYNOPSIS
  Shared-memory pool staircase for a Direct3D 11 adapter: how many compute-shader
  threadgroups of a given shared-memory size are resident at once?

.DESCRIPTION
  This is the measurement `gate.py` cannot get any other way. Residency on a
  device is floor(pool / shared-memory-per-group), and D3D11, DXGI and Vulkan
  report no per-unit pool, only the per-group cap. The decoder's block asks for
  ~31 KiB, so on the smallest AMD iGPU one number -- 1 or 3 resident blocks per
  compute unit -- decides whether the decoder beats the CPU there. See
  probe/README.md, "Determining the shared-memory pool empirically".

  Method: a compute shader declares `groupshared` memory of S bytes at T threads
  and runs a latency-bound dependent-load chain, so each group stays resident
  for a fixed, measurable time. Dispatch G groups for a sweep of G. If C groups
  are resident at once the elapsed time is flat up to G = C and steps near
  G = C + 1 (then again near 2C), so the staircase gives C directly. Divide by
  the adapter's compute-unit count (-Units; D3D11 cannot query it) and the pool
  follows as at least  groups-per-unit x S.

  THIS SCRIPT ONLY COLLECTS RAW CURVES. It prints a preliminary estimate, but the
  JSON it writes is the result: analyse it with probe/igpu-pool-analyze.py.

  RUN IT TWICE, at different times (ideally after a reboot), and analyse both files
  together. On the Apple M4 a boundary shape read 4 groups per core in one run and
  3.4 in another; a single run cannot tell a boundary shape from a clean one, and a
  pool bound built on it is wrong. The analyzer counts a shape only if it agrees in
  every run.

  CALIBRATE FIRST. Run it on a card whose occupancy is independently known (an
  RTX 5080: -Units 84; lmz measured 3-4 blocks per SM for the 31,744 B shape, and
  the queried pool of 102,400 B gives 3) and confirm the method recovers it before
  believing an answer on a device where nothing can check it.

.PARAMETER Adapter
  DXGI adapter index (see the list printed at the start). -1 lists them and exits.
.PARAMETER Units
  Compute units (CUs / SMs / Xe cores) of the chosen adapter. D3D11 cannot query
  it; take it from the vendor's spec or another tool. 0 = unknown: the script then
  reports total resident groups and cannot divide per unit.
.PARAMETER Quick
  Four shapes instead of seventeen (about a minute).
.PARAMETER Json
  Where to write the result (default: igpu-pool-<adapter>-<time>.json in the
  current directory).

.EXAMPLE
  powershell -File igpu-pool.ps1 -Adapter -1
  powershell -File igpu-pool.ps1 -Adapter 1 -Units 2
#>
[CmdletBinding()]
param(
    [int]    $Adapter        = -1,
    [int]    $Units          = 0,
    [int]    $MaxGroups      = 0,       # 0 = automatic from -Units
    [int]    $Reps           = 5,       # timed dispatches per G (median is used)
    [double] $MinT1Ms        = 0.8,     # one-group time the chain is calibrated to
    [double] $MaxDispatchSec = 1.0,     # stop a sweep before any dispatch nears the 2 s TDR
    [string] $Json           = '',
    [switch] $Quick,
    [switch] $IncludeSoftware
)

$ErrorActionPreference = 'Stop'

if (-not ('IGpuPool' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

public class PoolAdapterInfo {
    public int    Index;
    public string Description;
    public uint   VendorId;
    public uint   DeviceId;
    public ulong  DedicatedVideoMemory;
    public ulong  SharedSystemMemory;
    public bool   IsSoftware;
}

public class PoolShape {
    public string   Name;
    public int      T;
    public int      S;
    public int      Hops;
    public double   T1Ms;
    public int[]    G;
    public double[] MedianMs;
    public double[] MinMs;
    public double[] MaxMs;
    public bool     Verified;
    public string   VerifyNote;
    public string   Error;
    public int      FeatureLevel;
}

public static class IGpuPool
{
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

    // Raw vtable slots, as in igpu-bench.ps1 (slot numbers from d3d11.h / dxgi.h).
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

    // ID3D11Device        : CreateBuffer 3, CreateUnorderedAccessView 8, CreateComputeShader 18
    // ID3D11DeviceContext : Map 14, Unmap 15, Dispatch 41, CopyResource 47,
    //                       CSSetUnorderedAccessViews 68, CSSetShader 69
    // IDXGIFactory1       : EnumAdapters1 12      IDXGIAdapter1 : GetDesc1 10
    // ID3D11Blob (ID3DBlob): GetBufferPointer 3, GetBufferSize 4

    static void Rel(IntPtr p) {
        if (p != IntPtr.Zero) ((DRelease)Fn(p, 2, typeof(DRelease)))(p);
    }
    static void Check(int hr, string what) {
        if (hr < 0) throw new Exception(what + " failed: 0x" + hr.ToString("X8"));
    }

    public static PoolAdapterInfo[] Enumerate() {
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387"); // IDXGIFactory1
        IntPtr factory;
        Check(CreateDXGIFactory1(ref iid, out factory), "CreateDXGIFactory1");
        var list = new List<PoolAdapterInfo>();
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
                        var a = new PoolAdapterInfo();
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
        byte[] src = Encoding.ASCII.GetBytes(hlsl);
        IntPtr code, err;
        int hr = D3DCompile(src, (IntPtr)src.Length, "pool.hlsl", IntPtr.Zero, IntPtr.Zero,
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

    // The probe kernel. Every thread fills its share of S bytes of groupshared
    // memory, then follows a dependent chain through it: each load's address
    // depends on the previous load, so the group's run time is latency-bound and
    // does not depend on how many other groups are resident. A resident group
    // therefore occupies its shared memory for a fixed time, which is what makes
    // the staircase visible. Every thread writes its result, so none of the chain
    // can be removed; Sync is a tiny buffer that gives the host something small
    // to copy back to wait for the dispatch.
    static string PoolHlsl(int t, int s, int hops) {
        int words = s / 4;
        return
        "RWStructuredBuffer<uint> Out  : register(u0);\n" +
        "RWStructuredBuffer<uint> Sync : register(u1);\n" +
        "groupshared uint lds[" + words + "];\n" +
        "[numthreads(" + t + ",1,1)]\n" +
        "void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {\n" +
        "  uint th = tid.x;\n" +
        "  [loop] for (uint i = th; i < " + words + "u; i += " + t + "u)\n" +
        "    lds[i] = (i * 2654435761u) ^ 1515870810u;\n" +
        "  GroupMemoryBarrierWithGroupSync();\n" +
        "  uint v = (th ^ (gid.x * 1009u)) + 1u;\n" +
        "  [loop] for (uint k = 0; k < " + hops + "u; k++) {\n" +
        "    v = (v ^ (v >> 13)) * 1664525u + 1013904223u;\n" +
        "    v += lds[v % " + words + "u];\n" +
        "  }\n" +
        "  Out[gid.x * " + t + "u + th] = v;\n" +
        "  if (th == 0) Sync[gid.x & 63u] = v;\n" +
        "}\n";
    }

    // The same chain on the CPU, for the verification below. 1515870810 = 0x5a5a5a5a.
    static uint Expect(uint th, uint g, uint hops, uint words) {
        unchecked {
            uint v = (th ^ (g * 1009u)) + 1u;
            for (uint k = 0; k < hops; k++) {
                v = (v ^ (v >> 13)) * 1664525u + 1013904223u;
                uint idx = v % words;
                v += (idx * 2654435761u) ^ 1515870810u;
            }
            return v;
        }
    }

    static List<int> GridList(int maxG) {
        var l = new List<int>();
        for (int g = 1;   g <= 64  && g <= maxG; g += 1) l.Add(g);
        for (int g = 66;  g <= 200 && g <= maxG; g += 2) l.Add(g);
        for (int g = 204; g <= maxG;             g += 4) l.Add(g);
        return l;
    }

    public static PoolShape[] RunPool(int adapterIndex, int[] ts, int[] ss, string[] names,
                                      int maxG, int reps, double minT1Ms, double maxDispatchSec)
    {
        var results = new List<PoolShape>();
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        IntPtr factory = IntPtr.Zero, adapter = IntPtr.Zero, dev = IntPtr.Zero, ctx = IntPtr.Zero;
        try {
            Check(CreateDXGIFactory1(ref iid, out factory), "CreateDXGIFactory1");
            Check(((DEnumAdapters1)Fn(factory, 12, typeof(DEnumAdapters1)))(factory, (uint)adapterIndex, out adapter), "EnumAdapters1");
            int fl;
            Check(D3D11CreateDevice(adapter, 0 /* UNKNOWN */, IntPtr.Zero, 0,
                                    IntPtr.Zero, 0, 7 /* D3D11_SDK_VERSION */,
                                    out dev, out fl, out ctx), "D3D11CreateDevice");
            for (int i = 0; i < ts.Length; i++) {
                PoolShape ps;
                try {
                    ps = RunOneShape(dev, ctx, ts[i], ss[i], names[i], maxG, reps, minT1Ms, maxDispatchSec);
                } catch (Exception e) {
                    ps = new PoolShape();
                    ps.Name = names[i]; ps.T = ts[i]; ps.S = ss[i]; ps.Error = e.Message;
                }
                ps.FeatureLevel = fl;
                results.Add(ps);
            }
        } finally { Rel(ctx); Rel(dev); Rel(adapter); Rel(factory); }
        return results.ToArray();
    }

    static PoolShape RunOneShape(IntPtr dev, IntPtr ctx, int T, int S, string name,
                                 int maxG, int reps, double minT1Ms, double maxDispatchSec)
    {
        var ps = new PoolShape();
        ps.Name = name; ps.T = T; ps.S = S;

        var createBuf = (DCreateBuffer)Fn(dev, 3, typeof(DCreateBuffer));
        var createUav = (DCreateUav)Fn(dev, 8, typeof(DCreateUav));
        var createCs  = (DCreateCs)Fn(dev, 18, typeof(DCreateCs));
        var setShader = (DCsSetShader)Fn(ctx, 69, typeof(DCsSetShader));
        var setUavs   = (DCsSetUavs)Fn(ctx, 68, typeof(DCsSetUavs));
        var dispatch  = (DDispatch)Fn(ctx, 41, typeof(DDispatch));
        var copyRes   = (DCopyResource)Fn(ctx, 47, typeof(DCopyResource));
        var map       = (DMap)Fn(ctx, 14, typeof(DMap));
        var unmap     = (DUnmap)Fn(ctx, 15, typeof(DUnmap));

        IntPtr outBuf = IntPtr.Zero, syncBuf = IntPtr.Zero, stageOut = IntPtr.Zero, stageSync = IntPtr.Zero;
        IntPtr uavOut = IntPtr.Zero, uavSync = IntPtr.Zero, blob = IntPtr.Zero, cs = IntPtr.Zero;
        try {
            uint outWords = (uint)maxG * (uint)T;
            BufferDesc bd = new BufferDesc();
            bd.Usage = 0; bd.BindFlags = 0x80; bd.MiscFlags = 0x40; bd.StructureByteStride = 4;
            bd.ByteWidth = outWords * 4;
            Check(createBuf(dev, ref bd, IntPtr.Zero, out outBuf), "CreateBuffer(out)");
            bd.ByteWidth = 64 * 4;
            Check(createBuf(dev, ref bd, IntPtr.Zero, out syncBuf), "CreateBuffer(sync)");

            BufferDesc sd = new BufferDesc();
            sd.Usage = 3 /* STAGING */; sd.CpuAccessFlags = 0x20000 /* READ */;
            sd.ByteWidth = outWords * 4;
            Check(createBuf(dev, ref sd, IntPtr.Zero, out stageOut), "CreateBuffer(stageOut)");
            sd.ByteWidth = 64 * 4;
            Check(createBuf(dev, ref sd, IntPtr.Zero, out stageSync), "CreateBuffer(stageSync)");

            UavDesc ud = new UavDesc();
            ud.Format = 0; ud.ViewDimension = 1; ud.FirstElement = 0; ud.Flags = 0;
            ud.NumElements = outWords;
            Check(createUav(dev, outBuf, ref ud, out uavOut), "CreateUAV(out)");
            ud.NumElements = 64;
            Check(createUav(dev, syncBuf, ref ud, out uavSync), "CreateUAV(sync)");
            setUavs(ctx, 0, 2, new IntPtr[] { uavOut, uavSync }, new uint[] { 0xFFFFFFFFu, 0xFFFFFFFFu });

            // Copying the small Sync buffer back is ordered after the dispatch on the
            // same context, so Map blocks until the dispatch has finished.
            Action drain = delegate {
                copyRes(ctx, stageSync, syncBuf);
                MappedSub ms;
                Check(map(ctx, stageSync, 0, 1 /* MAP_READ */, 0, out ms), "Map(sync)");
                unmap(ctx, stageSync, 0);
            };
            Func<int, double> timeG = delegate(int g) {
                long a = Stopwatch.GetTimestamp();
                dispatch(ctx, (uint)g, 1, 1);
                drain();
                long b = Stopwatch.GetTimestamp();
                return (b - a) * 1000.0 / Stopwatch.Frequency;
            };

            // Calibrate the chain length so ONE group takes at least minT1Ms. Too
            // short and dispatch overhead hides the steps; this adapts to any clock.
            int hops = 2048;
            for (;;) {
                blob = Compile(PoolHlsl(T, S, hops));
                var bp = (DBlobPtr)Fn(blob, 3, typeof(DBlobPtr));
                var bs = (DBlobSize)Fn(blob, 4, typeof(DBlobSize));
                Check(createCs(dev, bp(blob), bs(blob), IntPtr.Zero, out cs), "CreateComputeShader");
                setShader(ctx, cs, IntPtr.Zero, 0);
                timeG(1);
                double a1 = timeG(1), a2 = timeG(1), a3 = timeG(1);
                double med = Math.Max(Math.Min(a1, a2), Math.Min(Math.Max(a1, a2), a3));
                if (med >= minT1Ms || hops >= (1 << 22) || med > 50.0) break;
                Rel(cs); cs = IntPtr.Zero; Rel(blob); blob = IntPtr.Zero;
                hops *= 2;
            }
            ps.Hops = hops;

            var grid = GridList(maxG);
            var gl = new List<int>();
            var med2 = new List<double>(); var mn = new List<double>(); var mx = new List<double>();
            double t1 = 0.0;
            foreach (int g in grid) {
                timeG(g);                                  // warm-up
                double[] v = new double[reps];
                for (int r = 0; r < reps; r++) v[r] = timeG(g);
                Array.Sort(v);
                double m = v[reps / 2];
                gl.Add(g); med2.Add(m); mn.Add(v[0]); mx.Add(v[reps - 1]);
                if (g == 1) t1 = m;
                if (t1 > 0.0 && m >= 3.3 * t1) break;      // three waves seen: enough
                if (m > maxDispatchSec * 1000.0) break;    // stay well clear of the 2 s TDR
            }
            ps.T1Ms = t1;
            ps.G = gl.ToArray(); ps.MedianMs = med2.ToArray(); ps.MinMs = mn.ToArray(); ps.MaxMs = mx.ToArray();

            // Verify the kernel really ran its chain: compare a few threads' results
            // with the same chain computed on the CPU. A failure does not discard the
            // curves; it flags them.
            bool ok = true;
            var note = new StringBuilder();
            int last = gl[gl.Count - 1];
            int[] vg = new int[] { 1, Math.Max(1, last / 2), last };
            for (int q = 0; q < vg.Length; q++) {
                int g = vg[q];
                dispatch(ctx, (uint)g, 1, 1);
                copyRes(ctx, stageOut, outBuf);
                MappedSub ms;
                Check(map(ctx, stageOut, 0, 1, 0, out ms), "Map(out)");
                try {
                    int[] gg = new int[] { 0, g / 2, g - 1 };
                    int[] tt = new int[] { 0, T / 2, T - 1 };
                    for (int p = 0; p < 3; p++) {
                        long ofs = ((long)gg[p] * T + tt[p]) * 4;
                        uint got = (uint)Marshal.ReadInt32(ms.pData, (int)ofs);
                        uint want = Expect((uint)tt[p], (uint)gg[p], (uint)hops, (uint)(S / 4));
                        if (got != want) {
                            ok = false;
                            note.Append("G=" + g + " group " + gg[p] + " thread " + tt[p] +
                                        ": got " + got + " want " + want + "; ");
                        }
                    }
                } finally { unmap(ctx, stageOut, 0); }
            }
            ps.Verified = ok;
            ps.VerifyNote = ok ? "chain output matched the CPU on 9 threads" : note.ToString();
        } finally {
            Rel(uavOut); Rel(uavSync); Rel(stageOut); Rel(stageSync); Rel(outBuf); Rel(syncBuf);
            Rel(cs); Rel(blob);
        }
        return ps;
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

$adapters = [IGpuPool]::Enumerate()
Write-Host ''
Write-Host 'Adapters' -ForegroundColor Cyan
Write-Host '--------'
foreach ($a in $adapters) {
    $tag = if ($a.IsSoftware) { ' [software]' } else { '' }
    '{0}: {1,-34} {2,-6} dev 0x{3:X4}  vram {4,6:N0} MB  shared {5,6:N0} MB{6}' -f `
        $a.Index, $a.Description.Trim(), (Get-VendorName $a.VendorId), $a.DeviceId,
        ($a.DedicatedVideoMemory / 1MB), ($a.SharedSystemMemory / 1MB), $tag | Write-Host
}

if ($Adapter -lt 0) {
    Write-Host ''
    Write-Host 'Choose one with -Adapter N and give its compute-unit count with -Units K.' -ForegroundColor Yellow
    Write-Host 'D3D11 cannot query the unit count; take it from the vendor spec.'
    exit 0
}
$chosen = $adapters | Where-Object { $_.Index -eq $Adapter }
if (-not $chosen) { Write-Host "No adapter $Adapter." -ForegroundColor Red; exit 1 }
if ($chosen.IsSoftware -and -not $IncludeSoftware) {
    Write-Host 'That is a software adapter; pass -IncludeSoftware to run it anyway.' -ForegroundColor Red; exit 1
}

if ($MaxGroups -le 0) {
    $MaxGroups = if ($Units -gt 0) { [Math]::Max(96, [Math]::Min(2048, $Units * 14)) } else { 600 }
}

# The shapes: the decoder's own threadgroup requests, an S sweep at its thread
# count, and a thread-count sweep in a small allocation that cannot bind.
$shapes = New-Object System.Collections.ArrayList
function Add-Shape($name, $t, $s) { [void]$shapes.Add(@{ Name = $name; T = $t; S = $s }) }
Add-Shape 'lmz_cuda_shared_dynamic'     192 31744   # what gate.py's constants describe
Add-Shape 'lmz_cuda_shared_full'        192 32768   # + its 1 KiB static starts array
Add-Shape 'lmz_metal_direct_compiled'   128 19456
if (-not $Quick) {
    Add-Shape 'lmz_metal_direct_source' 128 23552
    Add-Shape 'lmz_metal_perstream'      32 20992
    foreach ($s in 8192, 12288, 16384, 20480, 24576, 28672, 32768) { Add-Shape ("S_sweep_{0}" -f $s) 128 $s }
    foreach ($t in 64, 128, 256, 512, 1024) { Add-Shape ("T_sweep_{0}" -f $t) $t 1024 }
} else {
    Add-Shape 'S_sweep_24576' 128 24576
}

$ts = [int[]]($shapes | ForEach-Object { $_.T })
$ss = [int[]]($shapes | ForEach-Object { $_.S })
$names = [string[]]($shapes | ForEach-Object { $_.Name })

Write-Host ''
Write-Host ("Adapter {0}: {1}   units: {2}   shapes: {3}   max groups: {4}" -f `
    $chosen.Index, $chosen.Description.Trim(), $(if ($Units -gt 0) { $Units } else { 'unknown' }), $shapes.Count, $MaxGroups) -ForegroundColor Cyan
Write-Host 'Running. Close other GPU-heavy programs first; a sweep takes a minute or two.'
Write-Host '------------------------------------------------------------'

$start = Get-Date
$results = [IGpuPool]::RunPool($Adapter, $ts, $ss, $names, $MaxGroups, $Reps, $MinT1Ms, $MaxDispatchSec)

# A preliminary estimate only: the first G where the time steps up by >= 35% over
# the single-group baseline, taking the group before it as the wave-1 capacity.
function Get-WaveCap($g, $ms) {
    if ($g.Count -lt 6) { return $null }
    $base = (($ms[0..2] | Sort-Object)[1])
    for ($i = 1; $i -lt $g.Count; $i++) {
        if ($ms[$i] -ge 1.35 * $base) { return [int]$g[$i - 1] }
    }
    return $null
}

$table = @()
$out = @()
foreach ($r in $results) {
    if ($r.Error) {
        Write-Host ("  {0,-28} FAILED: {1}" -f $r.Name, $r.Error) -ForegroundColor Red
        $out += [ordered]@{ name = $r.Name; T = $r.T; S = $r.S; error = $r.Error }
        continue
    }
    $cap = Get-WaveCap $r.G $r.MedianMs
    $perUnit = $null; $poolLow = $null
    if ($cap -and $Units -gt 0) {
        $perUnit = [math]::Round($cap / $Units, 2)
        if (($cap % $Units) -eq 0) { $poolLow = [int]($perUnit * $r.S) }
    }
    $table += [pscustomobject]@{
        Shape        = $r.Name
        T            = $r.T
        'S (bytes)'  = $r.S
        'one-group ms' = [math]::Round($r.T1Ms, 2)
        'wave-1 cap G' = $cap
        'groups/unit'  = $perUnit
        'pool >= (B)'  = $poolLow
        Verified     = $r.Verified
    }
    $out += [ordered]@{
        name = $r.Name; T = $r.T; S = $r.S; hops = $r.Hops; t1_ms = $r.T1Ms
        verified = $r.Verified; verify_note = $r.VerifyNote; feature_level = $r.FeatureLevel
        preliminary_cap_groups = $cap
        G = @($r.G); median_ms = @($r.MedianMs); min_ms = @($r.MinMs); max_ms = @($r.MaxMs)
    }
}
Write-Host ''
$table | Format-Table -AutoSize

$video = $null
try { $video = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM, VideoProcessor } catch { }
$doc = [ordered]@{
    tool        = 'igpu-pool.ps1'
    schema      = 1
    started     = $start.ToString('o')
    finished    = (Get-Date).ToString('o')
    os          = [Environment]::OSVersion.VersionString
    powershell  = $PSVersionTable.PSVersion.ToString()
    cpu_threads = [Environment]::ProcessorCount
    adapter     = [ordered]@{
        index = $chosen.Index; description = $chosen.Description.Trim()
        vendor = (Get-VendorName $chosen.VendorId); device_id = ('0x{0:X4}' -f $chosen.DeviceId)
        dedicated_vram_mb = [int]($chosen.DedicatedVideoMemory / 1MB)
        shared_system_mb = [int]($chosen.SharedSystemMemory / 1MB)
        software = $chosen.IsSoftware
    }
    units_given_by_operator = $Units
    parameters  = [ordered]@{ reps = $Reps; min_t1_ms = $MinT1Ms; max_groups = $MaxGroups; max_dispatch_sec = $MaxDispatchSec; quick = [bool]$Quick }
    video_controllers = $video
    note = 'Raw curves. preliminary_cap_groups is a quick first-jump estimate only; analyse with probe/igpu-pool-analyze.py. units_given_by_operator was typed in, not queried.'
    shapes      = $out
}
if (-not $Json) { $Json = ('igpu-pool-{0}-{1}.json' -f $chosen.Index, (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$doc | ConvertTo-Json -Depth 8 | Set-Content -Path $Json -Encoding UTF8
Write-Host ("Wrote {0}" -f (Resolve-Path $Json))
Write-Host ''
Write-Host 'The table above is a PRELIMINARY estimate from ONE run. Run this script a second time later' -ForegroundColor Yellow
Write-Host '(different time, ideally after a reboot) and send BOTH JSON files back; they are the result.' -ForegroundColor Yellow
Write-Host 'If you ran this on a card with a known occupancy (e.g. -Units 84 on an RTX 5080), the'
Write-Host '(192, 31744) row should show a wave-1 cap near 252 (3 groups per SM): that is the calibration.'
