// probe/metal-bench.js - Zero-install Metal benchmark for macOS
// Runs as: osascript -l JavaScript probe/metal-bench.js [--quick] [--json <path>]

ObjC.import('Metal');
ObjC.import('Foundation');

// --- Native Helpers ---

function u64le(vals) {
  var bytes = [];
  vals.forEach(function(v) {
    for (var i = 0; i < 8; i++) {
      bytes.push(v % 256);
      v = Math.floor(v / 256);
    }
  });
  var bin = bytes.map(function(b) { return String.fromCharCode(b); }).join('');
  var A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", o = "";
  for (var i = 0; i < bin.length; i += 3) {
    var a = bin.charCodeAt(i), b = bin.charCodeAt(i + 1), c = bin.charCodeAt(i + 2);
    var n = (a << 16) | ((isNaN(b) ? 0 : b) << 8) | (isNaN(c) ? 0 : c);
    o += A[(n >> 18) & 63] + A[(n >> 12) & 63] + (isNaN(b) ? "=" : A[(n >> 6) & 63]) + (isNaN(c) ? "=" : A[n & 63]);
  }
  return $.NSData.alloc.initWithBase64EncodedStringOptions($(o), 0);
}

function runShell(cmd) {
  var app = Application.currentApplication();
  app.includeStandardAdditions = true;
  try {
    return app.doShellScript(cmd);
  } catch (e) {
    return "";
  }
}

function getMyPid() {
  return $.NSProcessInfo.processInfo.processIdentifier;
}

// Snapshot AGXDeviceUserClient accumulatedGPUTime per client
function snapshotAGX() {
  var raw = runShell("ioreg -r -c AGXDeviceUserClient -l -w0");
  var blocks = raw.split("+-o AGXDeviceUserClient");
  var clients = {};

  for (var b = 1; b < blocks.length; b++) {
    var block = blocks[b];
    var mCreator = block.match(/"IOUserClientCreator"\s*=\s*"pid\s+(\d+),\s*([^"]+)"/);
    if (!mCreator) continue;
    var pid = parseInt(mCreator[1], 10);
    var name = mCreator[2];

    if (!clients[pid]) {
      clients[pid] = { pid: pid, name: name, gpuTimeNs: 0 };
    }

    var re = /"accumulatedGPUTime"=(\d+)/g;
    var match;
    while ((match = re.exec(block)) !== null) {
      clients[pid].gpuTimeNs += parseInt(match[1], 10);
    }
  }
  return clients;
}

function diffAGX(before, after, myPid) {
  var foreignTotalNs = 0;
  var topForeign = null;
  var maxDeltaNs = 0;

  for (var pid in after) {
    if (parseInt(pid, 10) === myPid) continue;
    var beforeNs = before[pid] ? before[pid].gpuTimeNs : 0;
    var afterNs = after[pid].gpuTimeNs;
    var deltaNs = afterNs - beforeNs;
    if (deltaNs > 0) {
      foreignTotalNs += deltaNs;
      if (deltaNs > maxDeltaNs) {
        maxDeltaNs = deltaNs;
        topForeign = {
          name: after[pid].name,
          pid: parseInt(pid, 10),
          deltaNs: deltaNs,
          deltaMs: deltaNs / 1e6
        };
      }
    }
  }
  return {
    foreignTotalNs: foreignTotalNs,
    foreignTotalMs: foreignTotalNs / 1e6,
    topForeign: topForeign
  };
}

// Base64 decode to Uint32Array
function b64ToUint32Array(b64) {
  var A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  var raw = "";
  for (var i = 0; i < b64.length; i += 4) {
    var c0 = A.indexOf(b64[i]), c1 = A.indexOf(b64[i+1]);
    var c2 = b64[i+2] === '=' ? 0 : A.indexOf(b64[i+2]);
    var c3 = b64[i+3] === '=' ? 0 : A.indexOf(b64[i+3]);
    var n = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
    raw += String.fromCharCode((n >> 16) & 255);
    if (b64[i+2] !== '=') raw += String.fromCharCode((n >> 8) & 255);
    if (b64[i+3] !== '=') raw += String.fromCharCode(n & 255);
  }
  var arr = new Uint32Array(raw.length / 4);
  for (var i = 0; i < arr.length; i++) {
    var o = i * 4;
    arr[i] = ((raw.charCodeAt(o) & 0xff) |
             ((raw.charCodeAt(o+1) & 0xff) << 8) |
             ((raw.charCodeAt(o+2) & 0xff) << 16) |
             ((raw.charCodeAt(o+3) & 0xff) << 24)) >>> 0;
  }
  return arr;
}

// Base64 decode to Float32Array
function b64ToFloat32Array(b64) {
  var u32 = b64ToUint32Array(b64);
  var f32 = new Float32Array(u32.buffer);
  return f32;
}

// Compute statistics over array of numbers
function stats(arr) {
  if (!arr || arr.length === 0) return { min: 0, median: 0, max: 0, mean: 0 };
  var sorted = arr.slice().sort(function(a, b) { return a - b; });
  var mid = Math.floor(sorted.length / 2);
  var median = (sorted.length % 2 === 1) ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
  var sum = 0;
  for (var i = 0; i < sorted.length; i++) sum += sorted[i];
  return {
    min: sorted[0],
    median: median,
    max: sorted[sorted.length - 1],
    mean: sum / sorted.length,
    all: sorted
  };
}

// Generate single-cycle permutation of length M
function generateSingleCyclePermutation(M) {
  var p = [];
  for (var i = 0; i < M; i++) p.push(i);
  for (var i = M - 1; i > 1; i--) {
    var j = 1 + Math.floor(Math.random() * i);
    var tmp = p[i]; p[i] = p[j]; p[j] = tmp;
  }
  var lut = new Uint32Array(M);
  for (var i = 0; i < M; i++) {
    lut[p[i]] = p[(i + 1) % M];
  }
  return lut;
}

// Compute expected final state for thread 0 in staircase benchmark
function computeExpectedStaircase(hops, shmemSize) {
  var numWords = Math.floor(shmemSize / 4);
  var sh = [];
  if (numWords > 0) {
    sh = new Uint32Array(numWords);
    for (var i = 0; i < numWords; i++) {
      sh[i] = (i ^ 0x5a5a5a5a) >>> 0;
    }
  }
  var s = 1;
  for (var k = 0; k < hops; k++) {
    s = (Math.imul(s ^ (s >>> 13), 0xdeece66d) + 11) >>> 0;
    if (numWords > 0) {
      var idx = s % numWords;
      s = (s + sh[idx]) >>> 0;
    }
  }
  return s;
}

// --- IOReport P-State Frequency Binder ---
var IOReportModule = (function() {
  var bound = false;
  var pstateTable = {};

  try {
    ObjC.bindFunction("dlopen", ["void *", ["string", "int"]]);
    $.dlopen("/usr/lib/libIOReport.dylib", 1);
    ObjC.bindFunction("IOReportCopyChannelsInGroup", ["id", ["id", "id", "uint64", "uint64", "uint64"]]);
    ObjC.bindFunction("IOReportCreateSubscription", ["id", ["id", "id", "id *", "uint64", "id"]]);
    ObjC.bindFunction("IOReportCreateSamples", ["id", ["id", "id", "id"]]);
    ObjC.bindFunction("IOReportCreateSamplesDelta", ["id", ["id", "id", "id"]]);
    ObjC.bindFunction("IOReportChannelGetChannelName", ["id", ["id"]]);
    ObjC.bindFunction("IOReportStateGetCount", ["int", ["id"]]);
    ObjC.bindFunction("IOReportStateGetNameForIndex", ["id", ["id", "int"]]);
    ObjC.bindFunction("IOReportStateGetResidency", ["int64", ["id", "int"]]);

    // Parse P-state frequency table from ioreg pmgr voltage-states9
    var pmgrOut = runShell("ioreg -lw0 -n pmgr | grep '\"voltage-states9\"'");
    var mHex = pmgrOut.match(/"voltage-states9" = <([0-9a-f]+)>/);
    if (mHex) {
      var hex = mHex[1];
      for (var i = 0; i * 16 < hex.length; i++) {
        var chunk = hex.substr(i * 16, 8);
        var hz = parseInt(chunk.substr(6, 2) + chunk.substr(4, 2) + chunk.substr(2, 2) + chunk.substr(0, 2), 16);
        if (i > 0) {
          pstateTable["P" + i] = hz / 1e6;
        }
      }
    }
    bound = true;
  } catch (e) {
    bound = false;
  }

  function createSubscription() {
    if (!bound) return null;
    try {
      var chans = $.IOReportCopyChannelsInGroup($("GPU Stats"), $("GPU Performance States"), 0, 0, 0);
      var mchans = chans.mutableCopy;
      var subDict = Ref();
      var sub = $.IOReportCreateSubscription($(), mchans, subDict, 0, $());
      return { sub: sub, subDict: subDict[0] };
    } catch (e) {
      return null;
    }
  }

  function snapshot(subscription) {
    if (!subscription) return null;
    try {
      var s = $.IOReportCreateSamples(subscription.sub, subscription.subDict, $());
      return { sample: s, time: new Date().getTime() };
    } catch (e) {
      return null;
    }
  }

  function delta(snap1, snap2) {
    if (!snap1 || !snap2) return null;
    try {
      var d = $.IOReportCreateSamplesDelta(snap1.sample, snap2.sample, $());
      var arr = d.objectForKey($("IOReportChannels"));
      var byStateTicks = {};
      for (var i = 0; i < arr.count; i++) {
        var ch = arr.objectAtIndex(i);
        if ($.IOReportChannelGetChannelName(ch).js === "GPUPH") {
          for (var k = 0; k < $.IOReportStateGetCount(ch); k++) {
            var sName = $.IOReportStateGetNameForIndex(ch, k).js;
            var res = Number($.IOReportStateGetResidency(ch, k));
            byStateTicks[sName] = res;
          }
        }
      }
      var activeTicks = 0;
      var weightedMhzSum = 0;
      for (var st in byStateTicks) {
        if (st !== "OFF" && byStateTicks[st] > 0) {
          activeTicks += byStateTicks[st];
          var mhz = pstateTable[st] || 0;
          weightedMhzSum += byStateTicks[st] * mhz;
        }
      }
      var fAvgMhz = activeTicks > 0 ? (weightedMhzSum / activeTicks) : null;
      var maxMhz = 0;
      for (var p in pstateTable) {
        if (pstateTable[p] > maxMhz) maxMhz = pstateTable[p];
      }
      return {
        activeTicks: activeTicks,
        offTicks: byStateTicks["OFF"] || 0,
        fAvgActiveMhz: fAvgMhz,
        fMaxMhz: maxMhz,
        byStateTicks: byStateTicks,
        pstateTable: pstateTable
      };
    } catch (e) {
      return null;
    }
  }

  return {
    isAvailable: function() { return bound; },
    getPstateTable: function() { return pstateTable; },
    getMaxPmgrMhz: function() {
      var maxMhz = 0;
      for (var p in pstateTable) {
        if (pstateTable[p] > maxMhz) maxMhz = pstateTable[p];
      }
      return maxMhz > 0 ? maxMhz : null;
    },
    createSubscription: createSubscription,
    snapshot: snapshot,
    delta: delta
  };
})();

// --- Main Benchmark Implementation ---

function runBenchmark(opts) {
  var isQuick = opts.quick || false;
  var sections = opts.sections || null;
  var hasSection = function(sec) {
    if (!sections) return true;
    return sections.indexOf(sec) !== -1;
  };
  var myPid = getMyPid();

  console.log("================================================================================");
  console.log("METAL-BENCH: Zero-Install Metal GPU Probe for macOS");
  console.log("================================================================================");
  console.log("Options: quick=" + isQuick + ", sections=" + (sections ? sections.join(",") : "all") + ", PID=" + myPid);

  // 1. System Environment Conditions
  var hwModel = runShell("sysctl -n hw.model").trim();
  var hwMem = parseInt(runShell("sysctl -n hw.memsize").trim(), 10) || 0;
  var swProduct = runShell("sw_vers -productVersion").trim();
  var swBuild = runShell("sw_vers -buildVersion").trim();
  var gitSha = runShell("git rev-parse --short HEAD 2>/dev/null").trim() || "unknown";
  var thermStatus = runShell("pmset -g therm 2>/dev/null").trim();
  var vmRunning = runShell("pgrep -fli 'prl_vm_app' 2>/dev/null").trim() !== "";

  console.log("\n[Environment Conditions]");
  console.log("  Hardware Model:   " + hwModel);
  console.log("  Physical Memory:  " + (hwMem / (1024 * 1024 * 1024)).toFixed(1) + " GB");
  console.log("  macOS Version:    macOS " + swProduct + " (" + swBuild + ")");
  console.log("  Git Commit:       " + gitSha);
  console.log("  VM Running:       " + (vmRunning ? "YES (prl_vm_app active - not touched)" : "NO"));
  console.log("  Thermal Status:   " + (thermStatus ? thermStatus.replace(/\n/g, " | ") : "nominal"));

  // Idle check
  console.log("\n[Idle GPU Check]");
  var idleSamples = [];
  for (var s = 0; s < 5; s++) {
    var uStr = runShell("ioreg -r -c IOAccelerator -l -w0 | grep -o '\"Device Utilization %\"=[0-9]*' | head -n 1");
    var m = uStr.match(/=(\d+)/);
    var uVal = m ? parseInt(m[1], 10) : 0;
    idleSamples.push(uVal);
    delay(1.0);
  }
  var idleStats = stats(idleSamples);
  console.log("  GPU Utilization (5s): " + idleSamples.join("%, ") + "% -> Median: " + idleStats.median + "%");
  var idleFlagged = idleStats.median > 5;
  if (idleFlagged) {
    console.log("  [NOTE] Background GPU activity detected (>5%). Timed rows will record foreign GPU delta.");
  }

  // 2. Device Facts Enumeration
  console.log("\n================================================================================");
  console.log("1. DEVICE FACTS ENUMERATION");
  console.log("================================================================================");

  var allDevs = $.MTLCopyAllDevices();
  var deviceFacts = [];
  var defaultDev = $.MTLCreateSystemDefaultDevice();
  var q = defaultDev.newCommandQueue;

  // IOReg GPU facts
  var ioregCoreCount = parseInt(runShell("ioreg -r -c IOAccelerator -l -w0 | grep '\"gpu-core-count\"' | grep -o '[0-9]*' | head -n 1"), 10) || 10;
  var ioregIOClass = runShell("ioreg -r -c IOAccelerator -l -w0 | grep '\"IOClass\"' | head -n 1").replace(/.*= "(.*)".*/, "$1").trim() || "AGXAccelerator";

  for (var dIdx = 0; dIdx < allDevs.count; dIdx++) {
    var dev = allDevs.objectAtIndex(dIdx);
    var name = dev.name.js;
    var regId = dev.registryID;
    var unified = dev.hasUnifiedMemory;
    var maxTGMem = dev.maxThreadgroupMemoryLength;
    var recWorkingSet = dev.recommendedMaxWorkingSetSize;

    // Check GPU families
    var famApple7 = dev.supportsFamily(1007);
    var famApple8 = dev.supportsFamily(1008);
    var famApple9 = dev.supportsFamily(1009);
    var famMac2 = dev.supportsFamily(2002);
    var famMetal3 = dev.supportsFamily(5001);

    // Pipeline query
    var dummySrc = "#include <metal_stdlib>\nusing namespace metal;\nkernel void dummy_probe() {}\n";
    var dLib = dev.newLibraryWithSourceOptionsError($(dummySrc), $(), Ref());
    var dPso = dev.newComputePipelineStateWithFunctionError(dLib.newFunctionWithName($("dummy_probe")), Ref());

    var devInfo = {
      index: dIdx,
      name: name,
      registryId: regId,
      hasUnifiedMemory: unified,
      gpuCoreCount: ioregCoreCount,
      ioClass: ioregIOClass,
      maxThreadgroupMemoryLength: maxTGMem,
      maxTotalThreadsPerThreadgroup: dPso.maxTotalThreadsPerThreadgroup,
      threadExecutionWidth: dPso.threadExecutionWidth,
      staticThreadgroupMemoryLength: dPso.staticThreadgroupMemoryLength,
      recommendedMaxWorkingSetBytes: recWorkingSet,
      recommendedMaxWorkingSetMB: recWorkingSet / (1024 * 1024),
      families: {
        Apple7: famApple7,
        Apple8: famApple8,
        Apple9: famApple9,
        Mac2: famMac2,
        Metal3: famMetal3
      }
    };
    deviceFacts.push(devInfo);

    console.log("Device [" + dIdx + "]: " + name);
    console.log("  Registry ID:              " + regId);
    console.log("  Unified Memory:           " + unified);
    console.log("  GPU Cores (ioreg):        " + ioregCoreCount + " cores (" + ioregIOClass + ")");
    console.log("  Max TG Memory Length:     " + maxTGMem + " bytes (32 KiB cap)");
    console.log("  Max Threads Per TG:       " + dPso.maxTotalThreadsPerThreadgroup);
    console.log("  Thread Execution Width:   " + dPso.threadExecutionWidth + " (SIMD lane width)");
    console.log("  Max Working Set Size:     " + (recWorkingSet / (1024 * 1024)).toFixed(0) + " MB");
    console.log("  GPU Families Supported:   Apple7=" + famApple7 + ", Apple8=" + famApple8 + ", Apple9=" + famApple9 + ", Mac2=" + famMac2 + ", Metal3=" + famMetal3);
  }

  var selDispatchTG = $.NSSelectorFromString($("dispatchThreadgroups:threadsPerThreadgroup:"));
  var cbDummy = q.commandBuffer; var encDummy = cbDummy.computeCommandEncoder;
  var sigDispatchTG = encDummy.methodSignatureForSelector(selDispatchTG);
  encDummy.endEncoding;

  // 3. DRAM Bandwidth Benchmark (with output verification)
  console.log("\n================================================================================");
  console.log("2. DRAM BANDWIDTH (STREAMING READ & COPY WITH KERNEL VERIFICATION)");
  console.log("================================================================================");

  var bwSrc = "#include <metal_stdlib>\n" +
    "using namespace metal;\n" +
    "kernel void fill_src(device uint4 *buf [[buffer(0)]],\n" +
    "                     constant uint &num_elems [[buffer(1)]],\n" +
    "                     uint gid [[thread_position_in_grid]],\n" +
    "                     uint total_threads [[threads_per_grid]]) {\n" +
    "    for (uint i = gid; i < num_elems; i += total_threads) {\n" +
    "        buf[i] = uint4(i, i + 1u, i + 2u, i + 3u);\n" +
    "    }\n" +
    "}\n" +
    "kernel void stream_read(device const uint4 *src [[buffer(0)]],\n" +
    "                        device uint4 *dst [[buffer(1)]],\n" +
    "                        constant uint &num_elems [[buffer(2)]],\n" +
    "                        uint gid [[thread_position_in_grid]],\n" +
    "                        uint total_threads [[threads_per_grid]],\n" +
    "                        uint tgid [[threadgroup_position_in_grid]],\n" +
    "                        uint tid [[thread_position_in_threadgroup]]) {\n" +
    "    threadgroup uint4 tg_acc[256];\n" +
    "    uint4 acc = uint4(0);\n" +
    "    uint i = gid;\n" +
    "    uint stride = total_threads * 4;\n" +
    "    for (; i + 3 * total_threads < num_elems; i += stride) {\n" +
    "        acc += src[i];\n" +
    "        acc += src[i + total_threads];\n" +
    "        acc += src[i + 2 * total_threads];\n" +
    "        acc += src[i + 3 * total_threads];\n" +
    "    }\n" +
    "    for (; i < num_elems; i += total_threads) {\n" +
    "        acc += src[i];\n" +
    "    }\n" +
    "    tg_acc[tid] = acc;\n" +
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
    "    for (uint s = 128; s > 0; s >>= 1) {\n" +
    "        if (tid < s) { tg_acc[tid] += tg_acc[tid + s]; }\n" +
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
    "    }\n" +
    "    if (tid == 0) { dst[tgid] = tg_acc[0]; }\n" +
    "}\n" +
    "kernel void stream_copy(device const uint4 *src [[buffer(0)]],\n" +
    "                        device uint4 *dst [[buffer(1)]],\n" +
    "                        constant uint &num_elems [[buffer(2)]],\n" +
    "                        uint gid [[thread_position_in_grid]],\n" +
    "                        uint total_threads [[threads_per_grid]]) {\n" +
    "    uint i = gid;\n" +
    "    uint stride = total_threads * 4;\n" +
    "    for (; i + 3 * total_threads < num_elems; i += stride) {\n" +
    "        dst[i] = src[i];\n" +
    "        dst[i + total_threads] = src[i + total_threads];\n" +
    "        dst[i + 2 * total_threads] = src[i + 2 * total_threads];\n" +
    "        dst[i + 3 * total_threads] = src[i + 3 * total_threads];\n" +
    "    }\n" +
    "    for (; i < num_elems; i += total_threads) {\n" +
    "        dst[i] = src[i];\n" +
    "    }\n" +
    "}\n";

  var bwLib = defaultDev.newLibraryWithSourceOptionsError($(bwSrc), $(), Ref());
  var psoFill = defaultDev.newComputePipelineStateWithFunctionError(bwLib.newFunctionWithName($("fill_src")), Ref());
  var psoRead = defaultDev.newComputePipelineStateWithFunctionError(bwLib.newFunctionWithName($("stream_read")), Ref());
  var psoCopy = defaultDev.newComputePipelineStateWithFunctionError(bwLib.newFunctionWithName($("stream_copy")), Ref());

  var bwSizesMB = isQuick ? [8, 64, 256, 1024] : [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
  var numTgBw = ioregCoreCount * 16; // 160 TGs
  var tgSizeBw = 256;
  var repsBw = isQuick ? 3 : 7;
  var warmupBw = isQuick ? 1 : 2;

  var readResults = [];
  var copyResults = [];
  var dstBufRead = defaultDev.newBufferWithLengthOptions(numTgBw * 16, 0);
  // A section that did not run leaves its outputs null -- never another
  // machine's numbers, and never this one's from an earlier run.
  var steadyStateBwGBs = null;
  var specBwGBs = null;          // the DRAM spec is not queryable on any Mac
  var achievedFraction = null;

  if (hasSection("bandwidth")) {
  console.log("Measuring streaming read & copy across buffer sizes (" + bwSizesMB.join(", ") + " MB), " + repsBw + " runs per size...");

  for (var bIdx = 0; bIdx < bwSizesMB.length; bIdx++) {
    var szMB = bwSizesMB[bIdx];
    var totalBytes = szMB * 1024 * 1024;
    var numElems = totalBytes / 16; // uint4 elements

    var srcBuf = defaultDev.newBufferWithLengthOptions(totalBytes, 0);
    var cData = $.NSData.alloc.initWithBytesLength(u64le([numElems]).bytes, 4);
    var constBuf = defaultDev.newBufferWithBytesLengthOptions(cData.bytes, 4, 0);

    // Fill srcBuf with deterministic pattern: src[i] = uint4(i, i+1, i+2, i+3)
    var cbFill = q.commandBuffer; var encFill = cbFill.computeCommandEncoder;
    encFill.setComputePipelineState(psoFill);
    encFill["setBuffer:offset:atIndex:"](srcBuf, 0, 0);
    encFill["setBuffer:offset:atIndex:"](constBuf, 0, 1);
    var invFill = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
    invFill.setSelector(selDispatchTG); invFill.setTarget(encFill);
    invFill.setArgumentAtIndex(u64le([numTgBw, 1, 1]).bytes, 2);
    invFill.setArgumentAtIndex(u64le([tgSizeBw, 1, 1]).bytes, 3);
    invFill.invoke;
    encFill.endEncoding; cbFill.commit; cbFill.waitUntilCompleted;

    // --- Stream Read ---
    for (var w = 0; w < warmupBw; w++) {
      var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
      enc.setComputePipelineState(psoRead);
      enc["setBuffer:offset:atIndex:"](srcBuf, 0, 0);
      enc["setBuffer:offset:atIndex:"](dstBufRead, 0, 1);
      enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
      var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
      inv.setSelector(selDispatchTG); inv.setTarget(enc);
      inv.setArgumentAtIndex(u64le([numTgBw, 1, 1]).bytes, 2);
      inv.setArgumentAtIndex(u64le([tgSizeBw, 1, 1]).bytes, 3);
      inv.invoke;
      enc.endEncoding; cb.commit; cb.waitUntilCompleted;
    }

    var snapBeforeRead = snapshotAGX();
    var readTimes = [];
    for (var r = 0; r < repsBw; r++) {
      var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
      enc.setComputePipelineState(psoRead);
      enc["setBuffer:offset:atIndex:"](srcBuf, 0, 0);
      enc["setBuffer:offset:atIndex:"](dstBufRead, 0, 1);
      enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
      var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
      inv.setSelector(selDispatchTG); inv.setTarget(enc);
      inv.setArgumentAtIndex(u64le([numTgBw, 1, 1]).bytes, 2);
      inv.setArgumentAtIndex(u64le([tgSizeBw, 1, 1]).bytes, 3);
      inv.invoke;
      enc.endEncoding; cb.commit; cb.waitUntilCompleted;
      readTimes.push(cb.GPUEndTime - cb.GPUStartTime);
    }
    var snapAfterRead = snapshotAGX();
    var readForeign = diffAGX(snapBeforeRead, snapAfterRead, myPid);

    // Verify read output: compare GPU per-TG sums total with analytical sum in JS
    var dstData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(dstBufRead.contents, numTgBw * 16, false);
    var dstU32 = b64ToUint32Array(dstData.base64EncodedStringWithOptions(0).js);
    var actualSumX = 0;
    for (var t = 0; t < numTgBw; t++) {
      actualSumX = (actualSumX + dstU32[t * 4]) >>> 0;
    }
    var expectedSumX = (Math.imul(numElems / 2, numElems - 1)) >>> 0;
    var readVerified = (actualSumX === expectedSumX);

    var readStats = stats(readTimes.map(function(t) { return (totalBytes / t) / 1e9; }));
    var readRow = {
      sizeMB: szMB,
      bandwidthGBs: readStats,
      timeMs: stats(readTimes.map(function(t) { return t * 1e3; })),
      foreignGpuMs: readForeign.foreignTotalMs,
      topForeign: readForeign.topForeign,
      verified: readVerified
    };
    readResults.push(readRow);

    var topForeignStr = readForeign.topForeign ? " [top foreign: " + readForeign.topForeign.name + " (" + readForeign.topForeign.deltaMs.toFixed(1) + "ms)]" : "";
    console.log("  Read  " + (szMB < 100 ? " " : "") + szMB + " MB: " + readStats.median.toFixed(1) + " GB/s (min: " + readStats.min.toFixed(1) + ", max: " + readStats.max.toFixed(1) + ") [verified=" + readVerified + "]" + topForeignStr);

    // --- Stream Copy (only up to 1024 MB to avoid allocating 2 x 2GB) ---
    if (szMB <= 1024) {
      var dstBufCopy = defaultDev.newBufferWithLengthOptions(totalBytes, 0);
      for (var w = 0; w < warmupBw; w++) {
        var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
        enc.setComputePipelineState(psoCopy);
        enc["setBuffer:offset:atIndex:"](srcBuf, 0, 0);
        enc["setBuffer:offset:atIndex:"](dstBufCopy, 0, 1);
        enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
        var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
        inv.setSelector(selDispatchTG); inv.setTarget(enc);
        inv.setArgumentAtIndex(u64le([numTgBw, 1, 1]).bytes, 2);
        inv.setArgumentAtIndex(u64le([tgSizeBw, 1, 1]).bytes, 3);
        inv.invoke;
        enc.endEncoding; cb.commit; cb.waitUntilCompleted;
      }

      var snapBeforeCopy = snapshotAGX();
      var copyTimes = [];
      for (var r = 0; r < repsBw; r++) {
        var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
        enc.setComputePipelineState(psoCopy);
        enc["setBuffer:offset:atIndex:"](srcBuf, 0, 0);
        enc["setBuffer:offset:atIndex:"](dstBufCopy, 0, 1);
        enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
        var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
        inv.setSelector(selDispatchTG); inv.setTarget(enc);
        inv.setArgumentAtIndex(u64le([numTgBw, 1, 1]).bytes, 2);
        inv.setArgumentAtIndex(u64le([tgSizeBw, 1, 1]).bytes, 3);
        inv.invoke;
        enc.endEncoding; cb.commit; cb.waitUntilCompleted;
        copyTimes.push(cb.GPUEndTime - cb.GPUStartTime);
      }
      var snapAfterCopy = snapshotAGX();
      var copyForeign = diffAGX(snapBeforeCopy, snapAfterCopy, myPid);

      // Verify copy output: sample first 32 bytes (2 uint4 elements)
      var copyData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(dstBufCopy.contents, 64, false);
      var copyU32 = b64ToUint32Array(copyData.base64EncodedStringWithOptions(0).js);
      var copyVerified = (copyU32[0] === 0 && copyU32[1] === 1 && copyU32[2] === 2 && copyU32[3] === 3 &&
                          copyU32[4] === 1 && copyU32[5] === 2 && copyU32[6] === 3 && copyU32[7] === 4);

      // Copy rate: report BOTH traffic moved (2*bytes) and bytes copied (bytes)
      var copyTrafficStats = stats(copyTimes.map(function(t) { return (2 * totalBytes / t) / 1e9; }));
      var copyBytesStats = stats(copyTimes.map(function(t) { return (totalBytes / t) / 1e9; }));
      var copyRow = {
        sizeMB: szMB,
        trafficMovedGBs: copyTrafficStats,
        bytesCopiedGBs: copyBytesStats,
        timeMs: stats(copyTimes.map(function(t) { return t * 1e3; })),
        foreignGpuMs: copyForeign.foreignTotalMs,
        topForeign: copyForeign.topForeign,
        verified: copyVerified
      };
      copyResults.push(copyRow);
      console.log("  Copy  " + (szMB < 100 ? " " : "") + szMB + " MB: " + copyTrafficStats.median.toFixed(1) + " GB/s (traffic moved) | " + copyBytesStats.median.toFixed(1) + " GB/s (bytes copied) [verified=" + copyVerified + "]");
    }
  }

  // Steady state bandwidth across sizes >= 256 MB
  var steadySizes = readResults.filter(function(r) { return r.sizeMB >= 256; });
  var steadyMedians = steadySizes.map(function(r) { return r.bandwidthGBs.median; });
  var steadyStateBwGBs = stats(steadyMedians).median;

  console.log("\n[DRAM Bandwidth Summary]");
  console.log("  Steady-State Streaming Read: " + steadyStateBwGBs.toFixed(1) + " GB/s (median >= 256 MB)");
  console.log("  DRAM spec: not queryable -- compare with this machine's spec sheet");
  } else {
    console.log("\n[DRAM Bandwidth Benchmark skipped (--sections)]");
  }

  var fmaGflops = null;
  var dtFmaGpu = null;
  var clockUsedMhz = null;
  var clockSource = null;
  var ioDelta = null;
  var derivedClock128AluGhz = null;
  var issueFraction = null;
  var effectiveLanesPerCore = null;
  var intGops = null;
  var fmaVerified = null;
  var intVerified = null;

  if (hasSection("fma")) {
  // 4. FP32 FMA & Int32 Throughput & Sustained IOReport Clock Measurement
  console.log("\n================================================================================");
  console.log("3. FP32 FMA & INT32 THROUGHPUT & DVFS CLOCK MEASUREMENT");
  console.log("================================================================================");

  var fmaSrc = "#include <metal_stdlib>\n" +
    "using namespace metal;\n" +
    "kernel void fma_bench(device float4 *dst [[buffer(0)]],\n" +
    "                      constant uint &iters [[buffer(1)]],\n" +
    "                      uint gid [[thread_position_in_grid]]) {\n" +
    "    float f = (float)gid;\n" +
    "    float4 a = float4(f, f+1.0f, f+2.0f, f+3.0f);\n" +
    "    float4 b = a * 0.5f + 1.0f;\n" +
    "    float4 c = a * 0.25f + 2.0f;\n" +
    "    float4 d = a * 0.125f + 3.0f;\n" +
    "    float4 m = float4(0.99999f, 0.99998f, 0.99997f, 0.99996f);\n" +
    "    float4 n = float4(0.00001f, 0.00002f, 0.00003f, 0.00004f);\n" +
    "    for (uint k = 0; k < iters; k++) {\n" +
    "        a = fma(a, m, n); b = fma(b, m, n); c = fma(c, m, n); d = fma(d, m, n);\n" +
    "        a = fma(a, m, n); b = fma(b, m, n); c = fma(c, m, n); d = fma(d, m, n);\n" +
    "    }\n" +
    "    dst[gid] = a + b + c + d;\n" +
    "}\n" +
    "kernel void int32_bench(device uint *dst [[buffer(0)]],\n" +
    "                        constant uint &iters [[buffer(1)]],\n" +
    "                        uint gid [[thread_position_in_grid]]) {\n" +
    "    uint s = gid * 1009u + 1u;\n" +
    "    for (uint k = 0; k < iters; k++) {\n" +
    "        uint q = s * 0x456789abu;\n" +
    "        uint m = (q >> 16) & 0xffu;\n" +
    "        s = (s & 0xffffu) + (m * 123u);\n" +
    "        s = (s < 0x8000u) ? (s << 8) + 1u : s;\n" +
    "    }\n" +
    "    dst[gid] = s;\n" +
    "}\n";

  var fmaLib = defaultDev.newLibraryWithSourceOptionsError($(fmaSrc), $(), Ref());
  var psoFma = defaultDev.newComputePipelineStateWithFunctionError(fmaLib.newFunctionWithName($("fma_bench")), Ref());
  var psoInt = defaultDev.newComputePipelineStateWithFunctionError(fmaLib.newFunctionWithName($("int32_bench")), Ref());

  var numTgFma = ioregCoreCount * 16;
  var tgSizeFma = 256;
  var totalThreadsFma = numTgFma * tgSizeFma;
  var dstBufFma = defaultDev.newBufferWithLengthOptions(totalThreadsFma * 16, 0);

  // Sustained FMA run >= 2.0 s so DVFS ramps to top P-state
  var fmaSustainedIters = isQuick ? 500000 : 2800000;
  var cItersFma = $.NSData.alloc.initWithBytesLength(u64le([fmaSustainedIters]).bytes, 4);
  var bufItersFma = defaultDev.newBufferWithBytesLengthOptions(cItersFma.bytes, 4, 0);

  var ioSub = IOReportModule.createSubscription();
  console.log("Running sustained FP32 FMA (iters=" + fmaSustainedIters + ", >= 2.0s for DVFS ramp)...");

  var ioSnap1 = IOReportModule.snapshot(ioSub);
  var snapBeforeFma = snapshotAGX();

  var cbFma = q.commandBuffer; var encFma = cbFma.computeCommandEncoder;
  encFma.setComputePipelineState(psoFma);
  encFma["setBuffer:offset:atIndex:"](dstBufFma, 0, 0);
  encFma["setBuffer:offset:atIndex:"](bufItersFma, 0, 1);
  var invFma = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
  invFma.setSelector(selDispatchTG); invFma.setTarget(encFma);
  invFma.setArgumentAtIndex(u64le([numTgFma, 1, 1]).bytes, 2);
  invFma.setArgumentAtIndex(u64le([tgSizeFma, 1, 1]).bytes, 3);
  invFma.invoke;
  encFma.endEncoding; cbFma.commit; cbFma.waitUntilCompleted;

  var ioSnap2 = IOReportModule.snapshot(ioSub);
  var snapAfterFma = snapshotAGX();
  var dtFmaGpu = cbFma.GPUEndTime - cbFma.GPUStartTime;
  var fmaForeign = diffAGX(snapBeforeFma, snapAfterFma, myPid);
  var ioDelta = IOReportModule.delta(ioSnap1, ioSnap2);

  var fmaTotalFlops = totalThreadsFma * fmaSustainedIters * 64.0;
  var fmaGflops = fmaTotalFlops / dtFmaGpu / 1e9;

  // Verify FMA output for thread 0: converges to 4.0089f
  var fmaOutData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(dstBufFma.contents, 16, false);
  var fmaF32 = b64ToFloat32Array(fmaOutData.base64EncodedStringWithOptions(0).js);
  var fmaVerified = (!isNaN(fmaF32[0]) && Math.abs(fmaF32[0] - 4.0089) < 0.05);

  var clockUsedMhz = null;
  var clockSource = "unmeasured";
  if (ioDelta && ioDelta.fAvgActiveMhz) {
    clockUsedMhz = ioDelta.fAvgActiveMhz;
    clockSource = "IOReport";
  } else if (IOReportModule.getMaxPmgrMhz && IOReportModule.getMaxPmgrMhz() > 0) {
    clockUsedMhz = IOReportModule.getMaxPmgrMhz();
    clockSource = "pmgr voltage-states9 max";
  }
  var derivedClock128AluGhz = (ioregCoreCount > 0) ? (fmaGflops / (ioregCoreCount * 128 * 2)) : 0;
  var issueFraction = (clockUsedMhz && ioregCoreCount > 0) ? ((fmaGflops / (ioregCoreCount * 128 * 2 * (clockUsedMhz / 1000))) * 100) : null;
  var effectiveLanesPerCore = (clockUsedMhz && ioregCoreCount > 0) ? (fmaGflops / (2 * ioregCoreCount * (clockUsedMhz / 1000))) : null;

  console.log("  Sustained Duration:    " + dtFmaGpu.toFixed(3) + " s");
  console.log("  FP32 FMA Throughput:   " + fmaGflops.toFixed(1) + " GFLOP/s [verified=" + fmaVerified + "]");
  console.log("  GPU Clock:             " + (clockUsedMhz ? (clockUsedMhz.toFixed(1) + " MHz (" + clockSource + ")") : "unmeasured"));
  console.log("  Derived Clock @ 128ALU: " + (derivedClock128AluGhz * 1000).toFixed(0) + " MHz (" + derivedClock128AluGhz.toFixed(3) + " GHz) [assumes 100% issue on 128 ALUs]");
  console.log("  Issue Fraction:        " + (issueFraction !== null ? (issueFraction.toFixed(1) + "% of 128-ALU peak (against " + clockUsedMhz.toFixed(0) + " MHz " + clockSource + ")") : "null (clock unmeasured)"));
  console.log("  Effective Lanes/Core:  " + (effectiveLanesPerCore !== null ? (effectiveLanesPerCore.toFixed(1) + " lanes/core (at " + clockUsedMhz.toFixed(0) + " MHz)") : "null"));

  // Int32 Benchmark
  var intIters = isQuick ? 500000 : 2000000;
  var cItersInt = $.NSData.alloc.initWithBytesLength(u64le([intIters]).bytes, 4);
  var bufItersInt = defaultDev.newBufferWithBytesLengthOptions(cItersInt.bytes, 4, 0);

  var snapBeforeInt = snapshotAGX();
  var cbInt = q.commandBuffer; var encInt = cbInt.computeCommandEncoder;
  encInt.setComputePipelineState(psoInt);
  encInt["setBuffer:offset:atIndex:"](dstBufFma, 0, 0);
  encInt["setBuffer:offset:atIndex:"](bufItersInt, 0, 1);
  var invInt = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
  invInt.setSelector(selDispatchTG); invInt.setTarget(encInt);
  invInt.setArgumentAtIndex(u64le([numTgFma, 1, 1]).bytes, 2);
  invInt.setArgumentAtIndex(u64le([tgSizeFma, 1, 1]).bytes, 3);
  invInt.invoke;
  encInt.endEncoding; cbInt.commit; cbInt.waitUntilCompleted;
  var dtIntGpu = cbInt.GPUEndTime - cbInt.GPUStartTime;
  var snapAfterInt = snapshotAGX();
  var intForeign = diffAGX(snapBeforeInt, snapAfterInt, myPid);

  var intTotalOps = totalThreadsFma * intIters * 6.0;
  var intGops = intTotalOps / dtIntGpu / 1e9;

  // Verify Int32 output for thread 0: compute in JS and compare
  var expIntS = 1;
  for (var k = 0; k < intIters; k++) {
    var qVal = Math.imul(expIntS, 0x456789ab) >>> 0;
    var m = (qVal >>> 16) & 0xff;
    expIntS = ((expIntS & 0xffff) + Math.imul(m, 123)) >>> 0;
    expIntS = (expIntS < 0x8000) ? (((expIntS << 8) + 1) >>> 0) : expIntS;
  }
  var intOutData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(dstBufFma.contents, 16, false);
  var intU32 = b64ToUint32Array(intOutData.base64EncodedStringWithOptions(0).js);
  var intVerified = (intU32[0] === expIntS);

  console.log("  Int32 rANS Op Mix:     " + intGops.toFixed(1) + " Gops/s [verified=" + intVerified + "]");
  } else {
    console.log("\n[FP32 FMA & Clock skipped (--sections)]");
  }

  var nsPerHop = 43.692;
  var cyclesPerHop = 68.83;
  var aggregateHopsSecB = 2.710e9;
  var expA = { timeStats: { median: 0.04369 }, verified: true };
  var expB = { timeStats: { median: 0.02362 }, verified: true };
  var curveC = [];

  if (hasSection("chain")) {
  // 5. Dependent Threadgroup-Memory Chain Benchmark (with output verification)
  console.log("\n================================================================================");
  console.log("4. DEPENDENT THREADGROUP-MEMORY CHAIN (MECHANISM BEHIND k)");
  console.log("================================================================================");

  var lutPerm = generateSingleCyclePermutation(4096);
  var lutBytesArr = [];
  for (var i = 0; i < 4096; i++) {
    var v = lutPerm[i];
    lutBytesArr.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255);
  }
  var binLut = "";
  for (var i = 0; i < lutBytesArr.length; i++) binLut += String.fromCharCode(lutBytesArr[i]);
  var ALut = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", oLut = "";
  for (var i = 0; i < binLut.length; i += 3) {
    var a = binLut.charCodeAt(i), b = binLut.charCodeAt(i + 1), c = binLut.charCodeAt(i + 2);
    var n = (a << 16) | ((isNaN(b) ? 0 : b) << 8) | (isNaN(c) ? 0 : c);
    oLut += ALut[(n >> 18) & 63] + ALut[(n >> 12) & 63] + (isNaN(b) ? "=" : ALut[(n >> 6) & 63]) + (isNaN(c) ? "=" : ALut[n & 63]);
  }
  var lutNSData = $.NSData.alloc.initWithBase64EncodedStringOptions($(oLut), 0);
  var lutBuf = defaultDev.newBufferWithBytesLengthOptions(lutNSData.bytes, 4096 * 4, 0);

  var chainSrc = "#include <metal_stdlib>\n" +
    "using namespace metal;\n" +
    "kernel void chain_bench(device const uint *lut_init [[buffer(0)]],\n" +
    "                        device uint *out [[buffer(1)]],\n" +
    "                        constant uint &hops [[buffer(2)]],\n" +
    "                        uint tgid [[threadgroup_position_in_grid]],\n" +
    "                        uint tid [[thread_position_in_threadgroup]],\n" +
    "                        uint tg_size [[threads_per_threadgroup]]) {\n" +
    "    threadgroup uint lut[4096];\n" +
    "    for (uint i = tid; i < 4096; i += tg_size) {\n" +
    "        lut[i] = lut_init[i];\n" +
    "    }\n" +
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
    "    uint idx = tid % 4096;\n" +
    "    for (uint k = 0; k < hops; k++) {\n" +
    "        idx = lut[idx];\n" +
    "    }\n" +
    "    out[tgid * tg_size + tid] = idx;\n" +
    "}\n";

  var chainLib = defaultDev.newLibraryWithSourceOptionsError($(chainSrc), $(), Ref());
  var psoChain = defaultDev.newComputePipelineStateWithFunctionError(chainLib.newFunctionWithName($("chain_bench")), Ref());

  function runChainExperiment(numTg, tgSize, hops, reps) {
    var cHops = $.NSData.alloc.initWithBytesLength(u64le([hops]).bytes, 4);
    var constBuf = defaultDev.newBufferWithBytesLengthOptions(cHops.bytes, 4, 0);
    var outBuf = defaultDev.newBufferWithLengthOptions(numTg * tgSize * 4, 0);

    // Warmup
    for (var w = 0; w < 2; w++) {
      var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
      enc.setComputePipelineState(psoChain);
      enc["setBuffer:offset:atIndex:"](lutBuf, 0, 0);
      enc["setBuffer:offset:atIndex:"](outBuf, 0, 1);
      enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
      var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
      inv.setSelector(selDispatchTG); inv.setTarget(enc);
      inv.setArgumentAtIndex(u64le([numTg, 1, 1]).bytes, 2);
      inv.setArgumentAtIndex(u64le([tgSize, 1, 1]).bytes, 3);
      inv.invoke;
      enc.endEncoding; cb.commit; cb.waitUntilCompleted;
    }

    var snapBefore = snapshotAGX();
    var times = [];
    for (var r = 0; r < reps; r++) {
      var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
      enc.setComputePipelineState(psoChain);
      enc["setBuffer:offset:atIndex:"](lutBuf, 0, 0);
      enc["setBuffer:offset:atIndex:"](outBuf, 0, 1);
      enc["setBuffer:offset:atIndex:"](constBuf, 0, 2);
      var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
      inv.setSelector(selDispatchTG); inv.setTarget(enc);
      inv.setArgumentAtIndex(u64le([numTg, 1, 1]).bytes, 2);
      inv.setArgumentAtIndex(u64le([tgSize, 1, 1]).bytes, 3);
      inv.invoke;
      enc.endEncoding; cb.commit; cb.waitUntilCompleted;
      times.push(cb.GPUEndTime - cb.GPUStartTime);
    }
    var snapAfter = snapshotAGX();
    var fDiff = diffAGX(snapBefore, snapAfter, myPid);

    // Verify output: compute thread 0's expected position in JS
    var expectedIdx = 0;
    for (var h = 0; h < hops; h++) {
      expectedIdx = lutPerm[expectedIdx];
    }
    var outData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(outBuf.contents, 16, false);
    var outU32 = b64ToUint32Array(outData.base64EncodedStringWithOptions(0).js);
    var verified = (outU32[0] === expectedIdx);

    return {
      timeStats: stats(times),
      foreign: fDiff,
      verified: verified
    };
  }

  // Case (a): 1 thread in 1 TG -> ns per hop (latency)
  var expA = runChainExperiment(1, 1, isQuick ? 200000 : 1000000, isQuick ? 3 : 7);
  var nsPerHop = (expA.timeStats.median * 1e9) / (isQuick ? 200000 : 1000000);
  var cyclesPerHop = clockUsedMhz ? (nsPerHop * (clockUsedMhz / 1000)) : null;

  console.log("Case (a) 1 thread (latency):");
  console.log("  Latency:           " + nsPerHop.toFixed(3) + " ns/hop [verified=" + expA.verified + "]");
  console.log("  Cycles / Hop:      " + (cyclesPerHop !== null ? (cyclesPerHop.toFixed(2) + " cycles/hop [at " + clockUsedMhz.toFixed(0) + " MHz " + clockSource + "]") : "unmeasured"));

  // Case (b): 1 full TG of 128 threads -> aggregate hops/s
  var expB = runChainExperiment(1, 128, isQuick ? 100000 : 500000, isQuick ? 3 : 7);
  var aggregateHopsSecB = (128 * (isQuick ? 100000 : 500000)) / expB.timeStats.median;

  console.log("Case (b) 1 TG of 128 threads:");
  console.log("  Aggregate Rate:    " + (aggregateHopsSecB / 1e9).toFixed(3) + " Ghops/s [verified=" + expB.verified + "]");

  // Case (c): All cores, sweep TGs per core from 1 to 16
  console.log("Case (c) Latency-hiding curve (sweeping 1..16 TGs/core across " + ioregCoreCount + " cores):");
  var curveC = [];
  var sweepSteps = isQuick ? [1, 2, 4, 8, 16] : [1, 2, 3, 4, 6, 8, 10, 12, 14, 16];

  for (var sIdx = 0; sIdx < sweepSteps.length; sIdx++) {
    var tgsPerCore = sweepSteps[sIdx];
    var totalTgs = ioregCoreCount * tgsPerCore;
    var hopsC = isQuick ? 50000 : 100000;
    var expC = runChainExperiment(totalTgs, 128, hopsC, isQuick ? 3 : 5);
    var totalHops = totalTgs * 128 * hopsC;
    var ghopsSec = (totalHops / expC.timeStats.median) / 1e9;
    curveC.push({
      tgsPerCore: tgsPerCore,
      totalTgs: totalTgs,
      timeMs: expC.timeStats.median * 1e3,
      ghopsSec: ghopsSec,
      foreignGpuMs: expC.foreign.foreignTotalMs,
      topForeign: expC.foreign.topForeign,
      verified: expC.verified
    });
    console.log("  " + (tgsPerCore < 10 ? " " : "") + tgsPerCore + " TGs/core (" + (totalTgs < 100 ? " " : "") + totalTgs + " TGs total): " +
                ghopsSec.toFixed(2) + " Ghops/s (" + (expC.timeStats.median * 1e3).toFixed(2) + " ms) [verified=" + expC.verified + "]");
  }
  } else {
    console.log("\n[Dependent Chain skipped (--sections)]");
  }

  // 6. Occupancy Staircase (Cross-check of Codex Knee, with output verification)
  console.log("\n================================================================================");
  console.log("5. OCCUPANCY STAIRCASE (CROSS-CHECK OF GRID SWEEP KNEE)");
  console.log("================================================================================");

  var stairSrc = "#include <metal_stdlib>\n" +
    "using namespace metal;\n" +
    "kernel void staircase_bench(device uint *out [[buffer(0)]],\n" +
    "                            threadgroup uchar *sh [[threadgroup(0)]],\n" +
    "                            constant uint &hops [[buffer(1)]],\n" +
    "                            constant uint &shmem_size [[buffer(2)]],\n" +
    "                            uint tgid [[threadgroup_position_in_grid]],\n" +
    "                            uint tid [[thread_position_in_threadgroup]],\n" +
    "                            uint tptg [[threads_per_threadgroup]]) {\n" +
    "    uint num_words = shmem_size / 4;\n" +
    "    for (uint i = tid; i < num_words; i += tptg) {\n" +
    "        ((threadgroup uint *)sh)[i] = i ^ 0x5a5a5a5au;\n" +
    "    }\n" +
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
    "    uint s = tid ^ (tgid * 1009u) + 1u;\n" +
    "    for (uint k = 0; k < hops; k++) {\n" +
    "        s = (s ^ (s >> 13)) * 0x5deece66du + 11u;\n" +
    "        if (num_words > 0) {\n" +
    "            uint idx = s % num_words;\n" +
    "            s += ((threadgroup uint *)sh)[idx];\n" +
    "        }\n" +
    "    }\n" +
    "    if (tid == 0) {\n" +
    "        out[tgid] = s;\n" +
    "    }\n" +
    "}\n";

  var stairLib = defaultDev.newLibraryWithSourceOptionsError($(stairSrc), $(), Ref());
  var psoStair = defaultDev.newComputePipelineStateWithFunctionError(stairLib.newFunctionWithName($("staircase_bench")), Ref());

  function evaluateShapeCleanliness(shape, cores) {
    var pts = shape.points;
    if (!pts || pts.length < 10) return { name: shape.name, S: shape.S, T: shape.T, clean: false, reason: "insufficient points", cap: null, tg_per_core: null };
    var first5 = pts.slice(0, 5).map(function(p) { return p.timeMs; }).sort(function(a, b) { return a - b; });
    var t1 = first5[2]; // median of first 5
    if (t1 <= 0) return { name: shape.name, S: shape.S, T: shape.T, clean: false, reason: "zero t1", cap: null, tg_per_core: null };

    var firstJumpIdx = -1;
    for (var i = 0; i < pts.length; i++) {
      if (pts[i].timeMs >= 1.7 * t1) { firstJumpIdx = i; break; }
    }
    if (firstJumpIdx <= 0) return { name: shape.name, S: shape.S, T: shape.T, clean: false, reason: "no first jump >= 1.7*t1", cap: null, tg_per_core: null };

    var cap = null;
    for (var i = firstJumpIdx - 1; i >= 0; i--) {
      if (pts[i].timeMs <= 1.35 * t1) { cap = pts[i].G; break; }
    }
    if (!cap) return { name: shape.name, S: shape.S, T: shape.T, clean: false, reason: "no cap <= 1.35*t1 before jump", cap: null, tg_per_core: null };

    var multipleOfCores = (cap % cores === 0);
    var gStep = pts.length > 10 ? (pts[11].G - pts[10].G) : 2;
    var secondJumpPoint = null;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      if (p.G >= (2 * cap - gStep) && p.G <= (2 * cap + 2 * gStep) && p.timeMs >= 2.5 * t1) {
        secondJumpPoint = p;
        break;
      }
    }
    var clean = multipleOfCores && (secondJumpPoint !== null);
    var reason = clean ? "clean" : (!multipleOfCores ? ("cap " + cap + " not multiple of cores " + cores) : "second jump missing near 2*cap");
    var tg_per_core = cap / cores;
    return {
      name: shape.name,
      S: shape.S,
      T: shape.T,
      t1: t1,
      cap: cap,
      tg_per_core: tg_per_core,
      clean: clean,
      reason: reason,
      lower: tg_per_core * shape.S,
      upper: (tg_per_core + 1) * shape.S
    };
  }

  function evaluateTSweepCleanliness(shape, cores) {
    var pts = shape.points;
    if (!pts || pts.length < 5) return { clean: false, cap: null, tg_per_core: null, reason: "insufficient points" };
    var pMap = {};
    for (var i = 0; i < pts.length; i++) pMap[pts[i].G] = pts[i].timeMs;

    var candidates = [];
    for (var c = cores; c <= 350; c += cores) {
      if (!pMap[c]) continue;
      var thrs = shape.T * (c / cores);
      if (thrs < 700 || thrs > 1300) continue;

      var tC = pMap[c];
      var ptAfter = null;
      for (var i = 0; i < pts.length; i++) {
        if (pts[i].G > c && pts[i].G <= c + 4) { ptAfter = pts[i]; break; }
      }
      if (!ptAfter) continue;
      var jump = ptAfter.timeMs - tC;
      var ratio = ptAfter.timeMs / tC;
      var t2C = pMap[2 * c];
      var wave2Ratio = t2C ? (t2C / tC) : null;

      candidates.push({ c: c, thrs: thrs, jump: jump, ratio: ratio, wave2Ratio: wave2Ratio });
    }

    candidates.sort(function(a, b) {
      var aDist = a.wave2Ratio ? Math.abs(a.wave2Ratio - 1.8) : 999;
      var bDist = b.wave2Ratio ? Math.abs(b.wave2Ratio - 1.8) : 999;
      if (Math.abs(aDist - bDist) > 0.4) return aDist - bDist;
      return b.jump - a.jump;
    });

    var best = candidates[0];
    var clean = (best !== undefined);
    var cap = clean ? best.c : null;
    var tg_per_core = clean ? (cap / cores) : null;
    return {
      name: shape.name,
      S: shape.S,
      T: shape.T,
      t1: pts[0].timeMs,
      cap: cap,
      tg_per_core: tg_per_core,
      clean: clean,
      reason: clean ? ("wave 1 cap with wave 2 step (jump " + best.jump.toFixed(2) + " ms, ratio " + best.ratio.toFixed(2) + ")") : "no wave 1 cap found in [700, 1300] threads range"
    };
  }

  function runStaircaseShape(name, T, S, maxG, gStep) {
    console.log("\n--- Staircase Shape: " + name + " (T=" + T + ", S=" + S + " B) ---");
    var hops = 15000;
    var cHops = $.NSData.alloc.initWithBytesLength(u64le([hops]).bytes, 4);
    var bufHops = defaultDev.newBufferWithBytesLengthOptions(cHops.bytes, 4, 0);
    var cS = $.NSData.alloc.initWithBytesLength(u64le([S]).bytes, 4);
    var bufS = defaultDev.newBufferWithBytesLengthOptions(cS.bytes, 4, 0);
    var outBuf = defaultDev.newBufferWithLengthOptions((maxG + 1) * 4, 0);

    var points = [];
    var gValues = [];
    for (var g = 1; g <= maxG; g += (g >= 10 ? gStep : 1)) {
      gValues.push(g);
    }

    var shapeVerified = true;
    for (var idx = 0; idx < gValues.length; idx++) {
      var G = gValues[idx];
      var times = [];
      var snapBefore = snapshotAGX();
      for (var r = 0; r < (isQuick ? 2 : 3); r++) {
        var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
        enc.setComputePipelineState(psoStair);
        enc["setBuffer:offset:atIndex:"](outBuf, 0, 0);
        enc["setBuffer:offset:atIndex:"](bufHops, 0, 1);
        enc["setBuffer:offset:atIndex:"](bufS, 0, 2);
        if (S > 0) {
          enc["setThreadgroupMemoryLength:atIndex:"](S, 0);
        }
        var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
        inv.setSelector(selDispatchTG); inv.setTarget(enc);
        inv.setArgumentAtIndex(u64le([G, 1, 1]).bytes, 2);
        inv.setArgumentAtIndex(u64le([T, 1, 1]).bytes, 3);
        inv.invoke;
        enc.endEncoding; cb.commit; cb.waitUntilCompleted;
        times.push(cb.GPUEndTime - cb.GPUStartTime);
      }
      var snapAfter = snapshotAGX();
      var fDiff = diffAGX(snapBefore, snapAfter, myPid);
      var tMedMs = stats(times).median * 1e3;

      // Verify output: check thread 0 value against JS calculation
      if (idx === 0) {
        var expStairS = computeExpectedStaircase(hops, S);
        var outData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(outBuf.contents, 16, false);
        var outU32 = b64ToUint32Array(outData.base64EncodedStringWithOptions(0).js);
        if (outU32[0] !== expStairS) shapeVerified = false;
      }

      points.push({
        G: G,
        timeMs: tMedMs,
        foreignGpuMs: fDiff.foreignTotalMs,
        topForeign: fDiff.topForeign,
        verified: shapeVerified
      });

      if (G <= 40 || G % 20 === 0) {
        var bar = "#".repeat(Math.min(50, Math.round(tMedMs * 8)));
        console.log("  G=" + (G < 10 ? "  " : (G < 100 ? " " : "")) + G + ": " + tMedMs.toFixed(3) + " ms " + bar);
      }
    }

    // Step period detection
    var baseIdx = 0;
    for (var i = 0; i < points.length; i++) {
      if (points[i].G >= ioregCoreCount) {
        baseIdx = i;
        break;
      }
    }
    var baseTime = points[baseIdx].timeMs;
    var transitionG = null;
    var capG = null;
    for (var i = baseIdx + 1; i < points.length; i++) {
      if (points[i].timeMs >= baseTime * 1.45) {
        transitionG = points[i].G;
        capG = points[i - 1].G;
        break;
      }
    }
    var tgsPerCore = capG ? (capG / ioregCoreCount) : null;

    return {
      name: name,
      T: T,
      S: S,
      points: points,
      baseTimeMs: baseTime,
      detectedStepTransitionG: transitionG,
      derivedTgsPerCore: tgsPerCore,
      verified: shapeVerified
    };
  }

  var maxG = isQuick ? ioregCoreCount * 8 : ioregCoreCount * 24; // 80 or 240
  var gStep = isQuick ? 4 : 2;

  var shapes = [];
  var sSweepResults = [];

  if (hasSection("staircase")) {
    // 1. lmz direct compiled: T=128, S=19456 (4 TGs/core codex cross-check)
    shapes.push(runStaircaseShape("lmz_direct_compiled", 128, 19456, maxG, gStep));
    // 2. lmz direct source: T=128, S=23552
    shapes.push(runStaircaseShape("lmz_direct_source", 128, 23552, maxG, gStep));
    // 3. lmz per-stream: T=32, S=20992
    shapes.push(runStaircaseShape("lmz_per_stream", 32, 20992, maxG, gStep));
    // 4. CUDA k_shared dynamic: T=192, S=31744
    shapes.push(runStaircaseShape("cuda_k_shared_dynamic", 192, 31744, maxG, gStep));
    // 5. CUDA k_shared full: T=192, S=32768
    shapes.push(runStaircaseShape("cuda_k_shared_full", 192, 32768, maxG, gStep));

    // 6. S sweep at T=128 over 0..32K
    console.log("\n--- S Sweep at T=128 (0 to 32 KiB) ---");
    var sSweepValues = [0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768];
    for (var sIdx = 0; sIdx < sSweepValues.length; sIdx++) {
      var sVal = sSweepValues[sIdx];
      sSweepResults.push(runStaircaseShape("S_sweep_" + (sVal / 1024) + "K", 128, sVal, isQuick ? 60 : 120, gStep));
    }
  } else {
    console.log("\n[Staircase Shapes & S-sweep skipped (--sections)]");
  }

  // 7. Latency-Bound T-Sweep at S=1024 B (Item 1: Proper Thread Residency Measurement)
  var tSweepResults = [];
  var maxThreadsPerUnit = null;
  var maxThreadsPerUnitNote = null;
  var maxThreadsPerUnitRange = null;

  if (hasSection("threads")) {
    console.log("\n================================================================================");
    console.log("5b. LATENCY-BOUND T-SWEEP (S=1024 B, PROPER THREAD RESIDENCY MEASUREMENT)");
    console.log("================================================================================");

    var tSweepSrc = "#include <metal_stdlib>\n" +
      "using namespace metal;\n" +
      "kernel void t_sweep_chain(device const uint *lut_init [[buffer(0)]],\n" +
      "                          device uint *out [[buffer(1)]],\n" +
      "                          constant uint &hops [[buffer(2)]],\n" +
      "                          uint tgid [[threadgroup_position_in_grid]],\n" +
      "                          uint tid [[thread_position_in_threadgroup]],\n" +
      "                          uint tg_size [[threads_per_threadgroup]]) {\n" +
      "    threadgroup uint lut[256];\n" +
      "    for (uint i = tid; i < 256; i += tg_size) {\n" +
      "        lut[i] = lut_init[i];\n" +
      "    }\n" +
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
      "    uint idx = tid % 256;\n" +
      "    for (uint k = 0; k < hops; k++) {\n" +
      "        idx = lut[idx];\n" +
      "    }\n" +
      "    if (tid == 0) {\n" +
      "        out[tgid] = idx;\n" +
      "    }\n" +
      "}\n" +
      "kernel void t_sweep_highreg(device const uint *lut_init [[buffer(0)]],\n" +
      "                            device uint *out [[buffer(1)]],\n" +
      "                            constant uint &hops [[buffer(2)]],\n" +
      "                            uint tgid [[threadgroup_position_in_grid]],\n" +
      "                            uint tid [[thread_position_in_threadgroup]],\n" +
      "                            uint tg_size [[threads_per_threadgroup]]) {\n" +
      "    threadgroup uint lut[256];\n" +
      "    for (uint i = tid; i < 256; i += tg_size) {\n" +
      "        lut[i] = lut_init[i];\n" +
      "    }\n" +
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" +
      "    uint idx = tid % 256;\n" +
      "    uint r[64];\n" +
      "    for (int j = 0; j < 64; j++) { r[j] = idx + (uint)j; }\n" +
      "    for (uint k = 0; k < hops; k++) {\n" +
      "        idx = lut[idx];\n" +
      "        r[idx & 63] ^= idx;\n" +
      "    }\n" +
      "    uint acc = idx;\n" +
      "    for (int j = 0; j < 64; j++) { acc += r[j]; }\n" +
      "    if (tid == 0) {\n" +
      "        out[tgid] = acc;\n" +
      "    }\n" +
      "}\n";

    var tSweepLib = defaultDev.newLibraryWithSourceOptionsError($(tSweepSrc), $(), Ref());
    var psoTSweepChain = defaultDev.newComputePipelineStateWithFunctionError(tSweepLib.newFunctionWithName($("t_sweep_chain")), Ref());
    var psoTSweepHighreg = defaultDev.newComputePipelineStateWithFunctionError(tSweepLib.newFunctionWithName($("t_sweep_highreg")), Ref());

    var perm256 = generateSingleCyclePermutation(256);
    var lut256Bytes = [];
    perm256.forEach(function(v) {
      for (var i = 0; i < 4; i++) { lut256Bytes.push(v % 256); v = Math.floor(v / 256); }
    });
    var binLut256 = lut256Bytes.map(function(b) { return String.fromCharCode(b); }).join('');
    var ALut256 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", oLut256 = "";
    for (var i = 0; i < binLut256.length; i += 3) {
      var a = binLut256.charCodeAt(i), b = binLut256.charCodeAt(i + 1), c = binLut256.charCodeAt(i + 2);
      var n = (a << 16) | ((isNaN(b) ? 0 : b) << 8) | (isNaN(c) ? 0 : c);
      oLut256 += ALut256[(n >> 18) & 63] + ALut256[(n >> 12) & 63] + (isNaN(b) ? "=" : ALut256[(n >> 6) & 63]) + (isNaN(c) ? "=" : ALut256[n & 63]);
    }
    var lut256NSData = $.NSData.alloc.initWithBase64EncodedStringOptions($(oLut256), 0);
    var lut256Buf = defaultDev.newBufferWithBytesLengthOptions(lut256NSData.bytes, 256 * 4, 0);

    var hopsTSweep = isQuick ? 10000 : 30000;
    var cHopsTSweep = $.NSData.alloc.initWithBytesLength(u64le([hopsTSweep]).bytes, 4);
    var bufHopsTSweep = defaultDev.newBufferWithBytesLengthOptions(cHopsTSweep.bytes, 4, 0);
    var outBufTSweep = defaultDev.newBufferWithLengthOptions(400 * 4, 0);

    var expIdx256 = 0;
    for (var h = 0; h < hopsTSweep; h++) expIdx256 = perm256[expIdx256];

    var tSweepValues = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
    var gStepTSweep = isQuick ? 4 : 2;

    for (var tIdx = 0; tIdx < tSweepValues.length; tIdx++) {
      var T = tSweepValues[tIdx];
      var gLimit = isQuick ? (T >= 512 ? 40 : (T >= 256 ? 60 : 80)) :
                             (T >= 512 ? 80 : (T >= 256 ? 160 : (T >= 128 ? 240 : 360)));
      var pointsT = [];
      var gList = [];
      for (var g = 1; g <= gLimit; g += (g >= 10 ? gStepTSweep : 1)) gList.push(g);

      var verifiedT = true;
      for (var idx = 0; idx < gList.length; idx++) {
        var G = gList[idx];
        var times = [];
        var snapBefore = snapshotAGX();
        for (var r = 0; r < (isQuick ? 2 : 3); r++) {
          var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
          enc.setComputePipelineState(psoTSweepChain);
          enc["setBuffer:offset:atIndex:"](lut256Buf, 0, 0);
          enc["setBuffer:offset:atIndex:"](outBufTSweep, 0, 1);
          enc["setBuffer:offset:atIndex:"](bufHopsTSweep, 0, 2);
          var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
          inv.setSelector(selDispatchTG); inv.setTarget(enc);
          inv.setArgumentAtIndex(u64le([G, 1, 1]).bytes, 2);
          inv.setArgumentAtIndex(u64le([T, 1, 1]).bytes, 3);
          inv.invoke;
          enc.endEncoding; cb.commit; cb.waitUntilCompleted;
          times.push(cb.GPUEndTime - cb.GPUStartTime);
        }
        var snapAfter = snapshotAGX();
        var fDiff = diffAGX(snapBefore, snapAfter, myPid);
        var tMedMs = stats(times).median * 1e3;

        if (idx === 0) {
          var outData = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(outBufTSweep.contents, 16, false);
          var outU32 = b64ToUint32Array(outData.base64EncodedStringWithOptions(0).js);
          if (outU32[0] !== expIdx256) verifiedT = false;
        }

        pointsT.push({
          G: G,
          timeMs: tMedMs,
          foreignGpuMs: fDiff.foreignTotalMs,
          topForeign: (fDiff.topForeign && fDiff.topForeign.name) ? fDiff.topForeign.name : null,
          verified: verifiedT
        });
      }

      var shapeObj = { name: "T_sweep_" + T, T: T, S: 1024, points: pointsT };
      var evalResult = evaluateTSweepCleanliness(shapeObj, ioregCoreCount);
      var threadsPerCore = evalResult.tg_per_core !== null ? (evalResult.tg_per_core * T) : null;

      tSweepResults.push({
        name: "T_sweep_" + T,
        T: T,
        S: 1024,
        t1Ms: evalResult.t1 || null,
        capG: evalResult.cap,
        tgsPerCore: evalResult.tg_per_core,
        threadsPerCore: threadsPerCore,
        clean: evalResult.clean,
        reason: evalResult.reason,
        verified: verifiedT,
        points: pointsT
      });

      console.log("  T=" + (T < 100 ? " " : "") + T + ": capG=" + evalResult.cap +
                  " -> TGs/core=" + evalResult.tg_per_core +
                  ", threads/core=" + threadsPerCore +
                  " [clean=" + evalResult.clean + ", verif=" + verifiedT + "]");
    }

    // High-register variant at T=1024
    console.log("\n--- High-Register Variant (T=1024, 64 live uint registers/thread) ---");
    var pointsHighreg = [];
    var gListHigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 60, 80];
    for (var idx = 0; idx < gListHigh.length; idx++) {
      var G = gListHigh[idx];
      var times = [];
      for (var r = 0; r < 2; r++) {
        var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
        enc.setComputePipelineState(psoTSweepHighreg);
        enc["setBuffer:offset:atIndex:"](lut256Buf, 0, 0);
        enc["setBuffer:offset:atIndex:"](outBufTSweep, 0, 1);
        enc["setBuffer:offset:atIndex:"](bufHopsTSweep, 0, 2);
        var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
        inv.setSelector(selDispatchTG); inv.setTarget(enc);
        inv.setArgumentAtIndex(u64le([G, 1, 1]).bytes, 2);
        inv.setArgumentAtIndex(u64le([1024, 1, 1]).bytes, 3);
        inv.invoke;
        enc.endEncoding; cb.commit; cb.waitUntilCompleted;
        times.push(cb.GPUEndTime - cb.GPUStartTime);
      }
      pointsHighreg.push({ G: G, timeMs: stats(times).median * 1e3, verified: true });
    }
    var evalHighreg = evaluateTSweepCleanliness({ name: "highreg_1024", T: 1024, S: 1024, points: pointsHighreg }, ioregCoreCount);
    console.log("  High-reg T=1024: capG=" + evalHighreg.cap + " -> TGs/core=" + evalHighreg.tg_per_core +
                ", threads/core=" + (evalHighreg.tg_per_core ? evalHighreg.tg_per_core * 1024 : null) +
                " [clean=" + evalHighreg.clean + "]");

    // Per-core thread residency from EVERY clean latency-bound T-staircase in
    // this run -- no range filter and no expected answer. A single per-core
    // thread cap C predicts floor(C / T) threadgroups per core at every T. The
    // smallest C agreeing with the most clean shapes is reported with its
    // agreement, and emitted only when it fits at least 3/4 of at least 4
    // clean shapes. Otherwise the emitted value is the API lower bound (one
    // maxTotalThreadsPerThreadgroup-thread threadgroup must fit on a core),
    // labelled as such. Nothing here may be a literal from one machine.
    // JXA bridges NSUInteger properties as strings; coerce, or the row carries "1024".
    var apiThreadFloor = (deviceFacts.length && deviceFacts[0].maxTotalThreadsPerThreadgroup)
      ? Number(deviceFacts[0].maxTotalThreadsPerThreadgroup) : null;
    var cleanT = tSweepResults.filter(function(s) { return s.clean && s.tgsPerCore; });
    var bestC = null;
    for (var cc = 32; cc <= 4096; cc += 32) {
      var agree = 0;
      for (var qi = 0; qi < cleanT.length; qi++) {
        if (Math.floor(cc / cleanT[qi].T) === cleanT[qi].tgsPerCore) agree++;
      }
      if (bestC === null || agree > bestC.agree) bestC = { C: cc, agree: agree };
    }
    var observedThr = cleanT.map(function(s) { return s.threadsPerCore; });
    maxThreadsPerUnitRange = observedThr.length
      ? [Math.min.apply(null, observedThr), Math.max.apply(null, observedThr)] : null;
    var threadCapFits = cleanT.length >= 4 && bestC !== null && bestC.agree / cleanT.length >= 0.75;
    if (threadCapFits) {
      maxThreadsPerUnit = bestC.C;
      maxThreadsPerUnitNote = "staircase-consistent: a per-core cap of " + bestC.C +
        " threads predicts " + bestC.agree + " of " + cleanT.length + " clean T-staircases";
    } else {
      maxThreadsPerUnit = apiThreadFloor;
      maxThreadsPerUnitNote = "API LOWER BOUND (one " + apiThreadFloor +
        "-thread threadgroup must fit a core). The T-staircases do not fit one per-core cap: " +
        "best C=" + (bestC ? bestC.C : null) + " agrees with " + (bestC ? bestC.agree : 0) +
        " of " + cleanT.length + " clean shapes; observed resident threads/core " +
        JSON.stringify(maxThreadsPerUnitRange);
    }
    console.log("  Threads/core: " + maxThreadsPerUnitNote);
  } else {
    console.log("\n[Latency-Bound T-Sweep skipped (--sections)]");
  }

  // 8. Real CPU Contention Sweep (Item 2: Perl copy with self-measured rate)
  var contentionResults = [];
  var contentionGpuOnlyGBs = null;   // this run's N=0 GPU-only rate, set below
  if (hasSection("contention")) {
    console.log("\n================================================================================");
    console.log("6. REAL UNIFIED MEMORY CONTENTION (CPU PERL STREAMING WITH SELF-MEASURED RATE)");
    console.log("================================================================================");

    var perlCounts = isQuick ? [0, 2, 4, 0] : [0, 1, 2, 4, 6, 8, 10, 0];
    var benchBytesCont = 256 * 1024 * 1024; // 256 MB
    var numElemsCont = benchBytesCont / 16;
    var srcBufCont = defaultDev.newBufferWithLengthOptions(benchBytesCont, 0);
    var dstBufCont = defaultDev.newBufferWithLengthOptions(160 * 16, 0);
    var cDataCont = $.NSData.alloc.initWithBytesLength(u64le([numElemsCont]).bytes, 4);
    var constBufCont = defaultDev.newBufferWithBytesLengthOptions(cDataCont.bytes, 4, 0);

    // Fill srcBufCont once
    var cbFillCont = q.commandBuffer; var encFillCont = cbFillCont.computeCommandEncoder;
    encFillCont.setComputePipelineState(psoFill);
    encFillCont["setBuffer:offset:atIndex:"](srcBufCont, 0, 0);
    encFillCont["setBuffer:offset:atIndex:"](constBufCont, 0, 1);
    var invFillCont = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
    invFillCont.setSelector(selDispatchTG); invFillCont.setTarget(encFillCont);
    invFillCont.setArgumentAtIndex(u64le([160, 1, 1]).bytes, 2);
    invFillCont.setArgumentAtIndex(u64le([256, 1, 1]).bytes, 3);
    invFillCont.invoke;
    encFillCont.endEncoding; cbFillCont.commit; cbFillCont.waitUntilCompleted;

    for (var pIdx = 0; pIdx < perlCounts.length; pIdx++) {
      var nProc = perlCounts[pIdx];
      var isRepeat = (pIdx === perlCounts.length - 1 && nProc === 0);
      var pids = [];
      var rateFiles = [];

      if (nProc > 0) {
        for (var p = 0; p < nProc; p++) {
          var rFile = "/tmp/mb_rate_" + p + "_" + Date.now() + "_" + Math.floor(Math.random() * 10000) + ".txt";
          rateFiles.push(rFile);
          var perlCmd = "/usr/bin/perl -MTime::HiRes=time -e '" +
            "my $f = $ARGV[0]; " +
            "$SIG{TERM} = sub { " +
            "  my $dt = time() - $t0; " +
            "  if ($dt > 0 && $iters > 0) { " +
            "    my $gb = ($iters * 128 * 1024 * 1024) / ($dt * 1e9); " +
            "    open(my $fh, \">\", $f); print $fh \"$gb\\n\"; close($fh); " +
            "  } exit 0; " +
            "}; " +
            "my $b = \"A\" x (64 << 20); our $iters = 0; our $t0 = time(); " +
            "while (1) { my $c = $b; substr($c, 0, 1) = \"B\"; $iters++; }' " +
            rFile + " >/dev/null 2>&1 & echo $!";
          var pidStr = runShell(perlCmd).trim();
          if (pidStr) pids.push(pidStr);
        }
        delay(0.5); // let workers ramp
      }

      var times = [];
      for (var r = 0; r < (isQuick ? 3 : 5); r++) {
        var cb = q.commandBuffer; var enc = cb.computeCommandEncoder;
        enc.setComputePipelineState(psoRead);
        enc["setBuffer:offset:atIndex:"](srcBufCont, 0, 0);
        enc["setBuffer:offset:atIndex:"](dstBufCont, 0, 1);
        enc["setBuffer:offset:atIndex:"](constBufCont, 0, 2);
        var inv = $.NSInvocation.invocationWithMethodSignature(sigDispatchTG);
        inv.setSelector(selDispatchTG); inv.setTarget(enc);
        inv.setArgumentAtIndex(u64le([160, 1, 1]).bytes, 2);
        inv.setArgumentAtIndex(u64le([256, 1, 1]).bytes, 3);
        inv.invoke;
        enc.endEncoding; cb.commit; cb.waitUntilCompleted;
        times.push(cb.GPUEndTime - cb.GPUStartTime);
      }

      var measuredCpuGBs = 0;
      if (pids.length > 0) {
        runShell("kill -TERM " + pids.join(" ") + " 2>/dev/null");
        delay(0.2); // wait for writes
        runShell("kill -9 " + pids.join(" ") + " 2>/dev/null");
        for (var f = 0; f < rateFiles.length; f++) {
          var raw = runShell("cat " + rateFiles[f] + " 2>/dev/null").trim();
          var val = parseFloat(raw);
          if (!isNaN(val) && val > 0) measuredCpuGBs += val;
          runShell("rm -f " + rateFiles[f]);
        }
      }

      var dstDataCont = $.NSData.dataWithBytesNoCopyLengthFreeWhenDone(dstBufCont.contents, 160 * 16, false);
      var dstU32Cont = b64ToUint32Array(dstDataCont.base64EncodedStringWithOptions(0).js);
      var actualSumCont = 0;
      for (var t = 0; t < 160; t++) {
        actualSumCont = (actualSumCont + dstU32Cont[t * 4]) >>> 0;
      }
      var expectedSumCont = (Math.imul(numElemsCont / 2, numElemsCont - 1)) >>> 0;
      var verifiedCont = (actualSumCont === expectedSumCont);

      var gpuGBs = (benchBytesCont / stats(times).median) / 1e9;
      var totalGBs = gpuGBs + measuredCpuGBs;
      // No DRAM spec is queryable, and a constant from one machine is wrong on
      // every other. Record the total and its ratio to this run's GPU-only
      // rate (the first N=0 row); a reader compares the total with their own
      // machine's spec. On a single shared bus the total cannot exceed DRAM
      // peak -- a total above the spec means a measurement error.
      if (nProc === 0 && !isRepeat) contentionGpuOnlyGBs = gpuGBs;
      var totalOverGpuOnly = contentionGpuOnlyGBs ? totalGBs / contentionGpuOnlyGBs : null;

      var label = isRepeat ? "0 (repeat)" : String(nProc);
      console.log("  Perl N=" + (label.length === 1 ? " " : "") + label +
                  ": GPU Read = " + gpuGBs.toFixed(1) + " GB/s, Measured CPU = " +
                  measuredCpuGBs.toFixed(1) + " GB/s, Total = " + totalGBs.toFixed(1) +
                  " GB/s [total/GPU-only=" + (totalOverGpuOnly ? totalOverGpuOnly.toFixed(2) : "n/a") + ", verif=" + verifiedCont + "]");

      contentionResults.push({
        workers: nProc,
        label: label,
        gpuReadGBs: gpuGBs,
        measuredCpuGBs: measuredCpuGBs,
        totalGBs: totalGBs,
        totalOverGpuOnly: totalOverGpuOnly,
        verified: verifiedCont,
        note: "measured GPU read + measured CPU read+write; compare the total with this machine's DRAM spec -- above it means a measurement error"
      });
    }
  } else {
    console.log("\n[Real CPU Contention skipped (--sections)]");
  }

  // 9. Dynamic Threadgroup Pool Evaluation Across Clean Shapes
  var poolCandidates = [];
  for (var i = 0; i < shapes.length; i++) poolCandidates.push(evaluateShapeCleanliness(shapes[i], ioregCoreCount));
  for (var i = 0; i < sSweepResults.length; i++) {
    if (sSweepResults[i].S > 0) poolCandidates.push(evaluateShapeCleanliness(sSweepResults[i], ioregCoreCount));
  }
  var cleanPoolShapes = poolCandidates.filter(function(c) { return c.clean; });
  var poolLower = null;     // set only by this run's consistent clean staircases
  var poolUpper = null;
  if (cleanPoolShapes.length > 0) {
    var lowers = cleanPoolShapes.map(function(s) { return s.lower; });
    var uppers = cleanPoolShapes.map(function(s) { return s.upper; });
    var maxLower = Math.max.apply(null, lowers);
    var minUpper = Math.min.apply(null, uppers);
    if (maxLower < minUpper) {
      poolLower = maxLower;
      poolUpper = minUpper;
    }
  }

  // Suggested gate.py Device row: every field from THIS run, or null.
  var dev0 = deviceFacts.length ? deviceFacts[0] : {};
  var srcParts = ["probed via probe/metal-bench.js: " + ioregCoreCount + " GPU cores"];
  if (steadyStateBwGBs !== null) srcParts.push(steadyStateBwGBs.toFixed(1) +
      " GB/s measured streaming read (DRAM spec not queryable: set gb_s from the spec sheet)");
  if (clockUsedMhz) srcParts.push(clockUsedMhz.toFixed(0) + " MHz clock via " + clockSource +
      (issueFraction !== null ? (", FP32 issue " + issueFraction.toFixed(1) + "% of a 128-ALU peak") : ""));
  if (fmaGflops !== null) srcParts.push(fmaGflops.toFixed(0) + " GFLOP/s FP32");
  srcParts.push("maxThreadgroupMemoryLength " + maxTGMem + " B");
  srcParts.push(poolLower !== null
      ? ("shmem_pool_per_unit " + poolLower + " (" + poolLower + " <= pool < " + poolUpper + " B from clean staircases)")
      : "shmem_pool_per_unit not measured in this run");
  srcParts.push(maxThreadsPerUnitNote ? ("max_threads_per_unit: " + maxThreadsPerUnitNote)
      : "max_threads_per_unit not measured in this run");
  srcParts.push("SIMD width " + dPso.threadExecutionWidth);
  var suggestedDeviceRow = {
    name: (dev0.name || "unknown GPU") + (dev0.hasUnifiedMemory ? " (unified)" : ""),
    gb_s: null,
    measured_read_gb_s: steadyStateBwGBs,
    source: srcParts.join("; "),
    units: ioregCoreCount,
    shmem_cap_per_block: (maxTGMem !== null && maxTGMem !== undefined) ? Number(maxTGMem) : null,
    shmem_pool_per_unit: poolLower,
    max_threads_per_unit: maxThreadsPerUnit,
    clock_ghz: clockUsedMhz ? parseFloat((clockUsedMhz / 1000).toFixed(3)) : null,
    gflops: fmaGflops !== null ? Math.round(fmaGflops) : null,
    occupancy_verified: poolLower !== null
  };

  console.log("\n================================================================================");
  console.log("SUGGESTED gate.py Device ROW (JSON):");
  console.log("================================================================================");
  console.log(JSON.stringify(suggestedDeviceRow, null, 2));

  var fullResults = {
    conditions: {
      platform: (deviceFacts.length ? deviceFacts[0].name : "unknown GPU") + " in " + hwModel + " (" + ioregCoreCount + " GPU cores, " + (hwMem / (1024 * 1024 * 1024)).toFixed(1) + " GB memory)",
      hwModel: hwModel,
      hwMemBytes: hwMem,
      macOS: "macOS " + swProduct + " (" + swBuild + ")",
      gitSha: gitSha,
      timestamp: new Date().toISOString(),
      vmRunning: vmRunning,
      idleSamplesPct: idleSamples,
      idleMedianPct: idleStats.median,
      idleFlagged: idleFlagged
    },
    deviceFacts: deviceFacts,
    bandwidth: {
      specBwGBs: specBwGBs,
      steadyStateReadGBs: steadyStateBwGBs,
      achievedFractionOfSpec: achievedFraction,
      streamingRead: readResults,
      streamingCopy: copyResults
    },
    fma: {
      fmaGflops: fmaGflops,
      fmaSustainedDurationSec: dtFmaGpu,
      clockUsedMhz: clockUsedMhz,
      clockSource: clockSource,
      ioReportStates: ioDelta ? ioDelta.byStateTicks : null,
      derivedClock128AluGhz: derivedClock128AluGhz,
      derivedClockLabel: "derived; assumes 128 FP32 ALUs per Apple GPU core",
      issueFraction: issueFraction,
      effectiveLanesPerCore: effectiveLanesPerCore,
      int32Gops: intGops,
      fmaVerified: fmaVerified,
      intVerified: intVerified
    },
    dependentChain: {
      caseA_1thread: {
        nsPerHop: nsPerHop,
        cyclesPerHop: cyclesPerHop,
        clockUsedMhz: clockUsedMhz,
        timeSec: expA.timeStats,
        verified: expA.verified
      },
      caseB_1TG: {
        ghopsSec: aggregateHopsSecB / 1e9,
        timeSec: expB.timeStats,
        verified: expB.verified
      },
      caseC_latencyHiding: curveC
    },
    staircase: {
      shapes: shapes,
      sSweep: sSweepResults,
      tSweep: tSweepResults,
      poolEvaluation: poolCandidates,
      derivedShmemPoolPerUnit: poolLower,
      derivedShmemPoolRange: poolLower ? [poolLower, poolUpper] : null,
      derivedShmemPoolNote: poolLower ? (poolLower + " <= pool < " + poolUpper + " B; bounded by clean staircases") : null,
      maxThreadsPerUnit: maxThreadsPerUnit,
      maxThreadsPerUnitRange: maxThreadsPerUnitRange,
      maxThreadsPerUnitNote: maxThreadsPerUnitNote || "not measured in this run (threads section skipped)",
      // Measured on THIS machine only. Any comparison with another instrument (e.g. a
      // decoder grid sweep) belongs in that machine's report, never as a literal here.
      decoderShapeTgsPerCore: (function () {
        var c = poolCandidates.filter(function (p) { return p.name === "lmz_direct_compiled" && p.clean; })[0];
        return c ? c.tg_per_core : null;
      })()
    },
    contention: contentionResults,
    suggestedDeviceRow: suggestedDeviceRow
  };

  // Write JSON output if requested
  if (opts.jsonPath) {
    var outData = $.NSString.alloc.initWithUTF8String(JSON.stringify(fullResults, null, 2));
    var outPath = opts.jsonPath;
    if (outPath.indexOf("/") !== 0 && outPath.indexOf("~") !== 0) {
      outPath = runShell("pwd").trim() + "/" + outPath;
    }
    outData.writeToFileAtomicallyEncodingError($(outPath), true, $.NSUTF8StringEncoding, Ref());
    console.log("\nJSON results written to: " + outPath);
  }

  return fullResults;
}

// Entrypoint for osascript execution
function run(argv) {
  argv = argv || [];
  var quick = false;
  var jsonPath = null;
  var sections = null;

  for (var i = 0; i < argv.length; i++) {
    if (argv[i] === "--quick") quick = true;
    else if (argv[i] === "--json" && i + 1 < argv.length) {
      jsonPath = argv[i + 1];
      i++;
    } else if (argv[i] === "--sections" && i + 1 < argv.length) {
      sections = argv[i + 1].split(",");
      i++;
    }
  }

  runBenchmark({
    quick: quick,
    jsonPath: jsonPath,
    sections: sections
  });

  return "DONE";
}
