lmsluice 2-CU KIT -- run this on the machine with the small AMD iGPU (and its big NVIDIA card)
=========================================================================================

WHAT THIS IS FOR
  One question decides whether lmz's decoder can beat the CPU on the smallest AMD
  integrated GPU: how many "blocks" (threadgroups that each ask for ~31 KiB of
  shared memory) fit on one compute unit at once.  The answer is 1 or 3, and the
  answer to "does the GPU decode beat the CPU?" flips with it.  No driver reports
  it, so this kit MEASURES it.

  Nothing here installs anything.  It reads your GPU; it does not change settings.

WHAT YOU NEED
  Part 1 (D3D11 staircase)  : Windows PowerShell 5.1 (built into Windows 10/11).
  Part 2 (Vulkan decoder)   : Python 3.9+ and the GPU driver's Vulkan loader
                              (vulkan-1.dll ships with AMD/NVIDIA/Intel drivers).
  Analysis                  : Python 3 (any machine; you can do this part later).

BEFORE YOU START
  * Plug in the power; set Windows power mode to "Best performance".
  * Close browsers, games, video, anything using the GPU.  Do not use the machine
    while a measurement runs.
  * Windows will reset the GPU if one job takes > 2 seconds (TDR).  The scripts
    keep every job well under that; if the screen flickers once, just rerun.

PART 1 -- THE D3D11 STAIRCASE  (the important one)
  1. Open PowerShell in this folder and list your GPUs:

       powershell -ExecutionPolicy Bypass -File igpu-pool.ps1 -Adapter -1

     Note the index of the small AMD iGPU (call it A) and of the big NVIDIA card (N).

  2. CALIBRATE FIRST, on the big NVIDIA card (RTX 5080 has 84 SMs):

       powershell -ExecutionPolicy Bypass -File igpu-pool.ps1 -Adapter N -Units 84

     When it finishes, the row  lmz_cuda_shared_dynamic  should show a wave-1 cap
     near 252.  If it does not, STOP and send the file anyway: the method needs
     fixing before the small iGPU's answer means anything.

  3. Now the small iGPU.  -Units is its compute-unit count (lmz recorded 2 for this
     Radeon; use 2 unless you know otherwise):

       powershell -ExecutionPolicy Bypass -File igpu-pool.ps1 -Adapter A -Units 2

  4. REPEAT steps 2 and 3 a SECOND TIME, later (different time of day, ideally after
     a reboot).  Boundary cases look clean in one run and are not; two runs are how
     we tell.  This is not optional.

  Each run writes igpu-pool-<adapter>-<time>.json in this folder.  A run takes
  about a minute or two.  (-Quick does a fast four-shape version; use it only to
  check the script works, not for the answer.)

PART 2 -- THE VULKAN DECODER SWEEP  (if the "vulkan" folder is present)
  See vulkan\README.txt.  It runs lmz's own Vulkan decoder on a small prepared
  workload, checks the output byte-for-byte by SHA-256, and times it.  Same
  rule: do it twice.

SEND BACK
  Every igpu-pool-*.json and vulkan result *.json you produced, plus a note of:
    - which adapter index was which GPU, the Windows version, the driver version
    - anything odd (flicker, errors, a run that took very long)
  The JSON files contain adapter names, OS and driver versions, and timings.
  Nothing personal.

ANALYSIS  (can be done on any machine with Python 3)
       python igpu-pool-analyze.py runA1.json runA2.json
       python igpu-pool-analyze.py runN1.json runN2.json --expect lmz_cuda_shared_dynamic=252

  It states a pool-per-unit bound only from shapes that agree in every run, and
  says so plainly when they disagree.  A disagreement is a result too.

WHAT THIS DOES NOT MEASURE
  * Not decode speed: it measures residency.  Speed comes from Part 2 and from a
    real decode on this hardware, which nothing here is.
  * Not the CPU contending for the same memory bus.
  * The unit count is typed in, not detected.
  * The PowerShell/D3D11 script was written on a Mac and has NEVER run on Windows
    before this.  If it errors, send the error text; that is useful.
