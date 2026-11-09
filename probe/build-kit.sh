#!/bin/sh
# Assemble the 2-CU kit: everything someone needs to run the pool measurement on a
# machine this repository cannot reach. Reproducible: the kit is built from tracked
# files, never hand-edited.
#
#   probe/build-kit.sh [output-dir]        (default: results/)
#
# Contents:  README.txt (probe/kit-2cu-README.txt), igpu-pool.ps1, igpu-pool-analyze.py,
# igpu-bench.ps1 (bandwidth + FMA), and -- if the lmz Vulkan kit has been built --
# vulkan/ (standalone Python + precompiled SPIR-V + a prepared workload).
# Ends with SHA256SUMS.txt so the recipient can check the copy.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
out=${1:-$repo/results}
stamp=$(date +%Y%m%d)
name=2cu-kit-$stamp
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
dir=$tmp/$name
mkdir -p "$dir"

cp "$here/kit-2cu-README.txt"    "$dir/README.txt"
cp "$here/igpu-pool.ps1"         "$dir/"
cp "$here/igpu-pool-analyze.py"  "$dir/"
cp "$here/igpu-bench.ps1"        "$dir/"

# The Vulkan half is optional and lives in the lmz repository (a sibling checkout).
vk=${LMZ_VULKAN_KIT:-$repo/../lmz/scratchpad/gpu/vulkan/kit}
if [ -d "$vk" ] && [ -f "$vk/vulkan_sweep.py" ]; then
    mkdir -p "$dir/vulkan"
    # An explicit list: never a glob, so a stray result file or cache directory cannot ship.
    for f in README.txt vulkan_sweep.py vulkan.py lmz_rans.spv lmz_rans_stride.spv kit-workload.bin; do
        [ -f "$vk/$f" ] || { echo "Vulkan kit is missing $f (run scratchpad/gpu/vulkan/package_kit.py)" >&2; exit 1; }
        cp "$vk/$f" "$dir/vulkan/$f"
    done
    echo "included Vulkan kit from $vk ($(du -sh "$dir/vulkan" | cut -f1))"
else
    echo "no Vulkan kit found at $vk: building the D3D11 half only" >&2
    # Do not leave instructions that point at a folder that is not there.
    sed -i.bak -e '/^PART 2 -- THE VULKAN/,/^SEND BACK/{/^SEND BACK/!d;}' \
               -e '/^  Part 2 (Vulkan decoder)/,/^  Analysis /{/^  Analysis /!d;}' \
               -e 's/ and vulkan result \*\.json//' "$dir/README.txt" && rm -f "$dir/README.txt.bak"
fi

( cd "$dir" && find . -type f ! -name SHA256SUMS.txt | sort | xargs shasum -a 256 > SHA256SUMS.txt )
mkdir -p "$out"
( cd "$tmp" && zip -q -r "$out/$name.zip" "$name" )
echo "wrote $out/$name.zip"
( cd "$tmp" && unzip -l "$out/$name.zip" | tail -n +4 | sed '$d' )
