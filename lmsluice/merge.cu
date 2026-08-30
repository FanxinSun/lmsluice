// Interleaving decoded byte planes back into elements, on the device.
//
// lmz splits a tensor by byte position before coding it: for a 4-byte dtype,
// plane p holds byte p of every element, and the four planes are coded
// independently because each has its own statistics. `lmz_gpu_decode_batch_dev`
// gives those planes back plane-major -- all of plane 0, then all of plane 1 --
// and the original bytes are the transpose of that.
//
// LOAN -- lmz's, held here. See the loan register in ../docs/boundary.md.
// Retired by: a device decode entry point that returns plaintext (invariant I2).
//
// This kernel is the transpose, and it exists reluctantly. It sits on lmz's
// side of the boundary this package otherwise respects: decoding is lmz's job
// and lmsluice's job is dispatch. It is here because lmz's shipped kernel stops
// one step short of the plaintext, and the alternatives are worse -- the CUDA
// driver's own strided copy does this at 1.87 GB/s against a decoder running at
// 111, which would make the device route slower than the host one it exists to
// beat. It should move into lmz when lmz grows a fused path, and nothing here
// depends on it staying.
//
// Compiled to PTX, not to a shared library, so it loads through the same driver
// API as everything else in `cuda.py` and needs no CUDA runtime at load time.
// The driver JITs PTX for whatever architecture is actually present, which also
// means one artifact works on every card rather than one per sm_.

extern "C" {

// The general case. One thread per element, gathering nplanes bytes.
// Reads are coalesced within each plane; the write is not, which is why the
// specialisations below exist for the widths that actually occur.
__global__ void lmz_merge_n(const unsigned char *__restrict__ planes,
                            unsigned char *__restrict__ out,
                            unsigned nplanes, unsigned long long nelem)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    for (unsigned p = 0; p < nplanes; ++p)
        out[i * nplanes + p] = planes[(unsigned long long)p * nelem + i];
}

// Two bytes an element: fp16, bf16.
__global__ void lmz_merge_2(const unsigned char *__restrict__ planes,
                            unsigned char *__restrict__ out,
                            unsigned long long nelem)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    uchar2 v;
    v.x = planes[i];
    v.y = planes[nelem + i];
    // One aligned 2-byte store instead of two scattered 1-byte ones: the
    // gather side is already coalesced, so the write is what decides the rate.
    reinterpret_cast<uchar2 *>(out)[i] = v;
}

// Four bytes an element: fp32, int32.
__global__ void lmz_merge_4(const unsigned char *__restrict__ planes,
                            unsigned char *__restrict__ out,
                            unsigned long long nelem)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    uchar4 v;
    v.x = planes[i];
    v.y = planes[nelem + i];
    v.z = planes[2ULL * nelem + i];
    v.w = planes[3ULL * nelem + i];
    reinterpret_cast<uchar4 *>(out)[i] = v;
}

// Eight bytes an element: fp64, int64. Assembled as two 32-bit halves so the
// store is a single 8-byte transaction.
__global__ void lmz_merge_8(const unsigned char *__restrict__ planes,
                            unsigned char *__restrict__ out,
                            unsigned long long nelem)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    unsigned lo = 0, hi = 0;
    for (unsigned p = 0; p < 4; ++p)
        lo |= ((unsigned)planes[(unsigned long long)p * nelem + i]) << (8 * p);
    for (unsigned p = 0; p < 4; ++p)
        hi |= ((unsigned)planes[(unsigned long long)(p + 4) * nelem + i]) << (8 * p);
    uint2 v;
    v.x = lo;
    v.y = hi;
    reinterpret_cast<uint2 *>(out)[i] = v;
}

// Copy a plane the coder stored raw rather than coding. Such a plane never
// reaches the rANS batch, so it arrives as bytes and still has to be woven in.
__global__ void lmz_scatter_1(const unsigned char *__restrict__ src,
                              unsigned char *__restrict__ out,
                              unsigned nplanes, unsigned plane,
                              unsigned long long nelem)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    out[i * nplanes + plane] = src[i];
}

}  // extern "C"

// One launch for every chunk in a batch, instead of one launch each.
//
// A page-mapped archive has tens of thousands of 64 KiB blocks, and dispatching
// a kernel per block put the device route at 0.60 GB/s against 1.87 for the
// same data in 8 MiB blocks -- slower with more parallelism available, which is
// the signature of launch overhead rather than of the GPU. Chunks in a batch
// already share `nelem` and `nplanes` by construction, so the whole batch is
// one index space: job = t / nelem, element = t % nelem.
extern "C" __global__ void lmz_merge_batch(
    const unsigned char *__restrict__ planes,
    const unsigned long long *__restrict__ dsts,
    unsigned char *__restrict__ outbase,
    unsigned nplanes, unsigned long long nelem, unsigned njobs)
{
    unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= nelem * (unsigned long long)njobs) return;
    unsigned j = (unsigned)(t / nelem);
    unsigned long long i = t - (unsigned long long)j * nelem;
    const unsigned char *p = planes + (unsigned long long)j * nplanes * nelem;
    unsigned char *o = outbase + dsts[j];
    for (unsigned k = 0; k < nplanes; ++k)
        o[i * nplanes + k] = p[(unsigned long long)k * nelem + i];
}

// CODEC_BF16: bfloat16's own field boundaries, not byte boundaries.
//
// This is where a byte interleave silently produces wrong weights. lmz splits a
// BF16 word so that plane A holds the 8 exponent bits and plane B holds the
// sign in bit 7 with the 7 mantissa bits below it -- which is not the high and
// low byte, and reassembling it as though it were passes every shape check and
// fails only on the bytes. Mirrors merge_bf16_scalar in lmz's lmzcore.c:
//
//     w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F)
//
// One launch for a whole batch, as above; every job here has exactly 2 planes.
extern "C" __global__ void lmz_merge_bf16_batch(
    const unsigned char *__restrict__ planes,
    const unsigned long long *__restrict__ dsts,
    unsigned char *__restrict__ outbase,
    unsigned long long nelem, unsigned njobs)
{
    unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= nelem * (unsigned long long)njobs) return;
    unsigned j = (unsigned)(t / nelem);
    unsigned long long i = t - (unsigned long long)j * nelem;
    const unsigned char *p = planes + (unsigned long long)j * 2ULL * nelem;
    unsigned char *o = outbase + dsts[j];
    unsigned a = p[i];
    unsigned b = p[nelem + i];
    unsigned w = ((b & 0x80u) << 8) | (a << 7) | (b & 0x7Fu);
    o[2 * i] = (unsigned char)(w & 0xFFu);
    o[2 * i + 1] = (unsigned char)(w >> 8);
}

// CODEC_BF16 as lmz actually writes it: exponent coded, sign+mantissa stored.
//
// On real weights the sign and mantissa measure 7.92 bits of 8, so lmz does not
// code them -- every BF16 chunk comes out with plane 0 an rANS stream and plane
// 1 raw bytes. The two therefore arrive in different buffers: plane 0 from the
// batch decoder's output, plane 1 uploaded as itself. Taking them from separate
// bases is what lets that happen without a device-to-device shuffle first.
extern "C" __global__ void lmz_merge_bf16_split(
    const unsigned char *__restrict__ aplanes,
    const unsigned char *__restrict__ bplanes,
    const unsigned long long *__restrict__ dsts,
    unsigned char *__restrict__ outbase,
    unsigned long long nelem, unsigned njobs)
{
    unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= nelem * (unsigned long long)njobs) return;
    unsigned j = (unsigned)(t / nelem);
    unsigned long long i = t - (unsigned long long)j * nelem;
    unsigned long long at = (unsigned long long)j * nelem + i;
    unsigned a = aplanes[at];
    unsigned b = bplanes[at];
    unsigned w = ((b & 0x80u) << 8) | (a << 7) | (b & 0x7Fu);
    unsigned char *o = outbase + dsts[j];
    o[2 * i] = (unsigned char)(w & 0xFFu);
    o[2 * i + 1] = (unsigned char)(w >> 8);
}

// The general byte-split merge: any mix of coded and stored planes.
//
// A split chunk's planes are coded independently, and lmz stores rather than
// codes any plane that is already near 8 bits of entropy -- so a chunk can have
// any subset coded. Coded planes arrive from the batch decoder's output and
// stored ones from a second buffer, both packed job-major, and `plane_src` says
// which buffer each plane came from and its rank within that job:
// non-negative is a coded rank, negative is -(stored rank + 1).
//
// Every job in a batch shares the mask by construction, so it is one small
// array read once per plane rather than per element.
extern "C" __global__ void lmz_merge_masked(
    const unsigned char *__restrict__ coded,
    const unsigned char *__restrict__ stored,
    const unsigned long long *__restrict__ dsts,
    const int *__restrict__ plane_src,
    unsigned char *__restrict__ outbase,
    unsigned nplanes, unsigned ncoded, unsigned nstored,
    unsigned long long nelem, unsigned njobs)
{
    unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= nelem * (unsigned long long)njobs) return;
    unsigned j = (unsigned)(t / nelem);
    unsigned long long i = t - (unsigned long long)j * nelem;
    unsigned char *o = outbase + dsts[j];
    for (unsigned k = 0; k < nplanes; ++k) {
        int s = plane_src[k];
        unsigned char v;
        if (s >= 0)
            v = coded[((unsigned long long)j * ncoded + (unsigned)s) * nelem + i];
        else
            v = stored[((unsigned long long)j * nstored + (unsigned)(-s - 1)) * nelem + i];
        o[i * nplanes + k] = v;
    }
}
