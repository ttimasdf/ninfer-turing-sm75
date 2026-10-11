#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NINFER_MMA_PRE_SM80 1
#else
#define NINFER_MMA_PRE_SM80 0
#endif

namespace ninfer::ops {

#if NINFER_MMA_PRE_SM80
// Warp-uniform power-of-two scaling for the pre-sm_80 f16-MMA emulations of the wider-range
// bf16/tf32 shapes. bf16 and tf32 carry the FP32 exponent range; decay panels in the GDN
// family routinely exceed the FP16 range on both ends. Scaling by a power of two is exact
// on the bit level, and a single scale that pins the warp-max magnitude near 2^14 keeps
// every term the FP32 accumulator could register representable in FP16: values flushed as
// subnormal are below 2^-24 of the fragment max, i.e. below accumulation significance.
__device__ __forceinline__ int mma_warp_max_i32(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const int o = __shfl_xor_sync(0xffffffffu, v, off);
        v = o > v ? o : v;
    }
    return v;
}

// biased exponent of an f32's bits; zeros map to -127 (harmless for the max)
__device__ __forceinline__ int mma_exponent(unsigned bits) {
    return (bits == 0) ? -127 : static_cast<int>((bits >> 23) & 0xff) - 127;
}

// exact x * 2^e for finite x (bit-level exponent add)
__device__ __forceinline__ float mma_scale_exp2(float x, int e) {
    unsigned b = __float_as_uint(x);
    const int exp = static_cast<int>((b >> 23) & 0xff);
    if (exp == 0 || exp == 255) return x;  // zero/subnormal/inf/nan: leave (flush is fine below significance)
    const int ne = exp + e;
    if (ne <= 0) return 0.0f;              // fully below the scaled window
    if (ne >= 255) return __uint_as_float(0x7f800000u | (b & 0x80000000u)); // overflow stays inf-signed
    return __uint_as_float((b & 0x807fffffu) | (static_cast<unsigned>(ne) << 23));
}
#endif


// The four emulations below split the sm_80 MMA shapes into the shapes Turing (sm_75)
// provides: f16/bf16 m16n8k16 -> 2x m16n8k8.f16 (bf16 operands converted with a
// +/-65504 clamp), s8 m16n8k32 -> 4x m8n8k16.s8. All splits are register-local: the
// sm_80 fragments subdivide along k exactly into the sm_75 fragments, and the m8n8
// accumulator rows g / g+8 land on the same threads as the m16n8 accumulator pairs.
// Fragment layouts verified empirically (probe receipts in the sm75 port workspace);
// they are arch-stable per the PTX ISA.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ unsigned bf16x2_to_f16x2_clamped(unsigned v) {
    const __nv_bfloat162 bf = *reinterpret_cast<const __nv_bfloat162*>(&v);
    float lo = __bfloat162float(bf.x);
    float hi = __bfloat162float(bf.y);
    lo = fminf(fmaxf(lo, -65504.0f), 65504.0f);
    hi = fminf(fmaxf(hi, -65504.0f), 65504.0f);
    // two single cvt.rn.f16.f32 + manual pack: the two-input f16x2 pack cvt is sm_80+
    const unsigned l16 = static_cast<unsigned>(__half_as_ushort(__float2half_rn(lo)));
    const unsigned h16 = static_cast<unsigned>(__half_as_ushort(__float2half_rn(hi)));
    return l16 | (h16 << 16);
}
/* macro defined at file top */
#endif

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if NINFER_MMA_PRE_SM80
    // 2x m16n8k8.f16 over scaled operands ({a0,a1}|b0 cover k 0..7, {a2,a3}|b1 k 8..15).
    // bf16 mantissa (8 bits) is exact in f16; only the exponent range needs a warp-uniform
    // power-of-two rescale so the max magnitude lands at ~2^14 (see mma_scale_exp2). Terms
    // lost to f16 subnormal flush sit below FP32 accumulation significance.
    int ea = mma_exponent(a0);
    ea = ::max(ea, mma_exponent(a1)); ea = ::max(ea, mma_exponent(a2)); ea = ::max(ea, mma_exponent(a3));
    int eb = mma_exponent(b0); eb = ::max(eb, mma_exponent(b1));
    ea = mma_warp_max_i32(ea); eb = mma_warp_max_i32(eb);
    const int sa = (ea >= 254 || ea <= -126) ? 0 : (ea > 14 ? ea - 14 : 0);
    const int sb = (eb >= 254 || eb <= -126) ? 0 : (eb > 14 ? eb - 14 : 0);
    auto bf16x2_scaled_f16x2 = [&](unsigned v, int e) -> unsigned {
        const float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xffff0000u);
        const float ls = mma_scale_exp2(lo, -e), hs = mma_scale_exp2(hi, -e);
        const unsigned l16 = static_cast<unsigned>(__half_as_ushort(__float2half_rn(ls)));
        const unsigned h16 = static_cast<unsigned>(__half_as_ushort(__float2half_rn(hs)));
        return l16 | (h16 << 16);
    };
    const unsigned a0h = bf16x2_scaled_f16x2(a0, sa), a1h = bf16x2_scaled_f16x2(a1, sa);
    const unsigned a2h = bf16x2_scaled_f16x2(a2, sa), a3h = bf16x2_scaled_f16x2(a3, sa);
    const unsigned b0h = bf16x2_scaled_f16x2(b0, sb), b1h = bf16x2_scaled_f16x2(b1, sb);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(s0), "+f"(s1), "+f"(s2), "+f"(s3)
                 : "r"(a0h), "r"(a1h), "r"(b0h));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(s0), "+f"(s1), "+f"(s2), "+f"(s3)
                 : "r"(a2h), "r"(a3h), "r"(b1h));
    const float undo = mma_scale_exp2(1.0f, sa + sb);
    c0 += s0 * undo; c1 += s1 * undo; c2 += s2 * undo; c3 += s3 * undo;
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if NINFER_MMA_PRE_SM80
    // {a0,a1}|b0 cover k 0..7, {a2,a3}|b1 cover k 8..15: two k8 steps, same fragments.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if NINFER_MMA_PRE_SM80
    // Same k split as mma_f16; d0 keeps the row-gid pair, d1 the row-gid+8 pair.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if NINFER_MMA_PRE_SM80
    // 4x m8n8k16: {a0,a1}|b0 cover rows g/g+8 over k 0..15, {a2,a3}|b1 the same rows over
    // k 16..31. The m8n8 accumulator's row-g / col-2*tig pairs land exactly on the m16n8
    // accumulator's c0/c1 (row g) and c2/c3 (row g+8). Integer-exact.
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a2), "r"(b1));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if NINFER_MMA_PRE_SM80
    // Turing: tf32-bits operands carry the FP32 exponent range; rescale warp-uniformly so
    // the max magnitude lands at ~2^14, then one m16n8k8.f16 per step. A tf32 value has a
    // 10-bit mantissa, exact in f16, so the scaled product needs no hi/lo compensation;
    // terms flushed below f16 subnormal are below FP32 accumulation significance.
    // Layout (probed): a0=(row g,k 2t), a1=(row g+8,k 2t), a2=(row g,k 2t+1), a3=(row g+8,
    // k 2t+1); b0=(col g,k 2t), b1=(col g,k 2t+1); f16 m16n8k8 A halves are k {2t, 2t+1}.
    int ea = mma_exponent(a0); ea = ::max(ea, mma_exponent(a1));
    ea = ::max(ea, mma_exponent(a2)); ea = ::max(ea, mma_exponent(a3));
    int eb = mma_exponent(b0); eb = ::max(eb, mma_exponent(b1));
    ea = mma_warp_max_i32(ea); eb = mma_warp_max_i32(eb);
    const int sa = (ea >= 254 || ea <= -126) ? 0 : (ea > 14 ? ea - 14 : 0);
    const int sb = (eb >= 254 || eb <= -126) ? 0 : (eb > 14 ? eb - 14 : 0);
    auto tf32_scaled_f16 = [&](unsigned bits, int e) -> unsigned {
        return static_cast<unsigned>(__half_as_ushort(
            __float2half_rn(mma_scale_exp2(__uint_as_float(bits & 0xffffe000u), -e))));
    };
    const unsigned a0h = tf32_scaled_f16(a0, sa), a1h = tf32_scaled_f16(a1, sa);
    const unsigned a2h = tf32_scaled_f16(a2, sa), a3h = tf32_scaled_f16(a3, sa);
    const unsigned b0h = tf32_scaled_f16(b0, sb), b1h = tf32_scaled_f16(b1, sb);
    const unsigned ah = a0h | (a2h << 16), al = a1h | (a3h << 16), bh = b0h | (b1h << 16);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(s0), "+f"(s1), "+f"(s2), "+f"(s3)
                 : "r"(ah), "r"(al), "r"(bh));
    const float undo = mma_scale_exp2(1.0f, sa + sb);
    c0 += s0 * undo; c1 += s1 * undo; c2 += s2 * undo; c3 += s3 * undo;
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
