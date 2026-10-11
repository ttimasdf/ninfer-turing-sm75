#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

// Pre-sm_80 software fallback: synchronous load through registers. The copy is complete
// when cp_async returns, so cp_commit is a no-op. cp_wait degrades to a block barrier:
// the stage rings in this tree issue cp_* block-uniformly, which __syncthreads tracks
// correctly even for threads that left the loop early (exited threads count as arrived
// since Volta). The issue-then-compute overlap of the rings is preserved: loads for stage
// N+k land in a different buffer than the one being computed, and the barrier before reuse
// sits exactly where cp_wait was.
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Bytes == 16) {
        store_vec(smem_dst, load_vec<uint4>(reinterpret_cast<const uint4*>(gmem_src)));
    } else if constexpr (Bytes == 8) {
        store_vec(smem_dst, load_vec<uint2>(reinterpret_cast<const uint2*>(gmem_src)));
    } else {
        store_vec(smem_dst, load_vec<unsigned int>(reinterpret_cast<const unsigned int*>(gmem_src)));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if (src_bytes >= Bytes) {
        cp_async<Bytes>(smem_dst, gmem_src);
        return;
    }
    const unsigned char* src = static_cast<const unsigned char*>(gmem_src);
    unsigned char tmp[Bytes];
#pragma unroll
    for (int i = 0; i < Bytes; ++i) {
        tmp[i] = (i < src_bytes) ? src[i] : static_cast<unsigned char>(0);
    }
    if constexpr (Bytes == 16) {
        uint4 v;
        v.x = (tmp[0] | (tmp[1] << 8) | (tmp[2] << 16) | (tmp[3] << 24));
        v.y = (tmp[4] | (tmp[5] << 8) | (tmp[6] << 16) | (tmp[7] << 24));
        v.z = (tmp[8] | (tmp[9] << 8) | (tmp[10] << 16) | (tmp[11] << 24));
        v.w = (tmp[12] | (tmp[13] << 8) | (tmp[14] << 16) | (tmp[15] << 24));
        store_vec(smem_dst, v);
    } else if constexpr (Bytes == 8) {
        uint2 v;
        v.x = (tmp[0] | (tmp[1] << 8) | (tmp[2] << 16) | (tmp[3] << 24));
        v.y = (tmp[4] | (tmp[5] << 8) | (tmp[6] << 16) | (tmp[7] << 24));
        store_vec(smem_dst, v);
    } else {
        store_vec(smem_dst,
                  tmp[0] | (tmp[1] << 8) | (tmp[2] << 16) | (tmp[3] << 24));
    }
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    __syncthreads();
}

#else // native (sm_80+) paths — unchanged upstream code

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

#endif // __CUDA_ARCH__ < 800

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
