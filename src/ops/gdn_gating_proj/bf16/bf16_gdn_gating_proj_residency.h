#pragma once

// Shared per-SM residency for the BF16 GDN gating cooperative kernels.
//
// The compiled residency figures in bf16_gdn_gating_proj_plan.cpp were measured on sm_86, whose SM
// has 100 KiB of shared memory (two 40 KiB 27B CTAs). Turing's SM has 64 KiB, so the same kernel
// seats one CTA, and a cooperative grid the sm_86 figure admits is rejected by the driver with
// cudaErrorCooperativeLaunchTooLarge on the first wide prefill. The host has to derive its launch
// geometry from the device the kernel will actually run on -- the same split PrismML uses for its
// GDN column count (kernel on __CUDA_ARCH__, host on the highest compiled arch) -- so query the
// device and clamp instead of trusting the sm_86 constant.
//
// NINFER_SMEM_PER_SM_BYTES overrides the query so a narrow part's residency (Turing: 65536) can be
// reproduced on a wider one, exactly as NINFER_MAX_DYN_SMEM_BYTES does for per-block dynamic shared
// memory (see NINFER-T4-HANDOFF.md section 4.7).

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

inline std::int32_t device_shared_memory_per_sm() noexcept {
    if (const char* value = std::getenv("NINFER_SMEM_PER_SM_BYTES")) {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0) { return static_cast<std::int32_t>(parsed); }
    }
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<std::int32_t>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCachedDevices) {
        return 64 * 1024; // the smallest supported part
    }
    const std::int32_t known =
        cache[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (known > 0) { return known; }
    int per_sm = 0;
    if (cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) !=
            cudaSuccess ||
        per_sm <= 0) {
        per_sm = 64 * 1024;
    }
    cache[static_cast<std::size_t>(device)].store(static_cast<std::int32_t>(per_sm),
                                                  std::memory_order_relaxed);
    return static_cast<std::int32_t>(per_sm);
}

// How many CTAs of `cta_smem_bytes` one SM's shared memory admits; never fewer than one.
inline std::int32_t shared_memory_ctas_per_sm(std::int32_t cta_smem_bytes) noexcept {
    const std::int32_t cap = device_shared_memory_per_sm() / cta_smem_bytes;
    return cap < 1 ? 1 : cap;
}

} // namespace ninfer::ops::detail
