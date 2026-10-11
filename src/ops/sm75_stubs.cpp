// sm_75 (Turing) replacements for routes whose kernels require sm_80+ instructions.
//
// Turing has no bf16 or tf32 MMA and no double/quad-K MMA shapes (m16n8k16/m16n8k32), so the
// translation units owning these routes cannot be compiled there and are filtered out of
// ninfer_ops by the sm_75 branch in ops/CMakeLists.txt. These definitions keep the Op boundary
// link-complete and convert a mis-selected route into a precise runtime error instead of a link
// failure — the same pattern as fp8_sm86_stubs.cpp / nvfp4_sm86_stubs.cpp.
//
// Unavailable on sm_75:
//   - FP8 A16 attention-input GEMMs (bf16 MMA pipelines)
//   - NVFP4 A16 attention input (bf16 MMA)
//   - moe_experts_native (m16n8k32 INT8 MMA; Flash-Next expert banks)
//   - qsa_indexer (TF32 WMMA; Flash-Next n-gram block selection)
// The t2 / INT8 / Q4_Q5 / Q8 / BF16-dequant routes that ternary and groupwise-int artifacts
// use are compiled normally.

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan_unified.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/qsa_indexer.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

[[noreturn]] void reject_sm75() {
    throw std::runtime_error("this route requires an sm_80+ GPU (unavailable on Turing/sm_75)");
}

} // namespace

// --- Flash-Next native expert banks (m16n8k32 INT8 MMA) ----------------------------------------

Weight prepare_native_expert(const WeightInput& input, std::int32_t rows, std::int32_t columns,
                             bool integer_a8) {
    (void)input;
    (void)rows;
    (void)columns;
    (void)integer_a8;
    reject_sm75();
}

std::size_t moe_experts_native_workspace_bytes(std::int32_t tokens) {
    (void)tokens;
    reject_sm75();
}

void moe_experts_native(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        const Tensor& shared, const NativeMoeWeights& banks, const Tensor* parts,
                        WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    (void)m;
    (void)ids;
    (void)weights;
    (void)shared;
    (void)banks;
    (void)parts;
    (void)workspace;
    (void)y;
    (void)stream;
    reject_sm75();
}

// --- Flash-Next QSA indexer (TF32 WMMA) --------------------------------------------------------

void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, Tensor& pooled, Tensor& tail,
                        cudaStream_t stream) {
    (void)projection;
    (void)first_position;
    (void)weights;
    (void)eps;
    (void)pooled;
    (void)tail;
    (void)stream;
    reject_sm75();
}

std::size_t qsa_indexer_select_workspace_bytes(std::int32_t tokens, std::int32_t capacity) {
    (void)tokens;
    (void)capacity;
    reject_sm75();
}

void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const Tensor& pooled,
                        WorkspaceArena& workspace, Tensor& selected, Tensor& counts,
                        cudaStream_t stream) {
    (void)projection;
    (void)first_position;
    (void)weights;
    (void)eps;
    (void)pooled;
    (void)workspace;
    (void)selected;
    (void)counts;
    (void)stream;
    reject_sm75();
}

namespace detail {

// --- FP8 A16 attention input (bf16 MMA pipelines) ----------------------------------------------

void fp8_attn_input_a16_gemm_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                    Tensor&, cudaStream_t) {
    reject_sm75();
}

void fp8_attn_input_a16_small_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                         Tensor&, cudaStream_t) {
    reject_sm75();
}

// --- NVFP4 A16 attention input (bf16 MMA) ------------------------------------------------------

void nvfp4_attn_input_a16_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                 Tensor&, cudaStream_t) {
    reject_sm75();
}

namespace unified {

void fp8_attn_input_a16_gemm_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                    Tensor&, cudaStream_t) {
    reject_sm75();
}

void fp8_attn_input_a16_small_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                         Tensor&, cudaStream_t) {
    reject_sm75();
}

void nvfp4_attn_input_a16_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                 Tensor&, cudaStream_t) {
    reject_sm75();
}

} // namespace unified
} // namespace detail
} // namespace ninfer::ops
