// _inkling2_C: Inkling2 XPU ops (relative-bias paged attention).
#include "core/registration.h"

#include <torch/library.h>

namespace inkling2 {
void rel_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   at::Tensor& out, const at::Tensor& cu_seqlens_q,
                   const at::Tensor& seqused_k, const at::Tensor& block_table,
                   int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
                   int64_t window_left, const at::Tensor& rel, at::Tensor& lse,
                   int64_t num_splits);
void grouped_gemm_nvfp4(const at::Tensor& A, const at::Tensor& B, const at::Tensor& S,
                        const at::Tensor& F, at::Tensor& D, const at::Tensor& rows_per_expert,
                        int64_t N, int64_t K);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rel_attention(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor cu_seqlens_q, "
      "Tensor seqused_k, Tensor block_table, int max_seqlen_q, int max_seqlen_k, "
      "float softmax_scale, int window_left, Tensor rel, Tensor(b!) lse, "
      "int num_splits) -> ()");
  m.impl("rel_attention", at::kXPU, &inkling2::rel_attention);
  m.def(
      "grouped_gemm_nvfp4(Tensor A, Tensor B, Tensor S, Tensor F, Tensor(a!) D, "
      "Tensor rows_per_expert, int N, int K) -> ()");
  m.impl("grouped_gemm_nvfp4", at::kXPU, &inkling2::grouped_gemm_nvfp4);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
