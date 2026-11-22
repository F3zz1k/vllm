// Inkling2 NVFP4 grouped GEMM entry point (see moe_nvfp4.hpp).
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "moe_nvfp4.hpp"

namespace inkling2 {

// A (rows, K) bf16 grouped by expert (rows_per_expert), B (E, N, K/2) uint8 e2m1 pairs,
// S (E, N, K/16) uint8 e4m3, F (E,) fp32, D (rows, N) bf16.
void grouped_gemm_nvfp4(const at::Tensor& A, const at::Tensor& B, const at::Tensor& S,
                        const at::Tensor& F, at::Tensor& D, const at::Tensor& rows_per_expert,
                        int64_t N, int64_t K) {
  using namespace inkling2::moe;
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && D.scalar_type() == at::kBFloat16);
  TORCH_CHECK(B.dim() == 3 && B.size(1) == N && B.size(2) * 2 == K && B.element_size() == 1);
  TORCH_CHECK(S.dim() == 3 && S.size(1) == N && S.size(2) * kGroup == K && S.element_size() == 1);
  TORCH_CHECK(F.scalar_type() == at::kFloat && F.numel() == B.size(0));
  TORCH_CHECK(A.size(1) == K && D.size(1) == N && A.size(0) == D.size(0));
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && S.is_contiguous() && D.is_contiguous());
  TORCH_CHECK(K % 128 == 0 && N % 64 == 0, "K % 128 and N % 64");
  int E = B.size(0);
  auto& q = at::xpu::getCurrentXPUStream(A.device().index()).queue();
  at::Tensor atomic = at::zeros({1}, A.options().dtype(at::kInt));
  auto a = reinterpret_cast<const bfloat16_t*>(A.data_ptr());
  auto b = reinterpret_cast<const uint8_t*>(B.data_ptr());
  auto s = reinterpret_cast<const uint8_t*>(S.data_ptr());
  auto f = F.data_ptr<float>();
  auto d = reinterpret_cast<bfloat16_t*>(D.data_ptr());
  auto r = rows_per_expert.data_ptr<int>();
  auto at_ = atomic.data_ptr<int>();
  int avg_m = A.size(0) / E;
  if (avg_m <= 4) {
    launch<policy_m_8>(q, a, b, s, f, d, N, K, r, E, at_);
  } else if (avg_m <= 8) {
    launch<policy_m_16>(q, a, b, s, f, d, N, K, r, E, at_);
  } else if (avg_m <= 128) {
    launch<policy_m_32>(q, a, b, s, f, d, N, K, r, E, at_);
  } else {
    launch<policy_big>(q, a, b, s, f, d, N, K, r, E, at_);
  }
}

}  // namespace inkling2
