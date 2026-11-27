// Inkling2 relative-bias paged attention: entry point (built into
// libinkling2_attn_kernels_xe_2, called by the _inkling2_C bindings).
#include <torch/all.h>

#include "csrc/utils.h"
#include "csrc/xpu/inkling2/attn/rel_context.h"
#include "fmha_xe2.h"
#include "paged_decode_xe2.h"

namespace inkling2 {

// q [T, Hq, D]; k/v paged views [num_blocks, block_size, Hk, D] (strided);
// out [T, Hq, D]; cu_seqlens_q [B+1] int32; seqused_k [B] int32;
// block_table [B, max_blocks] int32; rel bf16 [T, Hq, E] (column d = bias at
// backward distance d; strided rows allowed); lse fp32 [Hq, T] (natural log
// of the softmax denominator, without the sink: the caller applies
// out *= sigmoid(lse - sink)). window_left < 0: no sliding window, else keys
// at distance <= window_left are visible. num_splits: KV splits of a
// decode-only batch (max_seqlen_q == 1).
void rel_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   at::Tensor& out, const at::Tensor& cu_seqlens_q,
                   const at::Tensor& seqused_k, const at::Tensor& block_table,
                   int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
                   int64_t window_left, const at::Tensor& rel, at::Tensor& lse,
                   int64_t num_splits, const std::optional<at::Tensor>& k_scale,
                   const std::optional<at::Tensor>& v_scale) {
  TORCH_CHECK(q.dim() == 3 && q.stride(-1) == 1, "q: [T, Hq, D], contiguous D");
  TORCH_CHECK(rel.scalar_type() == at::kBFloat16 && rel.stride(-1) == 1,
              "rel: bf16 [T, Hq, E]");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() &&
                  lse.size(0) == q.size(1) && lse.size(1) == q.size(0),
              "lse: fp32 [Hq, T]");
  // fp8 KV (e4m3 / e5m2 views of the cache): per-tensor scales, one fp32 element each.
  // The kernels fold k_scale into the softmax scale, so the relative bias (added before
  // that scale, divided by softmax_scale) is exact only at k_scale == 1, which is what
  // the model passes.
  const bool fp8_kv = k.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                      k.scalar_type() == at::ScalarType::Float8_e5m2;
  TORCH_CHECK(v.scalar_type() == k.scalar_type(), "k and v caches: one dtype");
  TORCH_CHECK(!fp8_kv || (k_scale.has_value() && v_scale.has_value()),
              "fp8 KV needs k_scale and v_scale");
  if (fp8_kv) {
    TORCH_CHECK(k_scale->scalar_type() == at::kFloat && k_scale->numel() == 1 &&
                    v_scale->scalar_type() == at::kFloat && v_scale->numel() == 1,
                "k_scale / v_scale: one fp32 element each");
  }
  // const optionals are not assignable: construct them once
  std::optional<const at::Tensor> ks =
      fp8_kv ? std::optional<const at::Tensor>(*k_scale) : std::nullopt;
  std::optional<const at::Tensor> vs =
      fp8_kv ? std::optional<const at::Tensor>(*v_scale) : std::nullopt;
  auto& queue = vllm::xpu::vllmGetQueue(q.device().index());
  xe2::g_rel = {rel.data_ptr(), static_cast<int>(rel.stride(0)),
                static_cast<int>(rel.stride(1)), static_cast<int>(rel.size(2))};
  bool is_local = window_left >= 0;
  int wl = is_local ? static_cast<int>(window_left) : -1;
  int wr = is_local ? 0 : -1;
  std::optional<const at::Tensor> none;
  std::optional<at::Tensor> lse_opt = lse;
  if (max_seqlen_q > 1) {
    // chunk prefill serves every row (decode rows too): no mixed-batch split
    xe2::cutlass_chunk_prefill_xe2(
        queue, q, k, v, out, block_table, cu_seqlens_q, seqused_k,
        static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k), ks, vs,
        softmax_scale, none, wl, wr, /*is_varlen=*/true, /*is_paged=*/true,
        /*is_causal=*/true, is_local, /*is_sink=*/false, lse_opt, none);
  } else {
    int splits = static_cast<int>(std::max<int64_t>(1, num_splits));
    int T = q.size(0), Hq = q.size(1), D = q.size(2);
    at::Tensor tmp_out = splits == 1 ? out
                                     : at::empty({T, Hq * splits, D}, q.options());
    at::Tensor lse_accum = at::empty({T, Hq, splits}, q.options().dtype(at::kFloat));
    int eff_left = is_local ? wl : static_cast<int>(max_seqlen_k);
    int eff_right = is_local ? 0 : static_cast<int>(max_seqlen_k);
    std::optional<at::Tensor> no_t;
    xe2::cutlass_paged_decode_xe2(
        queue, q, k, v, out, tmp_out, lse_accum, block_table, cu_seqlens_q, seqused_k,
        static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k), ks, vs,
        softmax_scale, none, eff_left, eff_right, /*is_varlen=*/true,
        /*is_paged=*/true, /*is_causal=*/false, is_local, /*is_sink=*/false, splits,
        none, no_t, no_t, lse_opt);
  }
  xe2::g_rel = {};
}

}  // namespace inkling2
