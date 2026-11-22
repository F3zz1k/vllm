/***************************************************************************************************
 * Inkling2 NVFP4 grouped GEMM for the routed experts (W4A16, BF16 activations).
 *
 * Derived from vllm-xpu-kernels csrc/xpu/grouped_gemm/xe_2 (gemm_xe2.hpp, grouped_gemm_xe2.hpp,
 * grouped_gemm_xe2_interface.hpp), Copyright (C) 2025 Intel Corporation, BSD-3-Clause; the
 * vendor licence text is kept in the README of this directory tree.
 *
 * Differences from the vendor MXFP4 path:
 *   - one E4M3 scale per 16 values (NVFP4), stored [E, N, K/16] as bytes. The BF16 DPAS atom
 *     sums K = 16, so each 32-wide k-tile holds two MMA k-blocks and each block gets its own
 *     scale: exact, no change to the tile shapes;
 *   - a per-expert fp32 factor (amax / outer denominator) applied to the accumulator;
 *   - BF16 activations only, no bias.
 * B is [E, N, K/2] bytes, value 2i in the low nibble (the checkpoint order, as for MXFP4).
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/numeric_types.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace inkling2 {
inline namespace inkling2_xpu_xe2 {
namespace moe {

using namespace cute;

struct policy_base {
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};
struct policy_big : policy_base {
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};
struct policy_m_8 : policy_base {
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};
struct policy_m_16 : policy_base {
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};
struct policy_m_32 : policy_base {
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

CUTE_DEVICE bfloat16_t scale_bf16(bfloat16_t& x, float& y) {
  uint16_t z = sycl::bit_cast<uint16_t>(x);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  asm("{\n"
      ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
      ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
      "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
      "}\n"
      : "+rw"(z)
      : "rw"(y));
#endif
  return sycl::bit_cast<bfloat16_t>(z);
}

// E4M3 (positive) -> fp32 without denormals: normal values rebias the exponent
// field (+120); subnormals (e == 0) are m * 2^-9.
CUTE_DEVICE float e4m3_to_float(uint32_t b) {
  uint32_t v = b & 0x7Fu;
  float normal = sycl::bit_cast<float>((v << 20) + (120u << 23));
  float sub = static_cast<float>(v & 7u) * 0x1p-9f;
  return (v >> 3) ? normal : sub;
}

static constexpr int kGroup = 16;  // NVFP4 scale group

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC, class ATensor,
          class BTensor, class DTensor, class TiledMMA>
CUTE_DEVICE void gemm_nvfp4(ATensor const& A,  // (M,K) bf16
                            BTensor const& B,  // (N,K) e2m1
                            const uint8_t* Scales,  // (N, K/16) e4m3
                            float factor, DTensor& C,  // (M,N)
                            Coord<int, int, cute::Underscore, int> blk_coord,
                            TiledMMA const& mma) {
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);
  static constexpr auto SG_N = tile_n / ATOM_N;
  static constexpr int prefetch_dist = 6;

  static constexpr auto thr_N = get<1>(tCrB.shape());
  static constexpr auto channel_num = get<0>(get<0>(tCrB.shape()));
  // MMA k-blocks per k-tile: the BF16 DPAS atom has K = 16 = one NVFP4 scale group.
  static constexpr int KB = tile_k / kGroup;
  static_assert(KB * kGroup == tile_k, "tile_k must be a multiple of the NVFP4 group");
  const int per_block = tCrB.size() / thr_N / channel_num / KB;

  auto n_tile_start = wg_n * tile_n;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / kGroup;
  int x_idx = sg_local_id / channel_num;

  float scales[KB][thr_N * channel_num];
  // 8 scales (128 values of K) per row in one 8-byte load, every 128 / tile_k k-tiles
  static constexpr int TILES_PER_LOAD = 128 / tile_k;
  uint64_t sraw[thr_N * channel_num];

  clear(tCrC);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile % TILES_PER_LOAD == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < thr_N; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < channel_num; ++c) {
          int real_idx = x_idx + c * (sg_local_range / channel_num);
          int sg_local_n = n * sg_local_range + real_idx;
          sraw[n * channel_num + c] = *reinterpret_cast<const uint64_t*>(
              Scales + (n_tile_start + n_sg_start + sg_local_n) * group_num + k_tile * KB);
        }
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      int shift = ((k_tile % TILES_PER_LOAD) * KB + kb) * 8;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < thr_N * channel_num; ++j) {
        scales[kb][j] = e4m3_to_float(static_cast<uint32_t>(sraw[j] >> shift));
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    CUTLASS_PRAGMA_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < thr_N; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < channel_num; ++c) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < per_block; ++i) {  // per_block is a compile-time value
            tCrB(cute::tuple(c, _), n, kb)[i] =
                scale_bf16(tCrB(cute::tuple(c, _), n, kb)[i], scales[kb][n * channel_num + c]);
          }
        }
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) {
    tCrC(i) *= factor;
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(make_gmem_ptr(ptr), make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(make_gmem_ptr(ptr), make_layout(shape, make_stride(c, _1{})));
}

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD, class TiledMMA>
CUTE_DEVICE void moe_gemm_nvfp4(const bfloat16_t* Activations, const uint8_t* Weights,
                                const uint8_t* Scales, const float* Factors,
                                bfloat16_t* Outputs, TiledMMA const& mma,
                                const int* rows_per_expert, const int32_t num_experts,
                                const int32_t gemm_n, const int32_t gemm_k,
                                int32_t* atomic_buffer,
                                const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  int pre_rows = 0;
  int pre_tiles = 0;
  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cum_rows = pre_rows + gemm_m;
    int cum_tiles = (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;
    if (group_m_id >= cum_tiles) {
      pre_rows = cum_rows;
      pre_tiles = cum_tiles;
      continue;
    }
    int64_t B_offset = static_cast<int64_t>(i) * gemm_n * gemm_k / 2;
    auto A_tensor = make_moe_tensor<const bfloat16_t, 'R'>(Activations + pre_rows * gemm_k,
                                                           gemm_m, gemm_k);
    // B stored (N, K) row-major = 'C' for the (N, K) view expected by the copy
    auto B_tensor = make_moe_tensor<const float_e2m1_t, 'R'>(
        reinterpret_cast<const float_e2m1_t*>(Weights + B_offset), gemm_n, gemm_k);
    auto D_tensor = make_moe_tensor<bfloat16_t, 'R'>(Outputs + pre_rows * gemm_n, gemm_m, gemm_n);
    const uint8_t* S = Scales + B_offset * 2 / kGroup;
    float factor = Factors[i];

    while (group_m_id < cum_tiles) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = group_m_id - pre_tiles;
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);
      gemm_nvfp4<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(A_tensor, B_tensor, S, factor,
                                                                 D_tensor, tile_coord, mma);
      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      sycl::group_barrier(item.get_group());
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cum_rows;
    pre_tiles = cum_tiles;
  }
}

template <class policy>
class NvFp4GemmName;

template <class policy>
void launch(sycl::queue& q, const bfloat16_t* A, const uint8_t* B, const uint8_t* S,
            const float* F, bfloat16_t* D, int gemm_n, int gemm_k, const int* rows_per_expert,
            int num_experts, int32_t* atomic_buffer) {
  auto op = XE_DPAS_TT<8, float, bfloat16_t>{};
  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  auto mma = MMA{};
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto threads = size(mma);
  static constexpr int MaxThreadsPerSM = 512;
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / threads, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  using CA = typename policy::GmemTiledCopyA;
  using CB = typename policy::GmemTiledCopyB;
  using CD = typename policy::GmemTiledCopyD;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<NvFp4GemmName<policy>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          moe_gemm_nvfp4<CA, CB, CD>(A, B, S, F, D, mma, rows_per_expert, num_experts, gemm_n,
                                     gemm_k, atomic_buffer, local_mem);
        });
  });
}

}  // namespace moe
}  // namespace inkling2_xpu_xe2
}  // namespace inkling2
