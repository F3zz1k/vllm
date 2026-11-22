cmake_minimum_required(VERSION 3.18)

# Inkling2 relative-bias attention: the vendor chunk-prefill / paged-decode
# kernels (copied) with the bias added, built only for the Inkling2 shapes.
set(VLLM_CHUNK_PREFILL_CONFIG
    "${CMAKE_CURRENT_LIST_DIR}/../kernel_configs/inkling2_chunk_prefill.conf")
set(VLLM_PAGED_DECODE_CONFIG
    "${CMAKE_CURRENT_LIST_DIR}/../kernel_configs/inkling2_paged_decode.conf")

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

add_xe2_kernel_library(inkling2_attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
