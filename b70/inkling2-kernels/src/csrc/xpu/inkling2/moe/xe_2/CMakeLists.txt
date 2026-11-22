cmake_minimum_required(VERSION 3.18)

# Inkling2 NVFP4 routed-expert grouped GEMM (vendor xe_2 4-bit GEMM, NVFP4 scales).
add_xe2_kernel_library(inkling2_moe_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
