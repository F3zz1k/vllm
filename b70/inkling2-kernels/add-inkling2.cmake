
#
# _inkling2_C: Inkling2 relative-bias attention and NVFP4 expert GEMM (own kernel
# libraries; the vendor libraries are not touched).
#
if(INKLING2_ATTN_ENABLED
   AND BUILD_SYCL_TLA_KERNELS
   AND VLLM_XPU_ENABLE_XE2)
  message(STATUS "Enabling _inkling2_C extension.")
  add_subdirectory(csrc/xpu/inkling2/attn/xe_2)
  add_subdirectory(csrc/xpu/inkling2/moe/xe_2)
  define_gpu_extension_target(
    _inkling2_C
    DESTINATION
    vllm_xpu_kernels
    LANGUAGE
    ${VLLM_GPU_LANG}
    SOURCES
    "csrc/xpu/inkling2/inkling2_bindings.cpp"
    COMPILE_FLAGS
    ${VLLM_GPU_COMPILE_FLAGS}
    LINK_FLAGS
    ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES
    ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
    ${VLLM_XPU_INCLUDE_DIR}
    LIBRARIES
    inkling2_attn_kernels_xe_2
    inkling2_moe_kernels_xe_2
    USE_SABI
    3
    WITH_SOABI)
endif()
