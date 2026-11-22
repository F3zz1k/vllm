# Inkling2 XPU kernels (Xe2 / Arc Pro B70)

Built as their own extension `_inkling2_C` with two kernel libraries, so the vendor libraries in the image stay
untouched and no other model's path changes.

- `attn/`: paged relative-position-bias attention (decode and chunked prefill) with per-token sinks, derived from the
  vllm-xpu-kernels `6b30c9b` + Lumnus kit attention (namespace `inkling2::`). 64-token KV pages only.
- `moe/xe_2/`: NVFP4 grouped GEMM for the routed experts: reads the checkpoint bytes directly (e2m1 pairs, an E4M3
  scale per 16 values applied per 16-wide MMA k-block, per-expert factor in the epilogue), no dequantised copy.
- `add-inkling2.cmake`, `top-level-cmake.patch`: the CMake hook (option `INKLING2_ATTN_ENABLED`).
- `build.sh`: standalone build in a toolchain image (`Dockerfile.toolchain`); `Dockerfile.layer` adds the result to an
  engine image. Steps: `../README-inkling2.md`.
