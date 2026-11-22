# Inkling 2 Mini on Intel Arc Pro B70 (vLLM, XPU)

Text-only serving of [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Intel Arc Pro B70
(Xe2) with vLLM: model code in `vllm/models/inkling2/xpu/`, kernels in `b70/inkling2-kernels/`, image kit in
`b70/image/`. Work in progress; see "Not done yet".

## Tested on

- GPUs: Intel Arc Pro B70 (BMG-G31, 32 GB), 1, 2 and 4 cards; multi-card runs with direct GPU-to-GPU P2P through a
  PCIe switch (kernel `iommu=pt`, ACS redirect off).
- Host: Linux 7.2.9, `xe` kernel driver; Docker.
- Image software: Ubuntu 24.04, compute-runtime 26.35.39758.10, IGC 2.41.5, Level Zero 1.32, torch 2.14.0+xpu,
  triton 3.8.0+xpu, oneAPI 2026.1.1 runtime, vllm-xpu-kernels 0.1.16.2 + upstream `6b30c9b` + Lumnus kit patches.
- Base: the [Lumnus b70-flash-next](https://github.com/Lumnus/b70-flash-next) kit at `v0.32.0-b70.7` (vLLM v0.32.0
  with B70 patches); this branch adds the Inkling 2 model code on top.
- Checkpoint: the published NVFP4/MXFP8 checkpoint (experts NVFP4 in layers 0-19, MXFP8 in layers 20-23), max model
  length 32,768.

## Build

1. Environment image (once; ~50 min cold, memory peak is the kernel build, `MAX_JOBS=6`):
   ```bash
   git clone https://github.com/Lumnus/b70-flash-next kit && git -C kit checkout v0.32.0-b70.7
   git archive -o vllm-src.tar <commit of this branch>
   KIT=$PWD/kit ENV_TAG=vllm-xpu-b70env:0.1.16.2 bash b70/image/build.sh env vllm-src.tar
   KIT=$PWD/kit bash b70/image/build.sh env-update vllm-src.tar   # -> vllm-xpu-b70env:0.1.16.2-w4a8
   ```
2. Inkling 2 kernel library (`_inkling2_C`, ~5 min first build), with the env image's own toolchain:
   ```bash
   KIT=$PWD/kit OUT=$PWD/inkling2-out bash b70/inkling2-kernels/build.sh
   ```
3. Engine layer (about a minute), then the kernel library on top:
   ```bash
   KIT=$PWD/kit TAG=vllm-xpu:b70-inkling2-engine bash b70/image/build.sh vllm vllm-src.tar
   docker build -f b70/inkling2-kernels/Dockerfile.layer --build-arg ENGINE=vllm-xpu:b70-inkling2-engine \
     -t vllm-xpu:b70-inkling2 inkling2-out
   ```
   `build.sh` archives a commit, not a tree, refuses an existing tag and stops when host `MemAvailable` drops under
   `FLOOR_GIB` (12).

## Run

Common docker arguments and environment (all required on B70; the residency shim keeps Level Zero from pinning a
host copy of every multi-card allocation, copy offload off avoids hangs on a lost device, the `CCL_SYCL_*` keys stop
oneCCL from leaking host memory per batch shape):

```bash
COMMON=(--rm --network host --ipc=host --privileged --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path
  --shm-size 32g -v /path/to/Inkling-2-Mini:/models/inkling-2-mini:ro
  -e LD_PRELOAD=/opt/b70/libl0_owner_residency.so -e ZE_ENABLE_TRACING_LAYER=1
  -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 -e VLLM_USE_V2_MODEL_RUNNER=1
  -e NEOReadDebugKeys=1 -e ClDeviceGlobalMemSizeAvailablePercent=100
  -e EnableDeferBacking=1 -e DeferBackingMemoryPressurePercent=0
  -e CCL_SYCL_ALLREDUCE_TMP_BUF=1 -e CCL_SYCL_ALLGATHERV_TMP_BUF=1
  -e CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1 -e CCL_SYCL_BROADCAST_TMP_BUF=1
  -e ExperimentalCopyThroughLock=0 -e ExperimentalH2DCpuCopyThreshold=0 -e ExperimentalD2HCpuCopyThreshold=0
  -e VLLM_INKLING2_EXPERTS=nvfp4 -e VLLM_INKLING2_DENSE=int8 -e VLLM_INKLING2_HEAD=int8)
SERVE=(serve /models/inkling-2-mini --served-model-name inkling-2-mini --trust-remote-code
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 32)
```

`ZE_AFFINITY_MASK` takes Level Zero (SYCL) device indices, which follow ascending PCIe address, not DRM card numbers;
keep a TP group behind one PCIe switch.

```bash
# TP=1
docker run "${COMMON[@]}" -e ZE_AFFINITY_MASK=0 --entrypoint vllm vllm-xpu:b70-inkling2 \
  "${SERVE[@]}" --tensor-parallel-size 1 --max-num-batched-tokens 8192
# TP=2
docker run "${COMMON[@]}" -e ZE_AFFINITY_MASK=0,1 --entrypoint vllm vllm-xpu:b70-inkling2 \
  "${SERVE[@]}" --tensor-parallel-size 2 --max-num-batched-tokens 16384
# TP=4
docker run "${COMMON[@]}" -e ZE_AFFINITY_MASK=0,1,2,3 --entrypoint vllm vllm-xpu:b70-inkling2 \
  "${SERVE[@]}" --tensor-parallel-size 4 --max-num-batched-tokens 16384
```

TP shards the attention heads; experts use EP = TP (each card holds 128 / TP experts). XPU graphs are on by default.

DSpark speculative decoding with the checkpoint's bundled drafter: add
`--speculative-config '{"method":"dspark","num_speculative_tokens":7}'` (on one card use
`--max-num-batched-tokens 4096 --gpu-memory-utilization 0.92` to leave room for the draft).

## Format settings

| variable | values (default first) | effect |
|---|---|---|
| `VLLM_INKLING2_EXPERTS` | `fp8`, `nvfp4`, `bf16` | routed experts of layers 0-19. `nvfp4` (recommended) runs the checkpoint's NVFP4 bytes directly in `_inkling2_C` (exact, fastest); `fp8` is a per-expert FP8 conversion (~1.4% weight error); `bf16` is exact but does not fit one card |
| `VLLM_INKLING2_MXFP8` | `fp8`, `bf16` | experts of layers 20-23 (MXFP8 in the checkpoint) |
| `VLLM_INKLING2_DENSE` | `bf16`, `int8` | attention projections; `int8` is per-row W8A8 (faster decode, top-1 -0.1 to -0.2 pt) |
| `VLLM_INKLING2_HEAD` | `bf16`, `int8`, `fp32` | output layer, same scheme; `fp32` is the config's head dtype (top-1 within noise of `bf16`, ~7 ms/token slower on one card) |
| `VLLM_INKLING2_PREFILL_EXPERTS` | `nvfp4`, `fp8`, `bf16` | experimental: a second expert copy used only for batches of 64+ tokens. `fp8` at TP=2: prefill +4-6%, KV memory per card 13.75 -> 4.06 GiB; not recommended there. `bf16` untested |

`--max-num-batched-tokens`: 8,192 at TP=1 (16,384 does not fit beside the weights on one card), 16,384 at TP=2/4.

## Measured (32k context, int8 dense + head, NVFP4 experts, XPU graphs)

| TP | decode, 1 stream @32k | 32 streams | prefill (30k prompt) | top-1 vs BF16 reference |
|--:|--:|--:|--:|--:|
| 1 | 55 t/s | n/m | 4.0k t/s | 0.995 |
| 2 | 63 t/s | 896 t/s | 6.2k t/s | 0.996 |
| 4 | 77 t/s | 1,198 t/s | 8.0k t/s | 0.995 |

BF16 dense: 48 / 59 / 75 t/s decode at TP=1/2/4, top-1 0.996-0.998.

DSpark (TP=1, int8 dense + head, greedy, 256 tokens, the six prompts of the decode benchmark in
[mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu)):

| workload | no speculation | DSpark k=7 | acceptance length |
|---|--:|--:|--:|
| task prompts, reasoning on | 64.4 t/s | 107.7 t/s | 3.23 |
| task prompts, reasoning off | 64.5 t/s | 81.4 t/s | 2.67 |
| short chat turns, reasoning off | 64.6 t/s | 63.8 t/s | 1.82 |

Speculation is a net loss only on short chat with reasoning off (as that project found); the
config is per server, so a deployment that serves mostly short chat can leave it off. Top-1 is the agreement of the prompt-token argmax
with a transformers BF16 reference over 4,421 tokens (20 prompts up to 2,600 tokens).

## Known limits

- KV pages must be 64 tokens (vLLM's XPU default): the SYCL attention kernels are built for that page size; other
  sizes fall back to a slower Triton path.
- Prefix caching works (the short-conv inputs are a per-sequence recurrent state, cached in vLLM's align mode).
- Not done yet: vision (not ported, text only), FP8 / INT8 KV cache, CPU KV offload (not validated), TP above
  4, DSpark at TP > 1 (not tested), other GPUs.
- Decode has ~2 ms per token of small-kernel overhead still being fused.

## Credits

- Model code, DSpark draft and speculator: the upstream vLLM Inkling 2 implementation (NVIDIA), ported here.
- [mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu), an independent single-B70 port,
  used as a reference and benchmark (no code taken): its decode benchmark prompts, and its finding that DSpark
  does not pay on short chat with reasoning off. At the shapes here, its Triton expert path measured 183 / 1,970 /
  24,475 us per call at 1 / 16 / 4,096 tokens against 122 / 670 / 9,521 us for the grouped GEMM in this branch,
  and its decode attention 83 / 436 us (local @1k / global @32k) against 46 / 272 us.
