# Inkling 2 Mini on Intel Arc Pro B70 (vLLM, XPU)

Text and image serving of [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Intel Arc Pro B70
(Xe2) with vLLM: model code in `vllm/models/inkling2/xpu/`, kernels in `b70/inkling2-kernels/`, image kit in
`b70/image/`.

## Tested on

- GPUs: Intel Arc Pro B70 (BMG-G31, 32 GB), 1, 2 and 4 cards; multi-card runs with direct GPU-to-GPU P2P through a
  PCIe switch (kernel `iommu=pt`, ACS redirect off).
- Host: Linux 7.2.9, `xe` kernel driver; Docker.
- Image software: Ubuntu 24.04, compute-runtime 26.35.39758.10, IGC 2.41.5, Level Zero 1.32, torch 2.14.0+xpu,
  triton 3.8.0+xpu, oneAPI 2026.1.1 runtime, vllm-xpu-kernels 0.1.16.2 + upstream `6b30c9b` + Lumnus kit patches.
- Base: the [Lumnus b70-flash-next](https://github.com/Lumnus/b70-flash-next) kit at `v0.32.0-b70.7` (vLLM v0.32.0
  with B70 patches); this branch adds the Inkling 2 model code on top.
- Checkpoint: the published NVFP4/MXFP8 checkpoint (experts NVFP4 in layers 0-19, MXFP8 in layers 20-23), context
  tested to 512k tokens.

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
  -e VLLM_INKLING2_DENSE=int8 -e VLLM_INKLING2_HEAD=int8 -e VLLM_INKLING2_MXFP8=fp8)
SERVE=(serve /models/inkling-2-mini --served-model-name inkling-2-mini --trust-remote-code
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 32)
```

Context fits up to ~64k on one card, ~393k at TP=2 and 1M at TP=4 (512k tested).

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
| `VLLM_INKLING2_EXPERTS` | `nvfp4`, `fp8`, `bf16` | experts of layers 0-19; `nvfp4` runs the checkpoint's bytes directly (exact, fastest); `fp8` re-quantises (lossy) |
| `VLLM_INKLING2_MXFP8` | `bf16`, `mxfp8`, `fp8` | experts of layers 20-23 with `EXPERTS=nvfp4`: `bf16` exact (2 GiB/layer); `mxfp8` the checkpoint bytes, exact, 1 GiB/layer, ~2% slower DSpark; `fp8` one scale per expert instead of per-32 blocks (lossy, 1 GiB/layer, fastest) |
| `VLLM_INKLING2_DENSE` | `bf16`, `int8` | attention projections; `int8` = per-row W8A8, faster decode, top-1 -0.1 to -0.2 pt |
| `VLLM_INKLING2_HEAD` | `bf16`, `int8`, `fp32` | output layer, same scheme; `bf16` matches the reference implementation, `fp32` (the config's dtype) costs ~9 ms/token |

The defaults reproduce the checkpoint (exact). One card with DSpark needs `--gpu-memory-utilization 0.92` for 32k
(or `MXFP8=mxfp8` at 0.85). The table below uses the fast set: `DENSE=int8 HEAD=int8 MXFP8=fp8`. It is
slightly lossy: top-1 agreement with the BF16 reference 0.994-0.995 against 0.996-0.998 for the exact defaults (the
reference itself in BF16 vs FP32 on CPU: 0.994), for about +15% plain decode on one card; needle results unchanged.

## Measured (230 W cap, int8 dense + head, NVFP4 experts, layers 20-23 fp8, XPU graphs)

Decode t/s without speculation ("plain") and with DSpark speculative decoding (k=7). Chat, greedy, 256 tokens,
reasoning on; short prompts are the six task prompts of
[mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu)'s decode bench; "@N" = one fresh
prompt of N tokens of varied filler plus a task; 32 streams = aggregate t/s on short prompts.

| | TP=1 plain | TP=1 DSpark | TP=2 plain | TP=2 DSpark | TP=4 plain | TP=4 DSpark |
|---|--:|--:|--:|--:|--:|--:|
| short prompts | 65.0 | 97.7 | 78.0 | 134.1 | 82.7 | 160.7 |
| @32k | 55.1 | 95.0 | 61.2 | 155.3 | 77.2 | 190.3 |
| @60k | 51.0 | 85.3 | | | | |
| @128k | n/a | n/a | 48.2 | 103.0 | 70.9 | 173.9 |
| @256k | n/a | n/a | 35.9 | 69.1 | 63.7 | 120.3 |
| @512k | n/a | n/a | n/a | n/a | 53.1 | 88.0 |
| 32 streams | 628 | 445 | 929 | 1,005 | 1,229 | 1,423 |

Prefill t/s (same with or without DSpark): TP=1 4.2k @32k, 3.9k @60k; TP=2 6.5k / 5.3k / 4.1k @32k / 128k / 256k;
TP=4 8.1k / 7.2k / 6.0k / 4.6k @32k / 128k / 256k / 512k. TP=1 is configured for 64k.

Exact defaults on one card (bf16 dense, head and layers 20-23): plain 54.8 short, 47.8 @30k; DSpark 83.4 short,
94.1 @30k. DSpark loses on one card at 32 streams (KV pool nearly full) and on short chat with reasoning off. DSpark
t/s depends on the prompt (two 32k fillers on one card: 79.9 and 95.0).

Quality: top-1 agreement with a transformers BF16 reference 0.994-0.995 for the fast set (exact defaults: 0.996-0.998; 4,421
prompt tokens). Needle (passcode at 10/50/90% depth) found 15/15 at 32k-512k; int8 and BF16 dense give the same
answers at 32k and 128k. Greedy output is bit-identical across runs, with and without DSpark.

## Known limits

- KV pages must be 64 tokens (vLLM's XPU default); other sizes fall back to a slower Triton path.
- TP=2 + KV offload + DSpark is stable at `--max-num-seqs 16`; at 32 concurrent 8k requests one rank runs out of
  memory. TP=1 and TP=4 pass 32.
- Images work (encoder replicated per rank); `--limit-mm-per-prompt '{"image":0}'` frees its memory on text-only servers.
- TP=8 is not blocked but untested. No expert offload to CPU. No FP8 / INT8 KV cache.

## Credits

- Model code, DSpark draft and speculator: the upstream vLLM Inkling 2 implementation (NVIDIA), ported here.
- [mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu): an independent single-B70 port, used
  as a reference and benchmark only (no code taken); its decode prompts are used above.
