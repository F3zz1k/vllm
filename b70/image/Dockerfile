# syntax=docker/dockerfile:1
# A vLLM image for Intel Arc Pro B70: the Lumnus 0.32.0-b70.5 environment plus three b70.7 kernel patches
# (Lumnus b70-flash-next kit @ v0.32.0-b70.7; the b70.5 parts are unchanged there), v0.32/README.md "Software" and "Build and run" steps 2-6, PROVENANCE.md
# sections 8-9) in these container conventions (ubuntu 24.04, compute-runtime and Level Zero in the image, uv venv at
# /opt/venv). Built by build.sh in this directory; README.md says how and what each stage holds.
#
#   base     ubuntu + compute-runtime 26.35.39758.10 / IGC 2.41.5 + Level Zero 1.32 + the Python environment
#            (torch 2.14.0+xpu, the fork's requirements/xpu.txt, PyPI vllm-xpu-kernels 0.1.16.2, pins in overrides.txt)
#   builder  base + oneAPI DPC++ 2026.1.1; builds the nine kernel libraries (W4A8 included), wu1ff's natives, libgdn_index64.so,
#            libb70_moe_decode.so, the oneCCL legacy plugin with oneCCL#229 and the L0 residency shim. Never shipped.
#   env      base + the built artefacts + the kernels-py patches: everything except vLLM (vllm-xpu-b70env:0.1.16.2-w4a8)
#   vllm     ENV_IMAGE + a vLLM source tree from the build context `vllm-src`, installed editable (vllm-xpu:b70-skv-<sha>)
ARG ENV_IMAGE=vllm-xpu-b70env:0.1.16.2-w4a8
# The image the kernel toolchain stage starts from: `base` for a build from scratch; build.sh env-update passes the
# previous env image, so an update rebuilds only the kernel libraries.
ARG KBASE=base
# env-update: the env image the new kernel libraries are laid over (everything else keeps its bytes)
ARG PREV_ENV=vllm-xpu-b70env:0.1.16.2

FROM ubuntu:24.04 AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
      build-essential curl git wget ca-certificates numactl libaio-dev ocl-icd-libopencl1 \
      python3.12 python3.12-dev python3-pip clinfo && rm -rf /var/lib/apt/lists/*
RUN mkdir /neo && cd /neo && \
    B=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10 && \
    G=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5 && \
    L=https://github.com/oneapi-src/level-zero/releases/download/v1.32.0 && \
    wget -q $G/intel-igc-core-2_2.41.5+22716_amd64.deb $G/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
      $B/intel-ocloc_26.35.39758.10-0_amd64.deb $B/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
      $B/libze-intel-gpu1_26.35.39758.10-0_amd64.deb $B/libigdgmm12_22.10.0_amd64.deb $B/ww35.sum \
      $L/libze1_1.32.0+u24.04_amd64.deb $L/libze-dev_1.32.0+u24.04_amd64.deb && \
    grep -v dbgsym ww35.sum | grep -E "ocloc_|opencl-icd_|libze-intel-gpu1_|libigdgmm12_" | sha256sum -c && \
    printf '%s\n' \
      "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  intel-igc-core-2_2.41.5+22716_amd64.deb" \
      "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  intel-igc-opencl-2_2.41.5+22716_amd64.deb" | sha256sum -c && \
    dpkg -i *.deb && cd / && rm -rf /neo
ENV PATH="/root/.local/bin:/opt/venv/bin:$PATH" VIRTUAL_ENV=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_HTTP_TIMEOUT=500 \
    LD_LIBRARY_PATH=/opt/venv/lib VLLM_TARGET_DEVICE=xpu VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 --seed /opt/venv
# Step 2 of the kit, without vLLM itself: the fork's requirements (requirements/ of the env's reference commit, staged by
# build.sh) plus the kit's pins. overrides.txt keeps those pins when requirements/xpu.txt names other versions (it pins
# vllm_xpu_kernels 0.1.15.4) and again when the vllm stage installs the engine. Step 3, first half: the PyPI wheel of
# vllm-xpu-kernels 0.1.16.2 (Python files and the attention libraries), checked against its sha256.
COPY env-req/ /opt/b70/env-req/
COPY overrides.txt /opt/b70/overrides.txt
RUN mkdir /tmp/kw && python -m pip download -q --no-deps vllm-xpu-kernels==0.1.16.2 -d /tmp/kw && \
    echo "86bfa26abbb605fab485236a386e6de07f76320177c456fa96e55c27502af17a  /tmp/kw/vllm_xpu_kernels-0.1.16.2-cp38-abi3-manylinux_2_28_x86_64.whl" | sha256sum -c && \
    uv pip install --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
      --overrides /opt/b70/overrides.txt -r /opt/b70/env-req/requirements/xpu.txt \
      torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0 \
      /tmp/kw/vllm_xpu_kernels-0.1.16.2-cp38-abi3-manylinux_2_28_x86_64.whl && \
    rm -rf /tmp/kw /root/.cache && \
    python - <<'EOF'
import importlib.metadata as m, triton
want = {"torch": "2.14.0+xpu", "vllm-xpu-kernels": "0.1.16.2", "oneccl": "2022.1.2", "oneccl-devel": "2022.1.2",
        "auto-round-lib": "0.16.0", "transformers": "5.17.0", "intel-sycl-rt": "2026.1.1", "dpcpp-cpp-rt": "2026.1.1",
        "intel-opencl-rt": "2026.1.1", "onemkl-sycl-blas": "2026.1.0", "triton": "3.8.0"}
got = {k: m.version(k) for k in want if k != "triton"} | {"triton": triton.__version__.split("+")[0]}
print(got)
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
assert not bad, f"version mismatch (got, want): {bad}"
EOF
# triton's XPU launcher compiles against CL/cl.h at run time; intel-sycl-rt ships only include/sycl (kit step 2 check)
RUN apt-get update -y && apt-get install -y --no-install-recommends opencl-c-headers && rm -rf /var/lib/apt/lists/* && \
    printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I/opt/venv/include -x c++ -fsyntax-only -

FROM ${KBASE} AS kernels
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ARG MAX_JOBS=6
# oneAPI DPC++ 2026.1.1 (the host keeps its 2026.0) and the kernels' build requirements (pyproject: setuptools < 80).
# None of this reaches the env stage.
RUN apt-get update -y && apt-get install -y --no-install-recommends gpg && \
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update -y && apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp-2026.1=2026.1.1-325 intel-oneapi-dpcpp-cpp-2026.1=2026.1.1-325 && \
    rm -rf /var/lib/apt/lists/* && \
    uv pip install "cmake>=3.26" ninja "packaging>=24.2" "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8" wheel regex jinja2 psutil && \
    . /opt/intel/oneapi/setvars.sh --force >/dev/null && icpx --version | grep 'Compiler 2026.1.1 '
ENV K=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels T=/opt/venv/lib/python3.12/site-packages/torch
# Step 3, second half: the compiled libraries from upstream 6b30c9b + kit patches 0001-0007 and 0014 (v0.32.0-b70.7 kit;
# 0001-0005 are the b70.5 set byte for byte, 0006 the W4A8 grouped GEMM, 0007 the 64-bit conv-state offsets, 0014 the fast
# e4m3 -> half KV reorder), FA2 / MQA-logits / Xe3P off, oneDNN on. 0014 sits after 0008-0013 in the kit, so it goes in
# with `patch` (one hunk lands 16 lines up with fuzz 1) and the two files must be the bytes of the cherry-pick. sycl-tla is
# the revision the kernels pin (87f6850) plus the kit's sycl-tla 0001 (the opt-in reorder 0014 switches on; its 0002 is
# unit tests only). 0014 only touches attn_kernels_xe_2, which FA2 off does not build: inert here until FA2 is built.
# Only the compiled libraries are taken (the wheel built here has no attention libraries), plus the W4A8 library and its
# Python module; they are laid over the PyPI files here too, because libgdn_index64.so links the rebuilt
# libgdn_attn_kernels_xe_2.so.
RUN --mount=type=bind,source=kit/vllm-xpu-kernels,target=/kit/patches --mount=type=bind,source=kit/sycl-tla,target=/kit/tla \
    S=87f6850680a580654b9ea2c80dbc01aeb36ad231 && git init -q /src/tla && cd /src/tla && \
    git fetch -q --depth 1 https://github.com/intel/sycl-tla.git $S && git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = $S && git -c user.name=b70 -c user.email=b70@localhost am -q /kit/tla/0001-*.patch && \
    C=6b30c9b92cc525c32729df988b5f918e7c5661a5 && git init -q /src/vxk && cd /src/vxk && \
    git fetch -q --depth 1 https://github.com/vllm-project/vllm-xpu-kernels.git $C && git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = $C && git -c user.name=b70 -c user.email=b70@localhost am -q /kit/patches/000[1-7]-*.patch && \
    patch -s -p1 --forward --no-backup-if-mismatch -F2 < /kit/patches/0014-*.patch && \
    printf '%s\n' \
      "84aeef869d3e814a8a5128443214cc104f6a78305f331be92e399a5740262ab2  csrc/xpu/attn/xe_2/CMakeLists.txt" \
      "52125a4a5274c294de35fa91d8d922942bb172ff9558232e47853872bae162bb  csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp" | sha256sum -c && \
    . /opt/intel/oneapi/setvars.sh --force >/dev/null && \
    VLLM_CUTLASS_SRC_DIR=/src/tla VLLM_XPU_ENABLE_XE3P=OFF FA2_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF VLLM_XPU_ENABLE_ONEDNN=ON \
      MAX_JOBS=$MAX_JOBS VLLM_VERSION_OVERRIDE=0.1.16.2+b70 python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 && \
    mkdir w /out /out/kernels && python -I -m zipfile -e dist/*.whl w && \
    for f in _C.abi3.so _moe_C.abi3.so _xpu_C.abi3.so xpumem_allocator.abi3.so libgdn_attn_kernels_xe_2.so \
             libgrouped_gemm_xe_2.so libgrouped_gemm_xe_default.so libmhc_kernels_xe_2.so libgrouped_gemm_w4a8_xe_2.so; do \
      install -m 0755 w/vllm_xpu_kernels/$f /out/kernels/$f && install -m 0755 w/vllm_xpu_kernels/$f $K/$f; done && \
    install -m 0644 w/vllm_xpu_kernels/b70_moe_w4a8.py /out/kernels/b70_moe_w4a8.py && \
    cd / && rm -rf /src/vxk /src/tla /root/.cache

FROM kernels AS builder
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
# Step 5.1 + 5.2: wu1ff's ten native libraries and libgdn_index64.so, each with the pack's own build script (unchanged).
RUN --mount=type=bind,source=kit/libgdn-index64,target=/kit/gdn --mount=type=bind,source=build-natives.sh,target=/build-natives.sh \
    C=5e5cf43f048d9b32146bb22925d4a2cf198400b0 && git init -q /src/wu1ff && cd /src/wu1ff && \
    git fetch -q --depth 1 https://github.com/wu1ff/B70-LLM-Controller.git $C && git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = $C && bash /build-natives.sh /src/wu1ff/model-packs/Qwen3.8-Flash-Next/patches/production /kit/gdn /out/kernels && \
    cd / && rm -rf /src/wu1ff /build
# Step 5.3: Niluthon's hot/cold decode op (the Flash-Next 2-card preset's B70_MOE_DECODE_OP).
RUN C=e536e3378f8cfb165c90a927d93163ea69618a64 && git init -q /src/nil && cd /src/nil && \
    git fetch -q --depth 1 https://github.com/Niluthon/qwen38-flash-tp2.git $C && git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = $C && . /opt/intel/oneapi/setvars.sh --force >/dev/null && mkdir -p /out/b70 && \
    icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -O3 -std=c++20 -shared -fPIC -ffast-math \
      -D_GLIBCXX_USE_CXX11_ABI=1 -I$T/include -I$T/include/torch/csrc/api/include -x c++ kernels/b70-moe-decode/b70_moe_decode.sycl \
      -o /out/b70/libb70_moe_decode.so -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib && \
    cd / && rm -rf /src/nil
# Step 6: the oneCCL legacy plugin with uxlfoundation/oneCCL#229 (head caeb329 on master-v2 c5108e7), against the venv's
# oneCCL 2022.1.2. CCL_PLUGIN points the shipped loader at it.
RUN C=caeb329166bb048b564a5dd8bd4e3d88c1ecdab3 && git init -q /src/ccl && cd /src/ccl && \
    git fetch -q --depth 1 https://github.com/uxlfoundation/oneCCL.git $C && git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = $C && . /opt/intel/oneapi/setvars.sh --force >/dev/null && \
    mkdir -p vinc /out/b70/oneccl-legacy && ln -sfn /opt/venv/include/oneapi vinc/oneapi && \
    icpx -w -fsycl -O2 -fPIC -shared -std=c++17 -DNDEBUG -Ivinc -Isrc -Iinclude \
      plugins/legacy/ccl_legacy.cpp src/debug.cpp -o /out/b70/oneccl-legacy/libccl_legacy.so \
      -L/opt/venv/lib -lccl -l:libccl.so.2 -ldl -Wl,--no-undefined -Wl,-soname,libccl_legacy.so && \
    cd / && rm -rf /src/ccl
# The Level Zero residency shim, with the build line in its header (replaces wu1ff's peer-residency shim; needs
# ZE_ENABLE_TRACING_LAYER=1 at run time).
RUN --mount=type=bind,source=l0_owner_residency.c,target=/src/l0_owner_residency.c \
    gcc -O2 -shared -fPIC -Wall /src/l0_owner_residency.c -o /out/b70/libl0_owner_residency.so -lze_loader

FROM base AS env
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ENV K=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
COPY --from=builder /out/kernels/ /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/
COPY --from=builder /out/b70/ /opt/b70/
# Step 4: the astra Python modules and hooks, written against the 0.1.16.2 Python files; the four results must be the
# validated bytes (fused_moe_interface.py 3ef25b12...). Then every shipped library is loaded once (catches a missing
# symbol or library at build time, not at the first request), and the manifest the vllm stage re-checks is written.
RUN --mount=type=bind,source=kit/vllm-xpu-kernels-py,target=/kit/py \
    cd /opt/venv/lib/python3.12/site-packages && for p in /kit/py/*.patch; do git apply -p1 < "$p"; done && \
    printf '%s\n' \
      "3ef25b12fd0e40dd17094eff2658a8bfdfa31217d45143a1273064ce2a597da7  vllm_xpu_kernels/fused_moe_interface.py" \
      "9346e3d41d2d7b890edf441624359db7d8d77214b19d776878bd0e5a1dbc3806  vllm_xpu_kernels/astra500_moe.py" \
      "6650c391e6defcf0433ace77ba7fb63ba69c5e88e43fb5d5f29dcc79318e7f9e  vllm_xpu_kernels/astra_structural.py" \
      "ed31326086a5d09717f6cb408ddd1e5769aea96755e149537611e5e659094bd6  vllm_xpu_kernels/astra_ownership.py" | sha256sum -c && \
    python - <<'EOF'
import ctypes, glob, torch
libs = sorted(glob.glob("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/*.so")) + ["/opt/b70/libb70_moe_decode.so"]
for lib in libs:
    torch.ops.load_library(lib)
ctypes.CDLL("/opt/b70/oneccl-legacy/libccl_legacy.so")
torch.ops._xpu_C.cutlass_grouped_gemm_w4a8  # the W4A8 grouped GEMM (kit patch 0006) is registered
import vllm_xpu_kernels.b70_moe_w4a8 as w4a8
assert w4a8.load_op() is torch.ops._xpu_C.cutlass_grouped_gemm_w4a8
print(f"loaded {len(libs)} torch op libraries, the W4A8 op and the oneCCL plugin")
EOF
RUN find $K /opt/b70 -type f \( -name '*.so' -o -name '*.py' \) | sort | xargs sha256sum > /opt/b70/MANIFEST.sha256 && \
    uv pip freeze > /opt/b70/env-freeze.txt && wc -l /opt/b70/MANIFEST.sha256 /opt/b70/env-freeze.txt
LABEL b70.env="0.1.16.2-w4a8" b70.kit="v0.32.0-b70.7" b70.torch="2.14.0+xpu" b70.compute-runtime="26.35.39758.10" \
      b70.vllm-xpu-kernels="0.1.16.2 + 6b30c9b and kit patches 0001-0007, 0014" b70.sycl-tla="87f6850 + kit 0001" b70.wu1ff="5e5cf43" b70.oneccl-plugin="caeb329"

# The update path (build.sh env-update): PREV_ENV with the kernel libraries of the `kernels` stage laid over its own, the
# W4A8 library and module added. Python files, kernels-py patches, natives, plugins and packages keep PREV_ENV's bytes
# (the freeze must not change); the manifest is rewritten so the vllm stage checks the new set.
FROM ${PREV_ENV} AS env-update
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ARG PREV_ENV
ENV K=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
COPY --from=kernels /out/kernels/ /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/
RUN echo "3ef25b12fd0e40dd17094eff2658a8bfdfa31217d45143a1273064ce2a597da7  $K/fused_moe_interface.py" | sha256sum -c && \
    python - <<'EOF'
import ctypes, glob, torch
libs = sorted(glob.glob("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/*.so")) + ["/opt/b70/libb70_moe_decode.so"]
for lib in libs:
    torch.ops.load_library(lib)
ctypes.CDLL("/opt/b70/oneccl-legacy/libccl_legacy.so")
torch.ops._xpu_C.cutlass_grouped_gemm_w4a8  # the W4A8 grouped GEMM (kit patch 0006) is registered
import vllm_xpu_kernels.b70_moe_w4a8 as w4a8
assert w4a8.load_op() is torch.ops._xpu_C.cutlass_grouped_gemm_w4a8
print(f"loaded {len(libs)} torch op libraries, the W4A8 op and the oneCCL plugin")
EOF
RUN uv pip freeze | cmp - /opt/b70/env-freeze.txt && \
    find $K /opt/b70 -type f \( -name '*.so' -o -name '*.py' \) | sort | xargs sha256sum > /opt/b70/MANIFEST.sha256 && \
    wc -l /opt/b70/MANIFEST.sha256
LABEL b70.env-base="${PREV_ENV}" b70.env="0.1.16.2-w4a8" b70.kit="v0.32.0-b70.7" b70.torch="2.14.0+xpu" b70.compute-runtime="26.35.39758.10" \
      b70.vllm-xpu-kernels="0.1.16.2 + 6b30c9b and kit patches 0001-0007, 0014" b70.sycl-tla="87f6850 + kit 0001" b70.wu1ff="5e5cf43" b70.oneccl-plugin="caeb329"

FROM ${ENV_IMAGE} AS vllm
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ARG VLLM_COMMIT
ARG VLLM_VERSION
# Kit step 2, the engine half. The environment must come out of it unchanged: the manifest of the kernel and native
# libraries holds, and no package of the env stage changed version (new requirements of the source tree may be added).
COPY --from=vllm-src . /workspace/vllm
WORKDIR /workspace/vllm
RUN test -n "$VLLM_COMMIT" && test -n "$VLLM_VERSION" && \
    VLLM_VERSION_OVERRIDE="$VLLM_VERSION" uv pip install --no-build-isolation --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
      --overrides /opt/b70/overrides.txt -e . && rm -rf /root/.cache && \
    sha256sum -c --quiet /opt/b70/MANIFEST.sha256 && \
    changed=$(LC_ALL=C comm -23 <(LC_ALL=C sort /opt/b70/env-freeze.txt) <(uv pip freeze | LC_ALL=C sort)) && \
    { [ -z "$changed" ] || { echo "the engine install changed the environment: $changed" >&2; exit 1; }; } && \
    python -c "import vllm, torch, vllm_xpu_kernels; print('vllm', vllm.__version__, 'torch', torch.__version__)"
# The editable install leaves the source tree without bytecode, so every process (API server, engine core, each
# worker) compiled ~2,700 modules at import and threw the result away with the container: ~4 s per process.
RUN python -m compileall -q -j 8 /workspace/vllm/vllm > /dev/null
# The KV offload copy list (its own Level Zero copy list in the engine's context; the source lives with its Python
# wrapper in the tree; VLLM_KV_OFFLOAD_COPY_LIST=1 uses it, preloaded with ZE_ENABLE_TRACING_LAYER=1).
RUN gcc -O2 -shared -fPIC -Wall -Werror vllm/v1/kv_offload/cpu/l0copy.c -o /opt/b70/libl0copy.so -lze_loader
LABEL vllm.commit="$VLLM_COMMIT" org.opencontainers.image.revision="$VLLM_COMMIT"
