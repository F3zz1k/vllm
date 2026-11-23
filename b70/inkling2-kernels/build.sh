#!/bin/bash
# Build the Inkling2 kernel library (_inkling2_C + libinkling2_{attn,moe}_kernels_xe_2.so) standalone, with the
# toolchain the b70 env image's kernels were built with. The vendor kernel libraries in the image are not touched.
#
#   KIT=<Lumnus b70-flash-next checkout @ v0.32.0-b70.7> OUT=<output dir> bash build.sh
#
# Knobs: ENV_IMAGE (default vllm-xpu-b70env:0.1.16.2-w4a8), TOOLCHAIN (image tag, default inkling2-kbuild),
# WORK (source and build dir, default ./work), MAX_JOBS (default 6), DOCKER_MEM (e.g. 24g: hard RAM cap for the
# compile container; each SYCL compile job can take several GiB).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
KIT="${KIT:?set KIT to a clean checkout of the Lumnus b70-flash-next kit at v0.32.0-b70.7}"
ENV_IMAGE="${ENV_IMAGE:-vllm-xpu-b70env:0.1.16.2-w4a8}"; TOOLCHAIN="${TOOLCHAIN:-inkling2-kbuild}"
WORK="$(realpath -m "${WORK:-$here/work}")"; OUT="$(realpath -m "${OUT:?set OUT}")"; MAX_JOBS="${MAX_JOBS:-6}"
mkdir -p "$WORK" "$OUT"

# 1. toolchain image: the env image + oneAPI DPC++ 2026.1.1 + sycl-tla 87f6850 with the kit's patch 0001
if ! docker image inspect "$TOOLCHAIN" >/dev/null 2>&1; then
  ctx="$WORK/toolchain-ctx"; rm -rf "$ctx"; mkdir -p "$ctx"
  cp "$here/Dockerfile.toolchain" "$ctx/Dockerfile"
  cp "$KIT"/v0.32/engine/patches/sycl-tla/0001-*.patch "$ctx/tla-0001.patch"
  docker build --build-arg ENV_IMAGE="$ENV_IMAGE" -t "$TOOLCHAIN" "$ctx"
fi

# 2. source: vllm-xpu-kernels 6b30c9b + kit patches 0001-0007 (the env's kernel source), plus this directory
src="$WORK/vllm-xpu-kernels"
if [ ! -d "$src/.git" ]; then
  C=6b30c9b92cc525c32729df988b5f918e7c5661a5
  git init -q "$src"
  git -C "$src" fetch -q --depth 1 https://github.com/vllm-project/vllm-xpu-kernels.git $C
  git -C "$src" checkout -q FETCH_HEAD
  git -C "$src" -c user.name=b70 -c user.email=b70@localhost am -q "$KIT"/v0.32/engine/patches/vllm-xpu-kernels/000[1-7]-*.patch
  patch -s -d "$src" -p1 < "$here/top-level-cmake.patch"
  cat "$here/add-inkling2.cmake" >> "$src/CMakeLists.txt"
fi
rm -rf "$src/csrc/xpu/inkling2"; cp -a "$here/src/csrc/xpu/inkling2" "$src/csrc/xpu/"

# 3. build only _inkling2_C and its two kernel libraries
mkdir -p "$WORK/build"
docker run --rm ${DOCKER_MEM:+--memory=$DOCKER_MEM --memory-swap=$DOCKER_MEM} -v "$src":/src/vxk -v "$WORK/build":/build -w /src/vxk -e MAX_JOBS="$MAX_JOBS" \
  -e VLLM_CUTLASS_SRC_DIR=/src/tla --entrypoint bash "$TOOLCHAIN" -c '
  set -e; . /opt/intel/oneapi/setvars.sh --force >/dev/null
  if [ ! -f /build/build.ninja ]; then
    cmake -S /src/vxk -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=xpu \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python \
      -DBASIC_KERNELS_ENABLED=OFF -DFA2_KERNELS_ENABLED=OFF -DMOE_KERNELS_ENABLED=OFF \
      -DGDN_KERNELS_ENABLED=OFF -DMQA_LOGITS_KERNELS_ENABLED=OFF -DMHC_KERNELS_ENABLED=OFF \
      -DXPU_SPECIFIC_KERNELS_ENABLED=OFF -DXPUMEM_ALLOCATOR_ENABLED=OFF -DVLLM_XPU_ENABLE_XE3P=OFF \
      -DVLLM_XPU_ENABLE_ONEDNN=OFF -DINKLING2_ATTN_ENABLED=ON
  fi
  cmake --build /build --target _inkling2_C -j "$MAX_JOBS"'
cp "$WORK"/build/_inkling2_C*.so "$WORK"/build/libinkling2_attn_kernels_xe_2.so \
   "$WORK"/build/libinkling2_moe_kernels_xe_2.so "$OUT/"
ls -la "$OUT"
