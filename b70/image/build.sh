#!/bin/bash
# Build the b70 vLLM images (../README-inkling2.md).
#
#   build.sh env  <vllm-src.tar>   the environment image (everything but vLLM), ENV_TAG; the tarball gives requirements/
#   build.sh env-update <vllm-src.tar>   ENV_TAG = PREV_ENV with only the kernel libraries rebuilt (Dockerfile env-update)
#   build.sh vllm <vllm-src.tar>   the engine image FROM ENV_TAG, tagged TAG (default vllm-xpu:b70-skv-<10-char sha>)
#
# <vllm-src.tar> is `git archive <commit>` of a vLLM tree (its commit id is read back with git get-tar-commit-id).
# Run it as a user unit so a dropped ssh does not matter:  systemd-run --user --unit=b70-build bash build.sh env x.tar
# Knobs: KIT (Lumnus kit checkout), ENV_TAG, TAG, MAX_JOBS (kernel build), FLOOR_GIB (stop the build below this
# MemAvailable), SHIM_SRC (the L0 residency shim source).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:?env or vllm}; tar=$(realpath "${2:?vllm source tarball}")
KIT="${KIT:?set KIT to a clean checkout of the Lumnus b70-flash-next kit at v0.32.0-b70.7}"; KIT_TAG=v0.32.0-b70.7
ENV_TAG="${ENV_TAG:-vllm-xpu-b70env:0.1.16.2-w4a8}"; PREV_ENV="${PREV_ENV:-vllm-xpu-b70env:0.1.16.2}"; MAX_JOBS="${MAX_JOBS:-6}"; FLOOR_GIB="${FLOOR_GIB:-12}"
SHIM_SRC="${SHIM_SRC:-$here/../l0_owner_residency.c}"
exec 9>"$here/.build.lock"; flock -n 9 || { echo "another build.sh holds $here/.build.lock" >&2; exit 1; }

commit=$(git get-tar-commit-id < "$tar") || { echo "$tar carries no commit id (use git archive <commit>)" >&2; exit 1; }
short=${commit:0:10}
log="$here/logs/$mode-$short-$(date +%Y%m%d-%H%M%S).log"; mkdir -p "$here/logs"
exec > >(tee -a "$log") 2>&1
echo "$(date -Is) build.sh $mode, vLLM source $commit, log $log"

# The build context: this directory's files, the shim, and the kit's patch directories the Dockerfile names.
[ "$(git -C "$KIT" describe --tags --exact-match HEAD)" = "$KIT_TAG" ] && [ -z "$(git -C "$KIT" status --porcelain)" ] ||
  { echo "kit $KIT is not a clean checkout of $KIT_TAG" >&2; exit 1; }
ctx="$here/ctx"; rm -rf "$ctx"; mkdir -p "$ctx/kit" "$ctx/env-req"
cp "$here/Dockerfile" "$here/overrides.txt" "$here/build-natives.sh" "$SHIM_SRC" "$ctx/"
for d in vllm-xpu-kernels vllm-xpu-kernels-py libgdn-index64 sycl-tla; do cp -a "$KIT/v0.32/engine/patches/$d" "$ctx/kit/"; done
src="$here/src-$short"; rm -rf "$src"; mkdir -p "$src"; tar -xf "$tar" -C "$src"

if [ "$mode" = env ]; then
  cp -a "$src/requirements" "$ctx/env-req/"
  args=(--target env --build-arg MAX_JOBS="$MAX_JOBS" --label b70.env-requirements="$commit" -t "$ENV_TAG")
  tag=$ENV_TAG
elif [ "$mode" = env-update ]; then
  docker image inspect "$PREV_ENV" >/dev/null
  args=(--target env-update --build-arg MAX_JOBS="$MAX_JOBS" --build-arg KBASE="$PREV_ENV" --build-arg PREV_ENV="$PREV_ENV" -t "$ENV_TAG")
  tag=$ENV_TAG
elif [ "$mode" = vllm ]; then
  tag="${TAG:-vllm-xpu:b70-skv-$short}"
  docker image inspect "$ENV_TAG" >/dev/null
  args=(--target vllm --build-arg ENV_IMAGE="$ENV_TAG" --build-arg VLLM_COMMIT="$commit"
        --build-arg VLLM_VERSION="${VLLM_VERSION:-0.32.0+g$short}" --build-context vllm-src="$src" -t "$tag")
else
  echo "mode must be env, env-update or vllm" >&2; exit 2
fi
if docker image inspect "$tag" >/dev/null 2>&1; then echo "$tag exists; never retag: pick another TAG" >&2; exit 1; fi

# Host-RAM gate: record the low point; stop the build below the floor (the kernel build is the memory peak).
avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
echo "MemAvailable at start: $(avail_mib) MiB, MAX_JOBS=$MAX_JOBS"
t0=$(date +%s)
docker build --progress=plain "${args[@]}" -f "$ctx/Dockerfile" "$ctx" & pid=$!
low=$(avail_mib)
while kill -0 $pid 2>/dev/null; do
  a=$(avail_mib); [ "$a" -lt "$low" ] && low=$a
  if [ "$a" -lt $((FLOOR_GIB * 1024)) ]; then
    echo "$(date -Is) ABORT: MemAvailable $a MiB below ${FLOOR_GIB} GiB, cancelling the build"; kill -INT $pid
  fi
  sleep 5
done
rc=0; wait $pid || rc=$?
rm -rf "$src" "$ctx"
echo "$(date -Is) build.sh $mode exit $rc after $(( $(date +%s) - t0 )) s; MemAvailable low point $low MiB; image $tag"
exit $rc
