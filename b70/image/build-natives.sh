#!/bin/bash
# Builder-stage helper (Dockerfile, kit "Build and run" steps 5.1 and 5.2): wu1ff's ten native libraries and
# libgdn_index64.so, each compiled by the pack's own build script, unchanged, in the /build layout its
# build/Dockerfile.native gives it (sources in /build, headers in /build/source, sycl-tla in /build/tla, the compiler
# under /toolchain/oneapi). Installed names follow that Dockerfile's export stage (libastra_hc_v3.so -> libastra_hc.so).
#   build-natives.sh PACK_DIR GDN_PATCH_DIR OUT_DIR
set -euo pipefail
pack=$1 gdn_patches=$2 out=$3
cd "$pack"
echo "5938a31e3940d1c90e88d9126788990a25a5e0ac968e53345461459b5ea21218  inputs/sycl-tla.tar.gz" | sha256sum -c
mkdir -p /opt/tla "$out" /toolchain/oneapi/opt/intel
tar -xzf inputs/sycl-tla.tar.gz -C /opt/tla --strip-components=1
ln -sfn /opt/intel/oneapi /toolchain/oneapi/opt/intel/oneapi
# the kit's libgdn-index64 patch (the null-block skip, vllm-xpu-kernels#660), with `patch -p1` as the kit says: one hunk
# needs fuzz 2 because wu1ff's copy already casts that offset to 64 bits. The patched files must be the reviewed bytes.
for p in "$gdn_patches"/*.patch; do patch -s -p1 --forward --no-backup-if-mismatch -d dependencies/gdn-headers < "$p"; done
(cd dependencies/gdn-headers/csrc/xpu/gdn_attn && sha256sum -c --quiet) <<'EOF'
551135840c079ba2f792c5c765464ae9df6c3656639b560d8a3c5b82acd4a6a2  causal_conv1d.hpp
c4af2bb25c5b1c27e8c0349b100732517f6dc421d4962105883446d34712d8dd  gated_delta_rule.hpp
4e2bd259b7537c7025c88c29325c79596aea498b09244edac2a801e4aa217b79  gdn_attn_interface.cpp
e85199c8552f3537d17583bd9c9134136c518f68db3220b08d8f4284d80ea47d  xe_2/chunk_causal_conv1d_tiled_xe2.hpp
d0ec43e1ddf3ac631c38f3a8559866f79cd05f3658addc0be104350a43133e9a  xe_2/chunk_causal_conv1d_xe2.hpp
EOF

build() {  # headers source-dir script output installed-name
  rm -rf /build && mkdir -p /build/source
  cp -a "dependencies/$1/." /build/source/
  ln -s /opt/tla /build/tla
  cp -a "native/$2/." /build/
  bash "/build/$3"
  install -m 0755 "/build/$4" "$out/$5"
}
build common-native-headers astra500-native            build.sh    libastra_moe.so          libastra_moe.so
build common-native-headers astra500-structural-hc     build-v3.sh libastra_hc_v3.so        libastra_hc.so
build common-native-headers astra500-structural-moe    build.sh    libastra_direct.so       libastra_direct.so
build common-native-headers astra500-ownership-groups  build.sh    libastra_groups.so       libastra_groups.so
build common-native-headers astra500-ownership-batch   build.sh    libastra_batch.so        libastra_batch.so
build common-native-headers astra500-beyond-prefill    build.sh    libbeyond_prefill.so     libbeyond_prefill.so
build common-native-headers astra500-dominant-native   build.sh    libdominant_w4_layout.so libdominant_w4_layout.so
build common-native-headers prefill-native             build.sh    libserving_prefill.so    libserving_prefill.so
build common-native-headers long-hc-selected           build.sh    liblong_hc.so            liblong_hc.so
build common-native-headers long-prefill-native        build.sh    liblong_prefill_v3.so    liblong_prefill_v3.so
# links the rebuilt libgdn_attn_kernels_xe_2.so, already laid over the venv's copy by the previous builder step
build gdn-headers           native                     build.sh    libgdn_index64.so        libgdn_index64.so
