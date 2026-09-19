#!/usr/bin/env bash
# Fork-only provisioning for the unchanged upstream build/test commands.
set -euo pipefail

profile=${1:?provisioning profile required}
task_root=/data/action/osd
task_downloads=${RUNNER_TEMP:-/tmp}/ncnn-fork-downloads
mkdir -p "$task_root" "$task_downloads"

download()
{
    local url=$1 archive=$2 checksum=${3:-}
    curl --fail --location --retry 3 --output "$task_downloads/$archive" "$url"
    if [ -n "$checksum" ]; then
        printf '%s  %s\n' "$checksum" "$task_downloads/$archive" | sha256sum --check -
    fi
    sha256sum "$task_downloads/$archive"
}

extract()
{
    local url=$1 destination=$2 checksum=${3:-}
    local archive=${url##*/}
    download "$url" "$archive" "$checksum"
    mkdir -p "$destination"
    tar -xf "$task_downloads/$archive" --strip-components=1 -C "$destination"
}

checkout()
{
    local url=$1 revision=$2 directory=$3
    git init "$directory"
    git -C "$directory" fetch --depth 1 "$url" "$revision"
    git -C "$directory" checkout --detach FETCH_HEAD
    test "$(git -C "$directory" rev-parse HEAD)" = "$revision"
}

install_qemu()
{
    checkout https://github.com/qemu/qemu.git 72b88908d12ee9347d13539c7dd9a252625158d1 "$task_downloads/qemu"
    (cd "$task_downloads/qemu"
     ./configure --prefix="$task_root/qemu-install" --target-list=riscv64-linux-user --disable-system
     make -j 2
     make install)
    "$task_root/qemu-install/bin/qemu-riscv64" --version
}

install_rvv()
{
    extract https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.05.19/riscv64-glibc-ubuntu-22.04-llvm.tar.xz \
        "$task_root/riscv" 57b7690ad66a36e08a3b4586099a6454b529e9681f0a7199c05bff15dc88b1b6
    "$task_root/riscv/bin/riscv64-unknown-linux-gnu-gcc" --version
    install_qemu
}

require_cpu()
{
    for feature in "$@"; do
        if ! grep -qw "$feature" /proc/cpuinfo; then
            echo "::error::Runner lacks $feature required by this native test; no fallback or pass claimed."
            exit 1
        fi
    done
}

case "$profile" in
    swiftshader)
        apt-get install -y clang libvulkan-dev
        ;;
    avx512)
        require_cpu avx512f avx512_vnni
        ;;
    harmonyos)
        archive=ohos-sdk-windows_linux-public.tar.gz
        download "https://repo.huaweicloud.com/harmonyos/os/5.0.3-Release/$archive" "$archive"
        native=$(tar -tf "$task_downloads/$archive" | grep -E '(^|/)native-linux-x64-5\.0\.3\.135-Release\.zip$')
        test -n "$native"
        tar -xf "$task_downloads/$archive" -C "$task_downloads" "$native"
        mkdir -p "$task_root/ohos-sdk/linux"
        unzip -q "$task_downloads/$native" -d "$task_root/ohos-sdk/linux"
        "$task_root/ohos-sdk/linux/native/build-tools/cmake/bin/cmake" --version
        ;;
    loongarch)
        apt-get install -y qemu-user-static
        extract https://github.com/sunhaiyong1978/CLFS-for-LoongArch/releases/download/8.0/loongarch64-clfs-8.0-cross-tools-gcc-full.tar.xz "$task_root/cross-tools"
        "$task_root/cross-tools/bin/loongarch64-unknown-linux-gnu-gcc" --version
        qemu-loongarch64-static --version
        ;;
    rvv)
        install_rvv
        ;;
    spacemit)
        extract https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz "$task_root/spacemit-toolchain-linux-glibc-x86_64-v1.2.4"
        extract https://archive.spacemit.com/spacemit-ai/qemu/jdsk-qemu-v10.0.2.tar.gz "$task_root/jdsk-qemu"
        "$task_root/jdsk-qemu/bin/qemu-riscv64" --version
        ;;
    xuantie)
        # These exact vendor packages were preinstalled on upstream's runner.
        # Do not silently substitute a different compiler or generic QEMU.
        : "${XUANTIE_GCC_URL:?Set the fork variable XUANTIE_GCC_URL for Xuantie GCC V3.4.0}"
        : "${XUANTIE_GCC_SHA256:?Set XUANTIE_GCC_SHA256}"
        : "${XUANTIE_QEMU_URL:?Set XUANTIE_QEMU_URL for Xuantie QEMU V5.4.1}"
        : "${XUANTIE_QEMU_SHA256:?Set XUANTIE_QEMU_SHA256}"
        extract "$XUANTIE_GCC_URL" "$task_root/Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.4.0" "$XUANTIE_GCC_SHA256"
        extract "$XUANTIE_QEMU_URL" "$task_root/xuantie-qemu-x86_64-Ubuntu-20.04-V5.4.1" "$XUANTIE_QEMU_SHA256"
        "$task_root/xuantie-qemu-x86_64-Ubuntu-20.04-V5.4.1/bin/qemu-riscv64" --version
        ;;
    elf32|elf64)
        bits=${profile#elf}
        triple=riscv${bits}-unknown-elf
        prefix="$task_root/riscv${bits}-elf"
        apt-get install -y autoconf automake libtool libboost-regex-dev libboost-system-dev libboost-dev libfdt-dev device-tree-compiler
        extract "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2025.01.20/riscv${bits}-elf-ubuntu-22.04-gcc-nightly-2025.01.20-nightly.tar.xz" "$prefix"
        export PATH="$prefix/bin:$PATH"
        checkout https://github.com/riscv/riscv-pk.git d8659a4e8e888bdc9caf840ad17bfe83239b1d64 "$task_downloads/riscv-pk"
        checkout https://github.com/riscv-software-src/riscv-isa-sim.git 5ef9a61f5fecdb9bf77da155172c8018ce820308 "$task_downloads/riscv-isa-sim"
        if [ "$bits" = 32 ]; then abi=ilp32d; else abi=lp64d; fi
        mkdir "$task_downloads/riscv-pk/build" "$task_downloads/riscv-isa-sim/build"
        (cd "$task_downloads/riscv-pk/build"
         ../configure --prefix="$prefix" --with-arch="rv${bits}gc_zicsr_zifencei" --host="$triple" --with-abi="$abi"
         make -j 2
         make install)
        (cd "$task_downloads/riscv-isa-sim/build"
         ../configure --prefix="$prefix"
         make -j 2
         make install)
        test -x "$prefix/bin/spike"
        test -x "$prefix/$triple/bin/pk"
        ;;
    coverage)
        apt-get install -y lcov qemu-user-static libvulkan-dev mesa-vulkan-drivers
        case "$GITHUB_JOB" in
            linux-gcc-x64)
                case "${FORK_MATRIX_NAME:-}" in
                    avx512vnni) require_cpu avx512f avx512_vnni ;;
                    avx512) require_cpu avx512f ;;
                    avx2) require_cpu avx2 fma f16c ;;
                    avx) require_cpu avx ;;
                    sse2) require_cpu sse2 ;;
                esac
                ;;
            linux-gcc-x64-sde*)
                extract https://github.com/nihui/ncnn-assets/releases/download/toolchain/sde-external-10.13.1-2026-07-28-lin.tar.xz \
                    "$task_root/sde-external-10.13.1-2026-07-28-lin" 94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50
                ;;
            linux-gcc-riscv64-rvv) install_rvv ;;
            linux-gpu-*) require_cpu avx512f avx512_vnni ;;
            linux-gcc-cross)
                case "${FORK_MATRIX_ARCH:?missing matrix architecture}" in
                    arm|arm-noinlineasm) triple=arm-linux-gnueabi ;;
                    armhf-*) triple=arm-linux-gnueabihf ;;
                    aarch64-*) triple=aarch64-linux-gnu ;;
                    mipsisa32r6el) triple=mipsisa32r6el-linux-gnu ;;
                    mipsisa64r6el) triple=mipsisa64r6el-linux-gnuabi64 ;;
                    powerpc) triple=powerpc-linux-gnu ;;
                    powerpc64le) triple=powerpc64le-linux-gnu ;;
                    riscv64) triple=riscv64-linux-gnu ;;
                    loongarch64-*) triple=loongarch64-linux-gnu ;;
                    *) echo 'Unknown cross-compiler matrix entry' >&2; exit 1 ;;
                esac
                apt-get install -y "g++-$triple"
                "$triple-g++" --version
                ;;
        esac
        ;;
    *) echo "Unknown provisioning profile: $profile" >&2; exit 1 ;;
esac

gcc --version
cmake --version
