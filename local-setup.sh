#!/bin/bash
set -e

sudo apt update
sudo apt upgrade -y

sudo apt install -y \
    qemu-user \
    qemu-user-static \
    gcc-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    gdb-multiarch \
    git \
    libglib2.0-dev \
    libfdt-dev \
    libpixman-1-dev \
    zlib1g-dev \
    ninja-build \
    build-essential \
    python3 \
    python3-pip \
    pkg-config

# Build QEMU from source only if plugin support is needed
git clone https://gitlab.com/qemu-project/qemu.git ~/qemu-riscv
cd ~/qemu-riscv
git checkout stable-8.2
mkdir -p build
cd build
../configure --target-list=riscv64-linux-user --enable-plugins --enable-debug
make -j"$(nproc)"
sudo make install

echo "Local setup complete."