FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    git \
    build-essential \
    ninja-build \
    python3 \
    python3-pip \
    pkg-config \
    gdb-multiarch \
    gcc-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    libglib2.0-dev \
    libfdt-dev \
    libpixman-1-dev \
    zlib1g-dev \
    qemu-user \
    qemu-user-static \
    ffmpeg   # ✅ REQUIRED for animation export

# ✅ Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib

# Build QEMU with plugins
RUN git clone https://gitlab.com/qemu-project/qemu.git /tmp/qemu && \
    cd /tmp/qemu && \
    git checkout stable-8.2 && \
    mkdir build && cd build && \
    ../configure --target-list=riscv64-linux-user --enable-plugins --enable-debug && \
    ninja -j"$(nproc)" && \
    ninja install && \
    mkdir -p /usr/local/lib/qemu/plugins && \
    cp tests/plugin/*.so /usr/local/lib/qemu/plugins/ && \
    rm -rf /tmp/qemu

ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]