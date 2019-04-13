FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    cmake \
    libblas3 \
    libblas-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir cupy-cuda92==5.3.0 chainer==5.3.0\
    matplotlib==3.0.3 pandas==0.24.1 chainercv==0.12.0\
    Pillow==5.4.1 opencv-python==4.0.0.21
