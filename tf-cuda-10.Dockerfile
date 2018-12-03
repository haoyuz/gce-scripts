# --------------------------------------------------------------------
# Dockerfile with experimental TensorFlow build with Cuda 10.0
# (forked from tfboyd's Dockerfile)
# --------------------------------------------------------------------

# Start from NVIDIA Cuda 10.0 image
FROM nvidia/cuda:10.0-base-ubuntu16.04

WORKDIR /root
ENV HOME /root

RUN apt-get update

# Install extras needed by most models
RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      curl \
      htop \
      zip \
      unzip

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
      python \
      python-dev \
      python-pip \
      python-setuptools \
      virtualenv \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv

# Install CUDA 10 and python2 and python 3.5 (default for Ubuntu 16.04)
RUN apt-get install -y --no-install-recommends \
      cuda-command-line-tools-10-0 \
      cuda-cublas-10-0 \
      cuda-cudart-10-0 \
      cuda-cufft-10-0 \
      cuda-curand-10-0 \
      cuda-cusolver-10-0 \
      cuda-cusparse-10-0 \
      libcudnn7=7.3.1.20-1+cuda10.0 \
      libnccl2=2.3.5-2+cuda10.0

# Setup Python3 environment
RUN pip3 install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
RUN pip3 install --upgrade setuptools
RUN pip3 install wheel pyyaml absl-py
RUN pip3 install https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp35-cp35m-linux_x86_64.whl

# Setup Python2 environment
RUN python -m pip install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
RUN python -m pip install --upgrade setuptools
RUN python -m pip install wheel pyyaml absl-py
RUN python -m pip install https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl

# Install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y bazel