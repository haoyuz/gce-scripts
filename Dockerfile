# --------------------------------------------------------------------
# Dockerfile to build MLPerf SSD image
# --------------------------------------------------------------------

# Start from tensorflow nightly GPU
FROM tensorflow/tensorflow:nightly-gpu

# File author / maintainer
MAINTAINER Haoyu Zhang

RUN apt-get update

# Install dependencies for pycocotools library. Note: this will not build and
# install pycocotools inside the docker image; instead, a built COCO Python API
# in host machine should to be mounted and imported in the model.
# (TODO: maybe install pycocotools directly inside docker.)
RUN apt-get install -y python-tk
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
# install dependencies required by matplotlib
RUN apt-get -y build-dep python-matplotlib
RUN pip install setuptools cython
RUN apt-get install -y python-matplotlib

# HACK: download and copy over ptxas binary file from Cuda 9.2. The ptxas binary
# in Cuda 9.0 (currently installed in nightly TensorFlow image) is known to
# miscompile XLA in TensorFlow. Ignore if don't use XLA.
RUN curl https://raw.githubusercontent.com/haoyuz/gce-scripts/cuda-9.2/cuda-bin/ptxas -o ptxas
RUN chmod +x ptxas
RUN mv ptxas /usr/local/cuda/bin/

