#!/bin/bash
#-------------------------------------------------------------------------------
# Startup script sent to GCE VM instance to install Cuda, docker, nvidia-docker.
# The entire script takes about five minutes to finish.
#-------------------------------------------------------------------------------

LOGIN=haoyuzhang

#-------------------------------------------------------------------------------
# Install CUDA 9.0
# See updated script:
#   https://cloud.google.com/compute/docs/gpus/add-gpus
#-------------------------------------------------------------------------------

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  apt-get update
  apt-get install cuda-9-0 -y
fi

#-------------------------------------------------------------------------------
# Install docker
#   https://github.com/docker/docker-install
#-------------------------------------------------------------------------------

dpkg --configure -a
curl -fsSL get.docker.com -o get-docker.sh
sh get-docker.sh

usermod -a -G docker $LOGIN

#-------------------------------------------------------------------------------
# Install nvidia-docker2
# This script need to be frequently updated. Follow instructions here:
#   https://github.com/NVIDIA/nvidia-docker
#-------------------------------------------------------------------------------

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
apt-get purge -y nvidia-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

