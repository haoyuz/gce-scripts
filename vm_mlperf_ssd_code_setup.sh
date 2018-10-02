#!/bin/bash

#-------------------------------------------------------------------------------
# Set up code for running MLPerf SSD model.
#-------------------------------------------------------------------------------

# Code
TF_BENCHMARKS_REPO="https://github.com/tensorflow/benchmarks.git"
TF_BENCHMARKS_DIR="benchmarks"
TF_MODELS_REPO="https://github.com/tensorflow/models.git"
TF_MODELS_DIR="models"
COCO_API_REPO="https://github.com/cocodataset/cocoapi.git"
COCO_API_DIR="cocoapi"

git clone ${TF_BENCHMARKS_REPO}
git clone ${TF_MODELS_REPO}
git clone ${COCO_API_REPO}

(
  cd ${TF_MODELS_DIR}/research
  # Install protobuf compiler and use it to process proto files
  wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
  sudo apt-get -y install zip unzip
  unzip protobuf.zip
  ./bin/protoc object_detection/protos/*.proto --python_out=.
)

(
  cd ${COCO_API_DIR}/PythonAPI
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py --user
  sudo apt-get -y build-dep python-matplotlib  # install dependencies required by matplotlib
  pip install --user setuptools cython matplotlib
  make
)

mount_disk() {
  sudo mkdir /data
  sudo chmod a+r /data
  sudo mount -o ro /dev/sdb /data
}

run_ssd_1gpu() {
  nvidia-docker run -it -v $HOME:$HOME -v /data:/data -v /tmp:/tmp \
    tensorflow/tensorflow:nightly-gpu bash -c \
    "PYTHONPATH=$HOME/${TF_MODELS_DIR}:$HOME/${TF_MODELS_DIR}/research:$HOME/${COCO_API_DIR}/PythonAPI \
    python $HOME/${TF_BENCHMARKS_DIR}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --model=ssd300 --data_name=coco \
    --num_gpus=1 --batch_size=128 --variable_update=parameter_server \
    --optimizer=momentum --weight_decay=5e-4 --momentum=0.9 \
    --backbone_model_path=/data/resnet34/model.ckpt-28152 --data_dir=/data/coco2017 \
    --num_epochs=60 --train_dir=/tmp/gce_1gpu_batch128_`date +%m%d%H%M` \
    --save_model_steps=5000 --max_ckpts_to_keep=250 --summary_verbosity=1 --save_summaries_steps=10 \
    --use_fp16 --alsologtostderr"
}

run_ssd_8gpu() {
  nvidia-docker run -it -v $HOME:$HOME -v /data:/data -v /tmp:/tmp \
    tensorflow/tensorflow:nightly-gpu bash -c \
    "PYTHONPATH=$HOME/${TF_MODELS_DIR}:$HOME/${TF_MODELS_DIR}/research:$HOME/${COCO_API_DIR}/PythonAPI \
    python $HOME/${TF_BENCHMARKS_DIR}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --model=ssd300 --data_name=coco \
    --num_gpus=8 --network_topology=gcp_v100 \
    --batch_size=64 --variable_update=parameter_server \
    --optimizer=momentum --weight_decay=5e-4 --momentum=0.9 \
    --backbone_model_path=/data/resnet34/model.ckpt-28152 --data_dir=/data/coco2017 \
    --num_epochs=60 --train_dir=/tmp/gce_8gpu_batch64_`date +%m%d%H%M` \
    --save_model_steps=1000 --max_ckpts_to_keep=250 --summary_verbosity=1 --save_summaries_steps=10 \
    --use_fp16 --alsologtostderr"
}