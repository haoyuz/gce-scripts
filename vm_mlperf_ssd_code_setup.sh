#!/bin/bash

#-------------------------------------------------------------------------------
# Set up code for running MLPerf SSD model.
#-------------------------------------------------------------------------------

# Code
TF_BENCHMARKS_REPO="https://github.com/tensorflow/benchmarks.git"
TF_BENCHMARKS_DIR="$HOME/benchmarks"
TF_MODELS_REPO="https://github.com/tensorflow/models.git"
TF_MODELS_DIR="$HOME/models"
COCO_API_REPO="https://github.com/cocodataset/cocoapi.git"
COCO_API_DIR="$HOME/cocoapi"

# Docker image
DOCKER_IMAGE="gcr.io/google.com/tensorflow-performance/mlperf/ssd:latest"
# DOCKER_IMAGE="tensorflow/tensorflow:nightly-gpu"


mount_data_disk() {
  sudo mkdir /data
  sudo chmod a+r /data
  sudo mount -o ro /dev/sdb /data
}

download_benchmarks() {
  pushd $HOME
  git clone ${TF_BENCHMARKS_REPO}
  popd
}

download_and_build_official_models() {
  pushd $HOME
  git clone ${TF_MODELS_REPO}
  cd ${TF_MODELS_DIR}/research
  # Install protobuf compiler and use it to process proto files
  wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
  sudo apt-get -y install zip unzip
  unzip protobuf.zip
  ./bin/protoc object_detection/protos/*.proto --python_out=.
  popd
}

download_and_build_pycocotools() {
  pushd $HOME
  git clone ${COCO_API_REPO}
  cd ${COCO_API_DIR}/PythonAPI
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py --user
  sudo apt-get -y build-dep python-matplotlib  # install dependencies required by matplotlib
  pip install --user setuptools cython
  sudo apt-get install -y python-matplotlib
  python setup.py build_ext --inplace
  rm -rf build
  popd
}

prepare_code() {
  download_benchmarks
  download_and_build_official_models
  download_and_build_pycocotools
}

download_docker_image() {
  gcloud docker -- pull ${DOCKER_IMAGE}
}

train_ssd() {
  num_gpus=$1
  batch_size_per_split=$2
  variable_update=$3
  num_epochs=$4
  use_fp16=$5
  xla_compile=$6

  run_id=`date +%m%d%H%M`
  train_dir="$HOME/ssd_gce_${num_gpus}gpu_batch${batch_size_per_split}"
  if [[ -n "$use_fp16" && "$use_fp16" = True ]]; then
    train_dir=${train_dir}_fp16
  fi
  if [[ -n "$xla_compile" && "$xla_compile" = True ]]; then
    train_dir=${train_dir}_xla
  fi
  train_dir=${train_dir}_${run_id}

  nvidia-docker run -it -v $HOME:$HOME -v /data:/data \
    ${DOCKER_IMAGE} bash -c \
    "PYTHONPATH=${TF_MODELS_DIR}:${TF_MODELS_DIR}/research:${COCO_API_DIR}/PythonAPI \
    python ${TF_BENCHMARKS_DIR}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
      --model=ssd300 --data_name=coco \
      --num_gpus=${num_gpus} --batch_size=${batch_size_per_split} --variable_update=${variable_update} \
      --optimizer=momentum --weight_decay=5e-4 --momentum=0.9 \
      --backbone_model_path=/data/resnet34/model.ckpt-28152 --data_dir=/data/coco2017 \
      --num_epochs=${num_epochs} --num_warmup_batches=0 --train_dir=${train_dir} \
      --save_model_steps=1000 --max_ckpts_to_keep=250 --summary_verbosity=1 --save_summaries_steps=10 \
      --use_fp16=${use_fp16:-False} --xla_compile=${xla_compile:-False} --alsologtostderr"

  echo ${train_dir}
}

eval_all_checkpoints_ssd() {
  train_dir=$1
  variable_update=$2

  eval_dir=${train_dir}/eval

  for ckpt_index in `ls -v ${train_dir}/*.index`; do
    ckpt_name=${ckpt_index%.index}

    nvidia-docker run -it -v $HOME:$HOME -v /data:/data \
      ${DOCKER_IMAGE} bash -c \
      "PYTHONPATH=${TF_MODELS_DIR}:${TF_MODELS_DIR}/research:${COCO_API_DIR}/PythonAPI \
      python ${TF_BENCHMARKS_DIR}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model=ssd300 --data_name=coco \
        --batch_size=64 --num_batches=78 --num_warmup_batches=0 \
        --variable_update=${variable_update} \
        --data_dir=/data/coco2017 --train_dir=${ckpt_name} \
        --eval --eval_dir=${eval_dir} \
        --summary_verbosity=1 --alsologtostderr"
  done
}

train_then_eval_ssd() {
  num_gpus=$1
  batch_size_per_split=$2
  variable_update=$3
  num_epochs=$4
  use_fp16=$5
  xla_compile=$6

  output_file="train_log`date +%m%d%H%M`"

  ( train_ssd ${num_gpus} ${batch_size_per_split} ${variable_update} ${num_epochs} ${use_fp16} ${xla_compile} ) 2>&1 | tee ${output_file}
  train_dir=$( tail -1 ${output_file} )
  mv ${outfile_file} ${train_dir}
  eval_all_checkpoints_ssd ${train_dir} ${variable_update}
}

# train_then_eval_ssd 1 128 "parameter_server" 60 True True