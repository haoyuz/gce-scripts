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
MLPERF_REPO="https://github.com/mlperf/training.git"
MLPERF_DIR="$HOME/training"

# See Dockerfile
DOCKER_IMAGE="gcr.io/google.com/tensorflow-performance/mlperf/ssd:latest"
PYTHONPATH=${TF_MODELS_DIR}:${TF_MODELS_DIR}/research:${COCO_API_DIR}/PythonAPI:${MLPERF_DIR}/compliance
DATA_DIR=/data/coco2017
BACKBONE_MODEL_PATH=/data/resnet34/model.ckpt-28152
GCE_VARIABLE_UPDATE_ARGS="--variable_update=replicated --all_reduce_spec=nccl --gradient_repacking=2 --network_topology=gcp_v100"

EVAL_STEPS_32x1="120000,160000,180000,200000,220000,240000"
EVAL_STEPS_128x1="30000,40000,45000,50000,55000,60000"
EVAL_STEPS_64x8="7500,10000,11250,12500,13750,15000"
EVAL_STEPS_128x8="3750,5000,5625,6250,6875,7500"

TARGET_ACCURACY=0.212
COMPLIANCE_FILE="$HOME/ssd_compliance_log.txt"


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

download_mlperf() {
  pushd $HOME
  git clone ${MLPERF_REPO}
  popd
}

prepare_code() {
  download_benchmarks
  download_and_build_official_models
  download_and_build_pycocotools
  download_mlperf
}

download_docker_image() {
  # First, activate service account that has download privilege
  # gcloud auth activate-service-account [ACCOUNT_NAME] --key-file=[PATH_TO_KEY_FILE]
  gcloud docker -- pull ${DOCKER_IMAGE}
}

run_experiment() {
  num_gpus=$1                # should be 1 or 8
  batch_size=$2
  eval_during_training_args=$3
  additional_args=$4
  exp_dir=$5

  num_epochs=60
  if [ ${num_gpus} == "8" ]; then num_epochs=80; fi
  variable_update_args=""
  if [ ${num_gpus} == "8" ]; then variable_update_args=${GCE_VARIABLE_UPDATE_ARGS}; fi
  datasets_num_private_threads=15
  if [ ${num_gpus} == "8" ]; then datasets_num_private_threads=100; fi
  num_inter_threads=25
  if [ ${num_gpus} == "8" ]; then num_inter_threads=160; fi

  nvidia-docker run -it -v $HOME:$HOME -v /data:/data ${DOCKER_IMAGE} \
    bash -c "PYTHONPATH=${PYTHONPATH} COMPLIANCE_FILE=${COMPLIANCE_FILE} \
      python /home/haoyuzhang/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model=ssd300 --data_name=coco \
        --data_dir=${DATA_DIR} --backbone_model_path=${BACKBONE_MODEL_PATH} \
        --optimizer=momentum --weight_decay=5e-4 --momentum=0.9 \
        --save_model_steps=1000 --max_ckpts_to_keep=5 \
        --summary_verbosity=1 --save_summaries_steps=100 \
        --num_gpus=${num_gpus} --batch_size=${batch_size} \
        --train_dir=${exp_dir} --eval_dir=${exp_dir}/eval \
        --use_fp16 --xla_compile \
        --num_epochs=${num_epochs} --num_eval_epochs=1.9 --num_warmup_batches=0 \
        ${eval_during_training_args} \
        --datasets_num_private_threads=${datasets_num_private_threads} --num_inter_threads=${num_inter_threads} \
        ${variable_update_args} ${additional_args}"
}

run_experiment_with_dir() {
  num_gpus=$1                # should be 1 or 8
  batch_size=$2
  eval_during_training_args=$3
  additional_args=$4

  exp_name="ssd_gpu${num_gpus}_batch${batch_size}_`date +%m%d%H%M`"
  exp_log=$HOME/${exp_name}.log
  exp_dir=$HOME/${exp_name}
  run_experiment ${num_gpus} ${batch_size} ${eval_during_training_args} ${additional_args} ${exp_dir} 2>&1 | tee ${exp_log}
}

mlperf_1gpu_experiment() {
  eval_during_training_args="--eval_during_training_at_specified_steps=${EVAL_STEPS_128x1}"
  additional_args="--ml_perf_compliance_logging --stop_at_top_1_accuracy=${TARGET_ACCURACY}"
  run_experiment_with_dir 1 128 ${eval_during_training_args} ${additional_args}
}

mlperf_8gpu_experiment() {
  eval_during_training_args="--eval_during_training_at_specified_steps=${EVAL_STEPS_64x8}"
  additional_args="--ml_perf_compliance_logging --stop_at_top_1_accuracy=${TARGET_ACCURACY}"
  run_experiment_with_dir 8 64 ${eval_during_training_args} ${additional_args}
}

convergence_tuning_1gpu_experiment() {
  eval_during_training_args="--eval_during_training_at_specified_epochs=`seq -s, 45 1 60`"
  additional_args="--stop_at_top_1_accuracy=${TARGET_ACCURACY}"
  run_experiment_with_dir 1 128 ${eval_during_training_args} ${additional_args}
}

convergence_tuning_8gpu_experiment() {
  eval_during_training_args="--eval_during_training_at_specified_epochs=`seq -s, 50 1 70`"
  additional_args="--stop_at_top_1_accuracy=${TARGET_ACCURACY}"
  run_experiment_with_dir 8 64 ${eval_during_training_args} ${additional_args}
}
