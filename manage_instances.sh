#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# Create VM instances and prepare libraries, data, and programs to run MLPerf
# SSD model on Google Cloud.
#
# Expected to work in open source.
#
# Prerequisites:
# * Google Cloud SDK (gcloud command)
# ------------------------------------------------------------------------------

# set -e

GCLOUD=gcloud
COMPUTE="compute"
SSH="ssh"

# OS image. --image-family uses the latest version of OS in specified image family.
# See details: https://cloud.google.com/compute/docs/instances/create-start-instance
IMAGE_FAMILY="ubuntu-1604-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DATA_DISK_NAME="mlperf-ssd-data"
DISK_IMAGE=""
DISK_IMAGE_PROJECT=""
DISK_SIZE_GB="100"
DISK_TYPE="pd-ssd"

# Resource
# When creating an instance with GPU, each GPU can have up to 12 vCPUs and 78GB
# memory. Standard machine types do not have 12-vCPU instances. See details:
#   https://cloud.google.com/compute/docs/gpus/
ZONE="us-west1-b"
GPU_TYPE="nvidia-tesla-v100"
GPU_COUNT=1
MIN_CPU_PLATFORM="Intel Skylake"
CPU_COUNT=12
MEMORY_GB=64
BOOT_DISK_SIZE_GB="200"
VM_STARTUP_SCRIPT="vm_startup_script_ubuntu1604.sh"

# MACHINE_TYPE="n1-standard-64"
# MACHINE_TYPE="n1-standard-96"

# Data
COCO_DATASET_TFRECORD="gs://tf-gpu-mlperf-ssd/coco2017"
RESNET34_CHECKPOINT="gs://tf-gpu-mlperf-ssd/resnet34"

# Code
TF_BENCHMARKS_REPO="https://github.com/tensorflow/benchmarks.git"
TF_MODELS_REPO="https://github.com/tensorflow/models.git"

# Required for instances with GPUs
MAINTENANCE_POLICY="TERMINATE"


get_data_disk_name() {
  instance_name=$1
  echo "${instance_name}-data"
}

create_data_disk() {
  disk_name=$1
  disk_size_gb=$2
  echo "Creating data disk ${disk_name}, ${disk_size_gb}GB..."
  $GCLOUD $COMPUTE disks create ${disk_name} --size $disk_size_gb --type ${DISK_TYPE}
  echo "Disk ${disk_name} created."
}

clone_data_disk() {
  disk_name=$1
  disk_snapshot_name=$2
  echo "Creating data disk ${disk_name} using snapshot ${disk_snapshot_name}"
  $GCLOUD $COMPUTE disks create ${disk_name} --source-snapshot ${disk_snapshot_name} --type ${DISK_TYPE}
  echo "Disk ${disk_name} created."
}

attach_data_disk() {
  instance_name=$1
  disk_name=$2
  attach_mode='ro'
  echo "Attaching data disk ${disk_name} (READ ONLY) to VM instance ${instance_name}..."
  $GCLOUD $COMPUTE instances attach-disk ${instance_name} --disk ${disk_name} --mode ${attach_mode}
}

attach_data_disk_rw() {
  instance_name=$1
  disk_name=$2
  echo "Attaching data disk ${disk_name} (READ WRITE) to VM instance ${instance_name}..."
  $GCLOUD $COMPUTE instances attach-disk ${instance_name} --disk ${disk_name}
  echo "Disk ${disk_name} attached. Now you can mount the disk in VM instance ${instance_name}"
}

detach_data_disk() {
  instance_name=$1
  disk_name=$2
  echo "Detaching data disk ${disk_name} from VM instance ${instance_name}..."
  $GCLOUD $COMPUTE instances detach-disk ${instance_name} --disk ${disk_name}
}

get_accelerator_spec() {
  gpu_type=$1
  gpu_count=$2
  echo "type=${gpu_type},count=${gpu_count}"
}

create_custom_instance_with_image_family() {
  instance_name=$1
  image_family=$2
  image_project=$3
  cpu_count=$4
  gpu_count=$5
  mem_size_gb=$6
  accelerator_spec=$(get_accelerator_spec ${GPU_TYPE} ${gpu_count})
  echo "Creating instance ${instance_name}..."
  $GCLOUD $COMPUTE instances create ${instance_name} \
    --zone ${ZONE} --image-family ${image_family} --image-project ${image_project} \
    --min-cpu-platform "${MIN_CPU_PLATFORM}" --custom-cpu ${cpu_count} \
    --custom-memory ${mem_size_gb} --accelerator ${accelerator_spec} \
    --maintenance-policy=${MAINTENANCE_POLICY} \
    --boot-disk-type=${DISK_TYPE} --boot-disk-size=${BOOT_DISK_SIZE_GB} \
    --metadata-from-file startup-script=${VM_STARTUP_SCRIPT}
  echo "Instance created; it will take ~5 min to run the startup script, after which it's recommended to reboot the VM."
}

start_instance() {
  instance_name=$1
  echo "Starting instance ${instance_name}..."
  gcloud compute instances start ${instance_name}
  echo "Instance running. Connect using $GCLOUD $COMPUTE $SSH ${instance_name}"
}

stop_instance() {
  instance_name=$1
  echo "Stopping instance ${instance_name}..."
  gcloud compute instances stop ${instance_name}
  echo "Instance ${instance_name} stopped."
}

delete_instance() {
  instance_name=$1
  read -p "Deleting instance ${instance_name}. Type Y to confirm " -n 1 -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud compute instances delete ${instance_name}
    echo "Instance ${instance_name} deleted."
  else
    echo "Instance ${instance_name} was not deleted."
  fi
}

create_1gpu_mlperf_ssd_instance() {
  instance_name="$USER-mlperf-ssd-1gpu"
  data_disk_name="mlperf-ssd-data"
  create_custom_instance_with_image_family ${instance_name} $IMAGE_FAMILY $IMAGE_PROJECT 12 1 64
  attach_data_disk ${instance_name} ${data_disk_name}
}

start_1gpu_mlperf_ssd_instance() {
  instance_name="$USER-mlperf-ssd-1gpu"
  data_disk_name="mlperf-ssd-data"
  start_instance ${instance_name}
  attach_data_disk ${instance_name} ${data_disk_name}
}

create_8gpu_mlperf_ssd_instance() {
  instance_name="$USER-mlperf-ssd-8gpu"
  data_disk_name="mlperf-ssd-data"
  create_custom_instance_with_image_family ${instance_name} $IMAGE_FAMILY $IMAGE_PROJECT 96 8 512
  attach_data_disk ${instance_name} ${data_disk_name}
}

start_8gpu_mlperf_ssd_instance() {
  instance_name="$USER-mlperf-ssd-8gpu"
  data_disk_name="mlperf-ssd-data"
  start_instance ${instance_name}
  attach_data_disk ${instance_name} ${data_disk_name}
}

# INSTANCE_NAME="my-test-instance"
# create_data_disk $DATA_DISK_NAME $DISK_SIZE_GB
# clone_data_disk mlperf-ssd-data-2 mlperf-ssd-data-snapshot
# delete_instance "my-hello-world-vm"
# attach_data_disk $INSTANCE_NAME $DATA_DISK_NAME
# start_instance $INSTANCE_NAME
# stop_instance $INSTANCE_NAME
# create_custom_instance_with_image_family $INSTANCE_NAME $IMAGE_FAMILY $IMAGE_PROJECT 12 1 64

