# gce-scripts

Scripts to manage VMs on Google Cloud. This script can help you create
a large number of VM instances with GPUs, automatically setup CUDA
and Docker on the VMs, and run TensorFlow models.

## Workflow for testing and deploying models on Google Cloud VMs

1. Install Google Cloud SDK
   https://cloud.google.com/sdk/

2. Login to Google Cloud account in a terminal, and correctly configure the
   default account, project, and zone.
   ```bash
   gcloud auth login [ACCOUNT]
   # list current configuration, make changes if necessary
   gcloud config list
   ```

3. Clone this repo. Make changes to scripts to best fit your needs. Import
   functions defined in `manage_instances.sh` into the shell. A branch of
   the code `cuda-9.2` is available to install Cuda 9.2 on VMs.
   ```bash
   git clone https://github.com/haoyuz/gce-scripts.git
   cd gce-scripts
   . manage_instances.sh 
   ```

4. Create your first VM instance with correct configurations. You can set
   numbers of GPUs and CPUs, as well as size of memory in GB. By default
   the script will install Ubuntu 16.04 on those VMs. The VM startup script
   will be automatically launched on the VM to install CUDA and Docker. It
   takes about 5 min to finish, after which you should restart the VM.
   ```bash
   # Create a VM with 1 NVIDIA V100 GPU (can choose to have up to 12 vCPU and 78 GB memory)
   create_custom_instance_with_image_family <instance-name> $IMAGE_FAMILY $IMAGE_PROJECT 12 1 64
   # Create a VM with 8 NVIDIA V100 GPUs
   # create_custom_instance_with_image_family <instance-name> $IMAGE_FAMILY $IMAGE_PROJECT 96 8 512
   
   # Wait 5 minutes for VM startup script to finish...

   # Restart the instance after VM startup script finishes
   stop_instance <instance-name>
   start_instance <instance-name>
   ```

5. For the first time, you want to create a separate data disk, attach the disk
   to the VM, and preprocess all data on the disk (for example, convert to
   TFRecord files). You can later attach it to multiple VM instances, or clone
   the data disk (from data disk template).
   ```bash
   # Create a separate data disk of desired size.
   create_data_disk <data-disk-name> <size-in-gb> $DISK_TYPE
   # Attach the data disk to VM (in READ WRITE mode)
   attach_data_disk_rw <instance-name> <data-disk-name>

   # Login to VM, mount data disk, and prepare data
   gcloud compute ssh <instance-name>

   # ------------------ BASH INSIDE THE VM --------------------------
   # sudo mkdir /data
   # sudo chmod +rw /data
   # sudo mount /deb/sdb /data
   # (donwload and preprocess dataset, save to /data)
   # ------------------------- END ----------------------------------

   # You can re-attach data disk in READ ONLY mode to prevent later changes to
   # the data. Note that you have to re-mount the disk inside VM. A data disk
   # can be attached to multiple VMs in READ ONLY mode.
   detach_data_disk <instance-name> <data-disk-name>
   attach_data_disk <instance-name> <data-disk-name>  # READ ONLY
   ```

6. You can run TensorFlow using nvidia-docker inside the VM.
   ```bash
   # (Bash inside VM)
   sudo chmod a+r /data
   sudo mount -o ro /dev/sdb /data

   nvidia-docker run -it -v $HOME:$HOME -v /data:/data tensorflow/tensorflow:nightly-gpu bash
   ```
   Examples of script to run object detection model (SSD) are provided in
   `vm_mlperf_ssd_code_setup.sh`.
