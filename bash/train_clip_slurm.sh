#!/bin/bash
#SBATCH --job-name=Search
#SBATCH --chdir=/home/htc/kchitranshi/      # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/kchitranshi/SCRATCH/%j.log     # Standard output and error log
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=gpu  # Specify the desired partition, e.g. gpu or big
#SBATCH --exclude=htc-gpu[020-023,037,038] # Only A40 GPU
#SBATCH --time=0-20:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --ntasks=1
#
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=

echo 'Getting node information'
date;hostname;id;pwd

echo 'Setting LANG to en_US.UTF-8'
LANG=en_US.UTF-8

which python
java -version

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Print GPUs'
/usr/bin/nvidia-smi

echo 'Running script'
cd Robust_mmfm
python vlm_eval/clip_train.py \
    --num_epochs 1 \
    --data_seeds 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 \
    --data_name MS_COCO \
    --method NONE \
    --batch_size 128 \
    --learning_rate 5e-7 \
    --save_model \
    --save_model_path ./fine_tuned_clip_models/NONE/
