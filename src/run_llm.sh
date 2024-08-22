#!/usr/bin/bash

# Log information
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Job starts at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Runs at:"
echo "$(hostnamectl)"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "Warning: CUDA_VISIBLE_DEVICES is not set. Manually set to two gpus."
fi

echo "Device (GPUs) := "$CUDA_VISIBLE_DEVICES
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# CUDA and Python environment
HOME_PATH=$(echo ~)
source "$HOME_PATH/micromamba/etc/profile.d/micromamba.sh"
micromamba activate env
# export CUDA_LAUNCH_BLOCKING=1

INPUT_DATA=data/model_generation/base.pkl
OUTPUT_DATA=data/model_generation/base.pkl
seed=0

python run_llm.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --template base \
    --max_new_tokens 10 \
    --use_chat_template \
    --seed $seed \
    --option_map $seed \
    --input $INPUT_DATA \
    --output $OUTPUT_DATA 

python cal_option.py $OUTPUT_DATA

echo "Q.E.D"
