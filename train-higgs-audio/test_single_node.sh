#!/bin/bash
# Test script for single-node functionality validation

#SBATCH --job-name=higgs_audio_test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

echo "Starting single-node test for Higgs Audio v2 training..."

# Simple test with 2 GPUs on single node
torchrun --nproc_per_node=2 trainer/train_v2_ddp.py \
  --model_path bosonai/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
  --train_data_file ../../higgs-audio/training_data/chatml/train_chatml_samples.json \
  --output_dir ./output_test \
  --save_steps 10 \
  --disable_evaluation \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --warmup_steps 10 \
  --bf16 \
  --use_lora

echo "Single-node test completed successfully!"