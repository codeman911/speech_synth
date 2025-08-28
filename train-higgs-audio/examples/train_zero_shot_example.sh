#!/bin/bash

# Example training script for zero-shot voice cloning with Higgs Audio v2
# This script demonstrates how to use the new train_v2.py script

# Single GPU training example
echo "Starting single GPU training for zero-shot voice cloning..."

python trainer/train_v2.py \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --train_data_file ./examples/chatml_zero_shot_train.json \
    --validation_split 0.05 \
    --output_dir ./output_zero_shot \
    --use_lora \
    --lora_rank 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --bf16

echo "Training completed!"

# Multi-GPU training example with torchrun
echo "Starting multi-GPU training for zero-shot voice cloning..."

torchrun --nproc_per_node=4 trainer/train_v2_ddp.py \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --train_data_file ./examples/chatml_zero_shot_train.json \
    --validation_split 0.05 \
    --output_dir ./output_zero_shot_ddp \
    --use_lora \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --bf16

echo "Multi-GPU training completed!"