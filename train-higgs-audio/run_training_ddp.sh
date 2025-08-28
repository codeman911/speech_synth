#!/bin/bash

# Multi-GPU DDP training script using train_v2_ddp.py
# This script requires a directory structure for training data

echo "Setting up training data directory structure..."
mkdir -p training_data/chatml/train_samples

# For DDP training, you need to organize your data in a directory structure
# where each sample has:
# - audio file (wav or mp3)
# - transcript file (txt)
# - metadata.json file describing the samples

# If you have training_data/chatml/train_chatml_samples.json, you need to convert it
# to the directory format using the conversion script:
# python convert_json_to_directory.py training_data/chatml/train_chatml_samples.json training_data/chatml/train_samples

# Run multi-GPU training with torchrun
echo "Starting multi-GPU DDP training..."

torchrun --nproc_per_node=2 trainer/train_v2_ddp.py \
    --model_path /path/to/your/higgs_audio_model \
    --audio_tokenizer_path /path/to/your/audio_tokenizer \
    --train_data_dir training_data/chatml/train_samples \
    --output_dir ./output_ddp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --bf16 \
    --report_to tensorboard \
    --logging_dir ./logs_ddp

echo "DDP training completed. Check the output_ddp directory for results."