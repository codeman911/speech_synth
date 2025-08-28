#!/bin/bash

# Single-GPU training script using train_v2.py for zero-shot voice cloning
# Uses training_data/chatml/train_chatml_samples.json as training data

torchrun --nproc_per_node=1 trainer/train_v2.py \
    --model_path /path/to/your/higgs_audio_model \
    --audio_tokenizer_path /path/to/your/audio_tokenizer \
    --train_data_file training_data/chatml/train_chatml_samples.json \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --bf16 \
    --report_to tensorboard \
    --logging_dir ./logs \
    --task_type zero_shot_voice_cloning

echo "Training completed. Check the output directory for results."