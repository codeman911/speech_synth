# Higgs Audio v2 Distributed Training Documentation

## Overview

This document explains the usage and parameters of the refactored [train_tts.sh](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/train_tts.sh) script, which has been updated to use [train_v2_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2_ddp.py) for efficient distributed training across multiple nodes.

## Script Configuration

The refactored script is configured for:
- 2 nodes with 8 GPUs each (16 total GPUs)
- SLURM job scheduling
- Distributed training with `torchrun`
- Mixed precision training (bf16)
- LoRA fine-tuning support

## SLURM Parameters

```bash
#SBATCH --job-name=higgs_audio_v2
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
```

## Environment Variables

The script sets the following environment variables for optimal distributed training:

```bash
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
```

## Training Command

The script executes the following training command using `torchrun`:

```bash
torchrun --nnodes=$NNODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$RDZV_ID \
  trainer/train_v2_ddp.py \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --train_data_file ../../higgs-audio/training_data/chatml/train_chatml_samples.json \
    --output_dir ./output_expmt_v1 \
    --save_steps 5000 \
    --disable_evaluation \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 50 \
    --warmup_steps 500 \
    --bf16 \
    --use_lora
```

## Parameter Descriptions

- `--model_path`: Path to the pretrained Higgs Audio v2 model
- `--audio_tokenizer_path`: Path to the audio tokenizer
- `--train_data_file`: Path to the ChatML JSON training file
- `--output_dir`: Directory to save model checkpoints
- `--save_steps`: Save checkpoint every X updates steps
- `--disable_evaluation`: Disable evaluation during training
- `--per_device_train_batch_size`: Training batch size per device
- `--learning_rate`: Learning rate for optimization
- `--num_train_epochs`: Number of training epochs
- `--logging_steps`: Log every X updates steps
- `--warmup_steps`: Number of warmup steps
- `--bf16`: Enable bfloat16 mixed precision training
- `--use_lora`: Enable LoRA training

## Usage Instructions

1. Ensure SLURM is properly configured on your system
2. Make sure the `tts_env` conda environment is set up with all required dependencies
3. Verify that the training data file exists at the specified path
4. Run the script with `sbatch train_tts.sh`

## Resource Allocation

- **GPUs**: 16 total GPUs (2 nodes × 8 GPUs)
- **CPUs**: 64 CPUs per node
- **Memory**: Automatically managed by SLURM

## Error Handling

The script includes error handling with:
- `set -euo pipefail` for strict error checking
- NCCL error handling configuration
- Proper process synchronization between nodes

## Testing and Validation

For single-node testing, modify the SLURM parameters:
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
```

And adjust the training parameters accordingly.