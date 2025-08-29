#!/bin/bash
#SBATCH --job-name=higgs_audio_v2
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -euo pipefail

# Rendezvous config for 2 nodes / 16 GPUs total
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
RDZV_ID=$SLURM_JOB_ID

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# One launcher per node; each spawns 8 local ranks => 16 total
srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -lc '
  eval "$(conda shell.bash hook)"
  conda activate tts_env
  torchrun --nnodes='"$NNODES"' \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id='"$RDZV_ID"' \
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
'