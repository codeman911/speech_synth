# Distributed Training Script Design for Higgs Audio TTS Model

## 1. Overview

This document outlines the design for refactoring the `train_tts.sh` script to efficiently run distributed training across multiple nodes (2 nodes with 8 GPUs each) for the Higgs Audio TTS model. The refactored script will maintain compatibility with the existing training command while optimizing resource utilization and training performance.

## 2. Current Implementation Analysis

### 2.1 Existing Script (`train_tts.sh`)
The current script uses SLURM for job scheduling and runs training with the following characteristics:
- 2 nodes with 8 GPUs each (16 total GPUs)
- Uses `srun` to launch training processes
- Activates a conda environment named `tts_env`
- Runs training with `torch.distributed.run`
- Uses a custom trainer script (`trainer.py`) with specific parameters

### 2.2 Required Training Command
The user wants to run training with `train_v2_ddp.py` using this command:
```bash
torchrun --nproc_per_node=8 trainer/train_v2_ddp.py \
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

## 3. Refactored Script Design

### 3.1 Key Design Principles
1. **Simplicity**: Avoid over-engineering while ensuring robust multi-node execution
2. **Compatibility**: Maintain the same conda environment and training parameters
3. **Efficiency**: Optimize resource allocation for 2 nodes with 8 GPUs each
4. **Reliability**: Implement proper error handling and process synchronization

### 3.2 Script Structure
The refactored script will:
1. Configure SLURM job parameters for 2 nodes with 8 GPUs each
2. Set up environment variables for distributed training
3. Launch training processes on each node using `torchrun`
4. Ensure proper coordination between nodes for distributed training

### 3.3 SLURM Configuration
- `#SBATCH --nodes=2`: Use 2 nodes
- `#SBATCH --gres=gpu:8`: Allocate 8 GPUs per node
- `#SBATCH --ntasks-per-node=1`: One task per node
- `#SBATCH --cpus-per-task=64`: Allocate sufficient CPUs (adjustable based on needs)

### 3.4 Distributed Training Setup
1. Determine the master node address for coordination
2. Set up rendezvous parameters for `torchrun`
3. Launch training processes on each node with appropriate distributed parameters

## 4. Implementation Plan

### 4.1 Environment Setup
- Use the same conda environment (`tts_env`)
- Set necessary environment variables for NCCL and distributed training
- Configure OMP_NUM_THREADS for optimal CPU utilization

### 4.2 Training Command Mapping
Map the user's `torchrun` command to the SLURM multi-node environment:
- `--nproc_per_node=8`: 8 processes per node (one per GPU)
- `--nnodes=2`: Total of 2 nodes
- `--node_rank`: Automatically determined by SLURM
- `--master_addr`: Set to the first node in the allocation
- `--master_port`: Use a fixed port (e.g., 29500)

### 4.3 Process Launching
Use `srun` to launch the training command on each node:
- Launch one process per node
- Each process will spawn 8 local processes (one per GPU)

## 5. Resource Allocation and Optimization

### 5.1 GPU Utilization
- 16 total GPUs (2 nodes × 8 GPUs)
- Efficient distribution of training workload
- Proper NCCL configuration for inter-node communication

### 5.2 CPU Allocation
- Allocate sufficient CPUs per task for data loading and preprocessing
- Set OMP_NUM_THREADS to optimize CPU usage without oversubscription

### 5.3 Memory Management
- Configure appropriate batch sizes for distributed training
- Utilize mixed precision training (bf16) to reduce memory footprint

## 6. Error Handling and Monitoring

### 6.1 Process Synchronization
- Implement proper barriers for training start and completion
- Handle node failures gracefully

### 6.2 Logging and Debugging
- Enable NCCL debug logging for troubleshooting
- Configure appropriate logging levels for training progress

## 7. Validation and Testing

### 7.1 Single Node Verification
- Test the refactored script on a single node first
- Verify training parameters and model convergence

### 7.2 Multi-Node Testing
- Run on 2 nodes to verify distributed training functionality
- Monitor performance scaling and resource utilization