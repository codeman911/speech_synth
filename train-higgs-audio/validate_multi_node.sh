#!/bin/bash
# Validation script for multi-node distributed training

echo "Validating multi-node distributed training setup..."

# Check if we're running under SLURM
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "ERROR: This script must be run under SLURM with sbatch"
    echo "Usage: sbatch validate_multi_node.sh"
    exit 1
fi

# Check SLURM environment
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Show node list
echo "Node list:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

# Check if required files exist
REQUIRED_FILES=(
    "trainer/train_v2_ddp.py"
    "../../higgs-audio/training_data/chatml/train_chatml_samples.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    else
        echo "Found required file: $file"
    fi
done

# Check if conda environment is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available"
    exit 1
else
    echo "conda is available"
fi

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "ERROR: torchrun is not available"
    exit 1
else
    echo "torchrun is available"
fi

echo "All validation checks passed. Ready for multi-node training."
echo "To run the full training, use: sbatch train_tts.sh"