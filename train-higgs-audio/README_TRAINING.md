# Higgs Audio Training Scripts

This directory contains scripts for training the Higgs Audio model using both single-GPU and multi-GPU (DDP) setups.

## Single-GPU Training

For single-GPU training, use the [train_v2.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2.py) script with [run_training.sh](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/run_training.sh):

```bash
./run_training.sh
```

This script uses a single JSON file for training data:
- `training_data/chatml/train_chatml_samples.json`

## Multi-GPU (DDP) Training

For multi-GPU training, use the [train_v2_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2_ddp.py) script with [run_training_ddp.sh](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/run_training_ddp.sh):

```bash
./run_training_ddp.sh
```

This script requires a directory structure for training data. If you have a JSON file, you can convert it using the conversion script:

```bash
python convert_json_to_directory.py training_data/chatml/train_chatml_samples.json training_data/chatml/train_samples
```

## Common Parameters

Both training scripts are configured with:
- `save_steps`: 50
- `eval_steps`: 50
- `bf16`: Enabled for mixed precision training
- `learning_rate`: 5e-5
- `per_device_train_batch_size`: 4

## Required Paths

Before running either script, you need to update the following paths in the shell scripts:
1. `--model_path`: Path to your Higgs Audio model
2. `--audio_tokenizer_path`: Path to your audio tokenizer

## Running the Scripts

1. Update the model paths in the shell scripts
2. For single-GPU training: `./run_training.sh`
3. For multi-GPU training:
   - Convert JSON data to directory format (if needed)
   - Update paths in `run_training_ddp.sh`
   - Run: `./run_training_ddp.sh`