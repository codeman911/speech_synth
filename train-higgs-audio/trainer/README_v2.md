# Higgs Audio v2 Zero-Shot Voice Cloning Training Pipeline

## Overview

This document describes the new training pipeline for Higgs Audio v2 that is specifically compatible with zero-shot voice cloning datasets in ChatML format. The implementation follows the same data processing pipeline as the inference script to ensure consistency between training and inference.

## Key Features

1. **Zero-Shot Voice Cloning Support**: Specifically designed for zero-shot voice cloning tasks
2. **ChatML Format Compatibility**: Processes datasets in ChatML JSON format
3. **Inference Alignment**: Follows the exact same data processing pipeline as the inference script
4. **LoRA Training Support**: Configurable LoRA training with target module selection
5. **Mixed Precision Training**: bfloat16 support for efficient training
6. **Multi-GPU Support**: DDP compatibility with proper parameter handling
7. **Whisper Integration**: Supports Whisper conditioning for better voice cloning

## Architecture

```
graph TD
    A[ChatML Dataset] --> B[ZeroShotVoiceCloningDataset]
    B --> C[HiggsAudioTrainer]
    C --> D[Training Loop]
    D --> E[Model Checkpoints]
    
    F[Audio Tokenizer] --> B
    G[Text Tokenizer] --> B
    H[Higgs Audio Model] --> C
    I[Whisper Processor] --> J[ExtendedHiggsAudioSampleCollator]
    J --> C
```

## Files

- `train_v2.py`: Single GPU training script
- `train_v2_ddp.py`: Multi-GPU training script with DDP support
- `chatml_zero_shot_example.json`: Sample ChatML dataset format
- `merger.py`: Script to merge LoRA adapters with base model
- `find_lora_adapters.py`: Helper script to locate LoRA adapters directories

## Dataset Format

The training pipeline expects datasets in ChatML JSON format with the following structure:

```
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant capable of generating speech in the voice of the provided reference audio."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Reference text that was spoken in the reference audio"
        },
        {
          "type": "audio",
          "audio_url": "path/to/reference/audio.wav",
          "raw_audio": "",
          "duration": null,
          "offset": null
        },
        {
          "type": "text",
          "text": "Please generate speech for given text in reference audio's voice: Target text to generate"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Target text to generate"
        },
        {
          "type": "audio",
          "audio_url": "path/to/target/audio.wav",
          "raw_audio": "",
          "duration": 2.13,
          "offset": null
        }
      ]
    }
  ],
  "start_index": 0,
  "speaker": "sample_id",
  "misc": {
    "sample_id": "sample_id",
    "ref_transcript": "Reference text",
    "target_transcript": "Target text",
    "duration": 2.13
  }
}
```

## Usage

### Single GPU Training

```bash
python trainer/train_v2.py \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --train_data_file ./data/train_chatml.json \
    --output_dir ./output \
    --use_lora \
    --lora_rank 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5
```

### Multi-GPU Training with torchrun

```bash
torchrun --nproc_per_node=4 trainer/train_v2_ddp.py \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --train_data_file ./data/train_chatml.json \
    --output_dir ./output \
    --use_lora \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5
```

## Configuration Parameters

### Model Arguments
- `--model_path`: Path to pretrained Higgs Audio model
- `--audio_tokenizer_path`: Path to audio tokenizer

### Data Arguments
- `--train_data_file`: Path to ChatML JSON training file
- `--eval_data_file`: Path to ChatML JSON evaluation file
- `--validation_split`: Fraction of training data to use for validation (0.0 to 1.0). If > 0, splits training data for validation. When used, the specified fraction of training data will be held out for validation and will not be used for training.

### Training Arguments
- `--output_dir`: Directory to save model checkpoints
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Training batch size per device
- `--learning_rate`: Learning rate for optimization

### LoRA Arguments
- `--use_lora`: Enable LoRA training
- `--lora_rank`: LoRA rank parameter
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: LoRA dropout rate

## Key Implementation Details

### Dataset Processing

The `ZeroShotVoiceCloningDataset` class handles the specific requirements of zero-shot voice cloning:

1. **ChatML Format Support**: 
   - Parses system, user, and assistant messages
   - Extracts audio URLs and text content
   - Maintains proper message ordering

2. **Audio Processing**:
   - Loads reference audio for conditioning
   - Encodes both reference and target audio
   - Resamples to 16kHz for Whisper compatibility

3. **Token Processing**:
   - Uses `prepare_chatml_sample` for text tokenization
   - Builds proper `ChatMLDatasetSample` objects
   - Handles audio token concatenation and indexing

### Collator Configuration

The `ExtendedHiggsAudioSampleCollator` is configured to match the inference script:

- **Whisper Integration**: Enabled for reference audio conditioning
- **Token IDs**: Uses model configuration for audio token indices
- **Delay Pattern**: Disabled to match inference behavior
- **Rounding**: Uses fixed round_to=1 to match inference

### Training Loop

The training loop follows standard Hugging Face Trainer patterns with customizations:

- **Loss Computation**: Handles both text and audio loss components
- **Mixed Precision**: Supports bfloat16 training
- **LoRA Training**: Integrates PEFT for parameter-efficient fine-tuning
- **Checkpointing**: Saves both full model and LoRA adapters

## Compatibility Features

### Inference Alignment
- **Same Audio Processing**: Uses identical audio encoding/decoding pipeline
- **Consistent Tokenization**: Shares tokenization logic with inference
- **Whisper Integration**: Matches inference Whisper conditioning

### Training Robustness
- **Error Handling**: Graceful handling of corrupted samples
- **Fallback Mechanisms**: Provides fallback implementations for missing components
- **Multi-GPU Support**: DDP compatibility with proper parameter handling

## Expected Outcomes

1. **Training Consistency**: Model trained with this pipeline will behave identically during inference
2. **Zero-Shot Capability**: Trained model will support zero-shot voice cloning as intended
3. **Performance**: Efficient training with LoRA and mixed precision support
4. **Compatibility**: Works with existing Higgs Audio model checkpoints and tokenizers

## Enhanced Debugging for LoRA Adapter Saving

To help debug issues with LoRA adapter saving, enhanced logging has been added to both training scripts:

1. **Detailed logging during LoRA setup**: After applying LoRA configuration, the scripts now log detailed information about the model structure
2. **Detailed logging during LoRA saving**: Before and during the saving process, the scripts log information about the model being saved and any errors that occur

If you're experiencing issues with LoRA adapters not being saved, check the logs for:
- "LoRA flag is set, attempting to save LoRA adapters..."
- "Model to save type: ..."
- Any error messages that might indicate why saving is failing

These enhanced logs will help identify whether the issue is with:
- The LoRA configuration not being applied correctly
- The model not being properly converted to a LoRA model
- Issues during the saving process

## LoRA Adapters and Model Merging

When training with LoRA enabled (`--use_lora`), the training scripts now save LoRA adapters separately from model checkpoints:

### Directory Structure
```
output/
├── checkpoint-100/           # Model checkpoint (full model when not using LoRA, not created when using LoRA)
├── checkpoint-200/           # Model checkpoint (full model when not using LoRA, not created when using LoRA)
├── lora_adapters/            # LoRA adapters (created only when using --use_lora)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
└── trainer_state.json
```

### Important Notes
1. **When using `--use_lora`**: Only LoRA adapters are saved in the `lora_adapters` directory, no full model checkpoints are created
2. **When not using `--use_lora`**: Full model checkpoints are saved in checkpoint directories
3. **To merge LoRA adapters with the base model, use the `lora_adapters` directory**

### Merging LoRA Adapters
To merge LoRA adapters with the base model, use the `merger.py` script:

```bash
python trainer/merger.py \
    --base_model_path bosonai/higgs-audio-v2-generation-3B-base \
    --lora_adapter_path ./output/lora_adapters \
    --output_path ./merged_model
```

### Finding LoRA Adapters
If you're unsure where your LoRA adapters are located, use the helper script:

```bash
python trainer/find_lora_adapters.py --path /path/to/your/training/output
```

This will search for valid LoRA adapters directories and show you their locations.

## Recent Fixes

### Checkpoint Saving Fix

A recent fix was implemented to resolve an issue with checkpoint saving during training:

- **Problem**: `TypeError: Trainer._save_checkpoint() takes 3 positional arguments but 4 were given`
- **Solution**: Updated the custom `_save_checkpoint` method in `train_v2_ddp.py` to properly handle different method signatures using a try-except approach
- **Impact**: Training with checkpoint saving now works correctly, including LoRA adapter saving during checkpoints

For more details about this fix, see [CHECKPOINT_FIX_SUMMARY.md](CHECKPOINT_FIX_SUMMARY.md).