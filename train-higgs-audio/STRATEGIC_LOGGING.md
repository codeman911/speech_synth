# Strategic Logging for Zero-Shot Voice Cloning Training

## Overview

This document explains how to use the strategic logging feature for the Higgs Audio zero-shot voice cloning training pipeline. The strategic logging system provides transparency into the training process without disrupting the existing training workflow.

## Enabling Strategic Logging

To enable strategic logging, use the `--enable_strategic_logging` flag when running the training script:

```bash
torchrun --nproc_per_node=NUM_GPUS trainer/train_v2_ddp.py \
    --model_path /path/to/model \
    --audio_tokenizer_path /path/to/audio_tokenizer \
    --train_data_file /path/to/train_data.json \
    --output_dir /path/to/output \
    --enable_strategic_logging \
    --strategic_logging_steps 100
```

### Command Line Arguments

- `--enable_strategic_logging`: Enable strategic logging for zero-shot voice cloning training
- `--strategic_logging_steps`: Log strategic information every X steps (default: 100)

## What Gets Logged

The strategic logging system includes four specialized callbacks that log different aspects of the training process:

### 1. Input Logger
Logs detailed information about model inputs at specified intervals:
- Complete input sequence with special tokens highlighted
- Shape and size of all input tensors
- Token decoding for text portions to show actual content
- Audio context information (number of audio segments, their positions)

### 2. Output Logger
Logs model predictions vs. ground truth for both text and audio:
- First and last 10 values of predicted vs. label comparisons for text tokens
- Decoded target text (what the model should generate)
- Decoded predicted text (what the model actually generated)
- Audio token predictions vs. ground truth (first and last 10 values)

### 3. Shared Attention Logger
Verifies that the model's DualFFN is properly training with shared attention:
- Attention pattern analysis for text-to-audio and audio-to-text cross-attention
- Verification that text layers are learning language representations
- Statistics on attention weights distribution

### 4. Zero-Shot Verification Logger
Confirms the model is learning zero-shot voice cloning capabilities:
- Reference audio conditioning effectiveness
- Voice cloning consistency across different speakers
- Cross-lingual adaptation verification (if applicable)

## Log Output Format

All logs are formatted for easy human readability and include timestamp information:

```
=== Zero-Shot Voice Cloning Training Log - Step 100 ===
Timestamp: 2023-08-15 14:32:45

Input Sequence Analysis:
├── Tokenized Input Shape: torch.Size([4, 512])
├── Decoded Text (first sample): "<|system|>Generate speech in the provided voice.<|end|><|user|>Hello, how are you today?<|AUDIO_IN|><|end|><|assistant|><|AUDIO_OUT|><|end|>"
├── Sample input_ids (first 20): [1, 32006, 15875, 263, 304, 369, 264, 1700, 369, 32007, ...]
├── Audio Tokens: AUDIO_IN tokens: torch.Size([2, 8, 100]), AUDIO_OUT tokens: torch.Size([1, 8, 200])
```

## Troubleshooting

### No Logs Appear
If you don't see any strategic logs:
1. Make sure you've enabled the `--enable_strategic_logging` flag
2. Check that the logging steps match your training configuration
3. Verify that the callbacks are being properly imported

### ImportError for Callbacks
If you see import errors for the strategic logging callbacks:
1. Make sure the `callbacks` directory is in your Python path
2. Verify that all required dependencies are installed
3. Check that there are no naming conflicts with other modules

## Best Practices

1. **Logging Frequency**: Set `--strategic_logging_steps` to a reasonable value (e.g., 100) to avoid overwhelming output
2. **Performance**: Strategic logging has minimal performance impact but can be disabled for production training
3. **Analysis**: Use the logs to verify that your training data is being processed correctly and that the model is learning as expected
