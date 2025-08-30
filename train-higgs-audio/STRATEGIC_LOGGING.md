# Strategic Logging for Zero-Shot Voice Cloning Training

This document explains how to use the strategic logging system for debugging zero-shot voice cloning training in the Higgs Audio model.

## Overview

The strategic logging system provides transparency into the Higgs Audio zero-shot voice cloning training pipeline without disrupting the existing training workflow. It consists of four specialized logging callbacks that focus on different aspects of the training process:

1. **Input Logger**: Verifies model inputs and audio conditioning
2. **Output Logger**: Tracks model predictions vs. ground truth
3. **Shared Attention Logger**: Verifies DualFFN training with shared attention
4. **Zero-Shot Verification Logger**: Confirms zero-shot voice cloning capabilities

## Usage

To enable strategic logging during training, use the following command line arguments:

```bash
python train_v2_ddp.py \
  --model_path /path/to/model \
  --train_data_file /path/to/data.json \
  --enable_strategic_logging \
  --strategic_logging_steps 100 \
  [other arguments...]
```

### Viewing Logs

The strategic logs will be visible in the training output with a timestamp prefix:

```
[2023-08-15 14:32:45] STRATEGIC LOG:
=== Zero-Shot Voice Cloning Training Log - Step 100 ===
...
```

If you want to save the logs to a file, you can redirect stderr:

```bash
python train_v2_ddp.py \
  --model_path /path/to/model \
  --train_data_file /path/to/data.json \
  --enable_strategic_logging \
  --strategic_logging_steps 100 \
  [other arguments...] 2>&1 | tee training_log.txt
```

### Command Line Arguments

- `--enable_strategic_logging`: Enable strategic logging for zero-shot voice cloning training
- `--strategic_logging_steps N`: Log strategic information every N steps (default: 100)

## Logging Components

### 1. Input Logger (`InputLoggerCallback`)

Logs detailed information about model inputs at specified intervals:

- Complete input sequence with special tokens highlighted
- Shape and size of all input tensors:
  - `input_ids`: Shape, data type, and sample values
  - `attention_mask`: Shape and sample values
  - `audio_features`: Shape and statistics (if present)
  - `audio_ids_concat`: Shape and sample values (if present)
  - `audio_waveforms_concat`: Shape and basic statistics (min, max, mean)
- Token decoding for text portions to show actual content
- Audio context information (number of audio segments, their positions)

### 2. Output Logger (`OutputLoggerCallback`)

Logs model predictions vs. ground truth:

- First and last 10 values of predicted vs. label comparisons for text tokens
- Decoded target text (what the model should generate)
- Decoded predicted text (what the model actually generated)
- Audio token predictions vs. ground truth (first and last 10 values)
- Loss components breakdown when available

### 3. Shared Attention Logger (`SharedAttentionLoggerCallback`)

Verifies that the model's DualFFN is training properly with shared attention:

- Attention pattern analysis for text-to-audio and audio-to-text cross-attention
- Verification that text layers are learning language representations
- Statistics on attention weights distribution
- Gradient flow analysis for shared components

### 4. Zero-Shot Verification Logger (`ZeroShotVerificationLoggerCallback`)

Confirms the model is properly learning zero-shot voice cloning:

- Reference audio conditioning effectiveness
- Voice cloning consistency across different speakers
- Cross-lingual adaptation verification (if applicable)
- Audio quality metrics (when reconstruction is available)

## Log Output Format

All logs are formatted for easy human readability and saved to the logging directory:

```
=== Zero-Shot Voice Cloning Training Log - Step 100 ===
Timestamp: 2023-08-15 14:32:45

Input Sequence Analysis:
├── Tokenized Input: [1, 32006, 15875, 263, 304, 369, 264, 1700, 369, 32007, ...]
├── Decoded Text: "Generate speech in the provided voice. Hello, how are you today?<|AUDIO_IN|><|AUDIO_OUT|>"
├── Audio Tokens: 1 AUDIO_IN tokens, 1 AUDIO_OUT tokens
└── Sequence Length: 128 tokens

Input Tensor Details:
├── input_ids: torch.Size([4, 128]) - dtype: torch.int64
├── attention_mask: torch.Size([4, 128]) - dtype: torch.int64
├── audio_features: torch.Size([1, 768, 750]) - dtype: torch.float32
│   ├── Min: -2.34, Max: 3.12, Mean: 0.04
│   └── Contains Whisper embeddings for voice conditioning
└── audio_ids_concat: torch.Size([8, 240]) - dtype: torch.int64
    ├── Codebook 0 sample: [12, 34, 56, 78, 90, ...]
    └── Total audio tokens: 240 across 8 codebooks

Reference Audio Conditioning:
├── Whisper Embedding Status: ENABLED (encode_whisper_embed=True)
├── Reference Audio Segments: 1
├── Segment 1: Position 42, Length 240 tokens, Speaker ID: speaker_001
└── Segment 2: Position 87, Length 240 tokens, Speaker ID: speaker_002

Model Output Analysis:
├── Loss: 5.2418
├── Gradient Norm: 0.4385
├── Target Text: "This is a test of the voice cloning system."
├── Predicted Text: "This is a test of the voice cloning system."
├── Text Match Accuracy: 97.2%
└── Audio Token Accuracy: 89.4%

Detailed Text Token Comparison (First 10):
Target:    [1234, 567, 890, 111, 222, 333, 444, 555, 666, 777]
Predicted: [1234, 567, 891, 111, 222, 333, 445, 555, 666, 778]
Match:     [True, True, False, True, True, True, False, True, True, False]

Detailed Audio Token Comparison (First 10):
Target:    [12, 34, 56, 78, 90, 12, 34, 56, 78, 90]
Predicted: [12, 34, 56, 78, 91, 12, 34, 57, 78, 90]
Match:     [True, True, True, True, False, True, True, False, True, True]

Shared Attention Verification:
├── Cross-Modal Attention Score: 0.76
├── Text Layer Activation: Normal
├── Audio Layer Activation: Normal
└── Gradient Flow Status: HEALTHY

Zero-Shot Capability Metrics:
├── Voice Cloning Consistency: 92.3%
├── Cross-Lingual Adaptation: 87.6%
├── Reference Audio Conditioning Effectiveness: 94.1%
└── Overall Zero-Shot Score: 91.3%
```

## Performance Considerations

1. **Minimal Overhead**: Logging only occurs at the configured logging intervals
2. **Selective Logging**: Only a subset of samples is logged for detailed analysis
3. **Efficient Formatting**: Prettified output is generated without expensive operations
4. **Memory Management**: Logs are written directly to files without accumulating in memory

## Troubleshooting

If you encounter issues with the strategic logging:

1. Ensure all required dependencies are installed
2. Check that the logging callbacks file is in the correct location
3. Verify that the command line arguments are correctly specified
4. Check the training logs for any error messages related to the callbacks

## Development

To modify or extend the logging callbacks:

1. Edit the `trainer/strategic_logging_callbacks.py` file
2. The callbacks follow the Hugging Face TrainerCallback interface
3. Each callback implements the `on_log` method which is called at each logging step
4. Use the `kwargs` parameter to access model inputs, outputs, and other training state

## Testing

To test the logging callbacks, run:

```bash
python test_strategic_logging.py
```

This will execute each callback with mock data to verify they work correctly.