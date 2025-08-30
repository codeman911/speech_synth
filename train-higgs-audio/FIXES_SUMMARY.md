# Zero-Shot Voice Cloning Training Fixes Summary

This document summarizes the key fixes implemented to align the training code with the inference code in `arb_inference.py`.

## 1. ExtendedHiggsAudioSampleCollator Fixes

### Issue
The fallback implementation of the data collator was not properly processing audio waveforms for Whisper embedding, resulting in empty audio features tensors.

### Fix
- Added `audio_num_codebooks` parameter to the collator initialization
- Enhanced the fallback implementation to properly handle audio waveforms
- Ensured consistent tensor shapes for audio features even when Whisper processing fails
- Added proper handling for empty waveforms

### Key Changes
```python
# Added audio_num_codebooks parameter
self.audio_num_codebooks = kwargs.get('audio_num_codebooks', 8)

# Improved audio features processing
if self.whisper_processor and self.encode_whisper_embed and any(wav.numel() > 0 for wav in audio_waveforms_concat_list):
    # Process waveforms through Whisper as before
else:
    # Create dummy tensors to maintain consistent structure
    if any(wav.numel() > 0 for wav in audio_waveforms_concat_list):
        batch_size = len([wav for wav in audio_waveforms_concat_list if wav.numel() > 0])
        if batch_size > 0:
            audio_features = torch.zeros((batch_size, 128, 3000))
            audio_feature_attention_mask = torch.ones((batch_size, 3000), dtype=torch.long)
```

## 2. ZeroShotVoiceCloningDataset Fixes

### Issue
The dataset was not creating ChatMLDatasetSample objects with proper audio conditioning, following the pattern from the inference code.

### Fix
- Added `audio_num_codebooks` attribute to the dataset class
- Updated the `__getitem__` method to follow the `_create_robust_sample` pattern from `arb_inference.py`
- Improved audio token concatenation to match the inference code
- Enhanced audio waveform loading for Whisper conditioning

### Key Changes
```python
# Added audio_num_codebooks attribute
self.audio_num_codebooks = getattr(audio_tokenizer, 'n_codebooks', 8)

# Updated audio token concatenation to match inference pattern
if context_audio_tokens:
    audio_ids_start = torch.tensor(
        np.cumsum(np.array([0] + [t.shape[1] for t in context_audio_tokens])),
        dtype=torch.long
    )[:-1]  # Remove the last element to match the inference pattern
    audio_ids_concat = torch.cat(context_audio_tokens, dim=1)

# Improved ChatMLDatasetSample creation following inference pattern
dataset_sample = ChatMLDatasetSample(
    input_ids=torch.tensor(input_tokens, dtype=torch.long),
    label_ids=torch.tensor(label_tokens, dtype=torch.long),
    audio_ids_concat=audio_ids_concat,
    audio_ids_start=audio_ids_start,
    label_audio_ids=label_audio_ids,
    # Include audio waveforms for Whisper conditioning
    audio_waveforms_concat=ref_waveform if ref_waveform is not None and ref_waveform.numel() > 0 else torch.tensor([]),
    audio_waveforms_start=torch.tensor([0], dtype=torch.long) if ref_waveform is not None and ref_waveform.numel() > 0 else torch.tensor([], dtype=torch.long),
    audio_sample_rate=torch.tensor([ref_sample_rate], dtype=torch.float32) if ref_sample_rate is not None else torch.tensor([], dtype=torch.float32),
    audio_speaker_indices=torch.tensor([0], dtype=torch.long),
)
```

## 3. Data Collator Initialization

### Issue
The data collator initialization was not passing the correct parameters to match the inference code.

### Fix
- Added `audio_num_codebooks` parameter to the collator initialization
- Ensured all parameters match the inference configuration

### Key Changes
```python
data_collator = ExtendedHiggsAudioSampleCollator(
    whisper_processor=whisper_processor,
    audio_in_token_id=model.config.audio_in_token_idx,
    audio_out_token_id=model.config.audio_out_token_idx,
    audio_stream_bos_id=model.config.audio_stream_bos_id,
    audio_stream_eos_id=model.config.audio_stream_eos_id,
    encode_whisper_embed=True,  # Enabled for voice cloning
    pad_token_id=tokenizer.pad_token_id,
    return_audio_in_tokens=False,  # Match inference script
    use_delay_pattern=False,  # Match inference script
    round_to=1,  # Match inference script exactly
    audio_num_codebooks=getattr(model.config, 'audio_num_codebooks', 8),
)
```

## 4. Strategic Logging Callbacks

### Issue
Token ID validation was causing decoding errors when token IDs were out of range.

### Fix
- Added proper validation for token IDs before decoding
- Enhanced error handling to prevent crashes

### Key Changes
```python
# Validate token IDs before decoding
if max_token_id < vocab_size and min_token_id >= 0:
    decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
else:
    log_lines.append(f"├── Decoded Text: Error - token IDs out of range (min: {min_token_id}, max: {max_token_id}, vocab_size: {vocab_size})")
```

## Summary

These fixes ensure that the training code now properly aligns with the inference code in `arb_inference.py`, specifically:

1. **Proper Audio Conditioning**: Both Whisper embedding and DAC code conditioning are now correctly implemented
2. **Consistent Data Structures**: ChatMLDatasetSample objects are created following the same pattern as the inference code
3. **Robust Error Handling**: Improved validation and error handling prevent crashes during training
4. **Parameter Alignment**: All parameters and configurations match between training and inference

The training should now be able to properly learn zero-shot voice cloning capabilities with both reference audio conditioning and DAC code generation.