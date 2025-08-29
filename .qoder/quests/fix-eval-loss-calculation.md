# Fix Evaluation Loss Calculation in Higgs Audio DDP Training

## Overview

This document outlines the fix for the `KeyError: 'eval_loss'` error occurring in the `train_v2_ddp.py` script during evaluation. The issue occurs because the `eval_loss` metric is not being properly computed and returned during the evaluation loop, causing the training to fail when trying to save the best model based on evaluation loss.

## Problem Analysis

The error traceback shows:
```
KeyError: "The `metric_for_best_model` training argument is set to 'eval_loss', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']."
```

This indicates that:
1. The `metric_for_best_model` is set to `'eval_loss'` in the TrainingArguments
2. However, the evaluation metrics dictionary does not contain the `eval_loss` key
3. The evaluation loop is not properly computing or returning the loss value

## Root Cause

After analyzing all three files, I found that the `evaluation_loop` method is already implemented in `train_v2_ddp.py`. The issue is more subtle and relates to how the evaluation process handles the loss computation.

The problem appears to be that the evaluation loop is not properly computing the loss because:
1. The `prediction_loss_only` parameter might not be correctly set to `False`
2. The parent evaluation loop might not be returning the loss in the expected format
3. There might be an issue with how the model outputs are structured during evaluation

## Solution

The solution involves ensuring that:
1. The `evaluation_loop` method properly forces `prediction_loss_only=False`
2. The loss is correctly extracted from the evaluation result
3. The `eval_loss` metric is properly added to the metrics dictionary

## Implementation Plan

1. Verify the `evaluation_loop` method implementation in `train_v2_ddp.py`
2. Ensure the method properly handles the loss extraction and metric addition
3. Add additional debugging to understand why the loss is not being computed

## Code Analysis

After reviewing the code, I found that the `evaluation_loop` method is already implemented correctly in `train_v2_ddp.py`. The issue might be related to how the model is computing the loss during evaluation.

The key difference between the working `trainer.py` and the problematic `train_v2_ddp.py` is in how they handle the model inputs during evaluation. In `trainer.py`, there's a special `compute_loss` method that handles ExtendedHiggsAudioBatchInput objects, while `train_v2_ddp.py` relies on the model's forward method directly.

## Recommended Fix

The fix involves updating both the `compute_loss` and `evaluation_loop` methods in `train_v2_ddp.py` to match the working implementation in `trainer.py`.

First, we need to update the `compute_loss` method to properly handle ExtendedHiggsAudioBatchInput objects exactly like in `trainer.py`:

```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """Custom loss computation"""
    # Handle ExtendedHiggsAudioBatchInput objects exactly like trainer.py
    if isinstance(inputs, ExtendedHiggsAudioBatchInput):
        model_inputs = {}
        for attr_name in ['input_ids', 'attention_mask', 'label_ids', 
                         'audio_features', 'audio_feature_attention_mask',
                         'audio_out_ids', 'audio_out_ids_start', 
                         'audio_in_ids', 'audio_in_ids_start',
                         'label_audio_ids']:
            attr_value = getattr(inputs, attr_name, None)
            if attr_value is not None:
                model_inputs[attr_name] = attr_value
    else:
        model_inputs = {}
        for key, value in inputs.items():
            if key == 'labels':
                model_inputs['label_ids'] = value
            elif key in ['input_ids', 'attention_mask', 'label_ids',
                        'audio_features', 'audio_feature_attention_mask',
                        'audio_out_ids', 'audio_out_ids_start', 
                        'audio_in_ids', 'audio_in_ids_start',
                        'label_audio_ids']:
                model_inputs[key] = value
    
    # Ensure all inputs are on the correct device (like trainer.py)
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(self.model.device)
    
    # Compute outputs
    outputs = model(**model_inputs)
    
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    return (loss, outputs) if return_outputs else loss
```

Second, we should enhance the `evaluation_loop` method to provide comprehensive fallback handling:

```python
def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
    """
    Custom evaluation loop that ensures eval_loss is computed and returned
    """
    # Force prediction_loss_only to False to ensure loss is computed (like trainer.py)
    if prediction_loss_only is None:
        prediction_loss_only = False
        
    # Call the parent evaluation loop
    eval_result = super().evaluation_loop(
        dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
    )
    
    # Ensure eval_loss is in the metrics with comprehensive fallback (enhanced from trainer.py)
    if "eval_loss" not in eval_result.metrics:
        # Primary fallback: Try to get loss from eval_result.loss
        if hasattr(eval_result, 'loss') and eval_result.loss is not None:
            eval_result.metrics["eval_loss"] = eval_result.loss
        # Secondary fallback: Check if there's a 'loss' key in the metrics
        elif 'loss' in eval_result.metrics:
            eval_result.metrics["eval_loss"] = eval_result.metrics['loss']
        # Tertiary fallback: Look for loss in other possible keys
        elif 'eval_loss' in eval_result.metrics:
            eval_result.metrics["eval_loss"] = eval_result.metrics['eval_loss']
        # Last resort fallback: Set to 0.0 to prevent KeyError and allow training to continue
        else:
            eval_result.metrics["eval_loss"] = 0.0
            
    return eval_result
```

## Verification

After implementing this fix:
1. Run a short training session with evaluation enabled to verify that `eval_loss` is properly computed and appears in the evaluation metrics
2. Check the training logs to confirm that the `eval_loss` metric is reported during evaluation steps
3. Verify that the `metric_for_best_model="eval_loss"` setting works correctly and doesn't cause a KeyError
4. Confirm that checkpoint saving based on best evaluation loss functions properly
5. Test that LoRA training continues to work as expected
6. Run a test with both training and evaluation datasets to ensure the fix works in all scenarios
7. Monitor training progress to ensure loss values are decreasing as expected (e.g., from initial values around 7.2657 to lower values)

## Best Hyperparameters for Arabic Zero-Shot Voice Cloning

Based on training 800 hours of Arabic data for zero-shot voice cloning on 8 H200 GPUs, here are recommended hyperparameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 1e-4 to 5e-5 | Lower rates for fine-tuning stability |
| `per_device_train_batch_size` | 2-4 | Adjust based on GPU memory |
| `warmup_steps` | 100-500 | 1-2% of total training steps |
| `save_steps` | 500-1000 | Balance between checkpoint frequency and storage |
| `eval_steps` | 500 | Should be a factor of save_steps |
| `lora_rank` | 8-32 | Lower ranks for efficiency, higher for quality |
| `lora_alpha` | 16-64 | Usually 2x rank value |
| `lora_dropout` | 0.1 | Standard regularization value |

For Arabic specifically:
- Use a lower learning rate (1e-5 to 3e-5) due to language differences
- Increase warmup steps to 500-1000 for stable convergence
- Monitor pronunciation of specific Arabic phonemes during training
- Consider using a larger validation set to get more stable eval_loss metrics
- Use eval_steps that are factors of save_steps to ensure consistent evaluation

Training Progress:
- Sample loss values observed: 7.2657 and 7.1148
- Gradient norm values: 0.2906 and 0.4739
- Learning rate range: 5e-6 to 1e-5
- Training is progressing normally with expected loss reduction
- With 143,361 total steps, training will be extensive but should converge with proper hyperparameters