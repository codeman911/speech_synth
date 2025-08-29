# Fix Model Loading Issue and Add --num_of_samples Argument

## Overview

This design addresses two issues in the `arb_inference.py` script:
1. Fix the model loading issue when using a local model path
2. Add a `--num_of_samples` argument to limit the number of samples processed

## Problem Analysis

### Model Loading Issue
The error occurs when trying to load a model from a local path. The Hugging Face library is interpreting the local path as a repository ID, which causes a validation error.

### Missing --num_of_samples Feature
The current implementation processes all samples in the input file. Users need the ability to limit processing to a specific number of samples for testing purposes.

## Solution Design

### 1. Fix Model Loading Issue

The issue is in the model loading code where we directly pass the model path to `HiggsAudioModel.from_pretrained`. We need to ensure that local paths are properly handled.

**Changes to make:**
- Add proper validation for the model path
- Handle local paths correctly by checking if the path exists locally before attempting to load from Hugging Face Hub

### 2. Add --num_of_samples Argument

**Changes to make:**
- Add a new command-line argument `--num_of_samples` with a default of None (process all samples)
- Modify the `process_chatml_file` method to respect this limit
- Update the main function to pass this parameter to the processing method

## Implementation Plan

### 1. Command Line Interface Changes

Add the new argument to the Click command:

```python
@click.option(
    "--num_of_samples",
    type=int,
    default=None,
    help="Number of samples to process (default: all samples)"
)
```

This should be added after the `--adaptive_max_tokens` option in the `@click.command()` decorator.

### 2. Main Function Changes

Update the main function signature to include the new parameter:

```python
def main(
    chatml_file,
    output_dir,
    model_path,
    audio_tokenizer_path,
    device,
    device_id,
    temperature,
    top_k,
    top_p,
    seed,
    max_new_tokens,
    adaptive_max_tokens,
    num_of_samples  # Add this parameter
):
    """
    Arabic Zero-Shot Voice Cloning Inference Script
    
    Process ChatML format data to generate Arabic speech with reference voice characteristics.
    """
    logger.info("Starting Arabic Voice Cloning Inference")
    
    # Initialize inference engine
    inference_engine = ArabicVoiceCloningInference(
        model_path=model_path,
        audio_tokenizer_path=audio_tokenizer_path,
        device=device,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=True,
        adaptive_max_tokens=adaptive_max_tokens
    )
    
    # CRITICAL: Validate pipeline configuration for debugging silence issues
    logger.info("\n🔍 Running comprehensive pipeline validation...")
    validation_results = inference_engine.validate_pipeline_configuration()
    
    # Check for critical issues that would cause silence generation
    critical_issues = validation_results.get("issues", [])
    if critical_issues:
        logger.error(f"\n🚨 CRITICAL ISSUES DETECTED: {critical_issues}")
        logger.error("🗺️ These issues may cause silence generation or poor voice cloning quality")
        logger.warning("⚠️ Proceeding with known issues - results may be suboptimal")
    else:
        logger.info("✅ Pipeline validation passed - proceeding with generation")
    
    # Process ChatML file with the new parameter
    results = inference_engine.process_chatml_file(
        chatml_file=chatml_file,
        output_dir=output_dir,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        num_of_samples=num_of_samples  # Add this line
    )
    
    # ... rest of existing code ...
```

### 3. Process Method Changes

Update the `process_chatml_file` method signature and implementation:

```python
def process_chatml_file(
    self,
    chatml_file: str,
    output_dir: str,
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    num_of_samples: Optional[int] = None  # Add this parameter
) -> List[Dict[str, Any]]:
    """
    Process a ChatML file and generate Arabic speech for all samples.
    
    Args:
        chatml_file: Path to ChatML JSON file
        output_dir: Directory to save generated audio files
        temperature: Sampling temperature
        top_k: Top-k sampling parameter  
        top_p: Top-p sampling parameter
        seed: Random seed
        num_of_samples: Number of samples to process (default: all samples)
        
    Returns:
        List of processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ChatML data
    logger.info(f"Loading ChatML data from {chatml_file}")
    with open(chatml_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        samples = [samples]
    
    # Apply the sample limit if specified
    if num_of_samples is not None and num_of_samples > 0:
        original_count = len(samples)
        samples = samples[:num_of_samples]
        logger.info(f"Processing only first {num_of_samples} samples (out of {original_count} total)")
    
    results = []
    
    # ... rest of existing code ...
```

### 4. Model Loading Fix

Add a helper method to properly handle model loading:

```python
def _load_model_safely(self, model_path: str):
    """
    Safely load model from either local path or Hugging Face Hub.
    
    Args:
        model_path: Path to model directory or Hugging Face model ID
        
    Returns:
        Loaded model instance
    """
    try:
        # Check if model_path is a local directory
        if os.path.exists(model_path) and os.path.isdir(model_path):
            logger.info(f"Loading model from local path: {model_path}")
            return HiggsAudioModel.from_pretrained(
                model_path,
                device_map=self._device,
                torch_dtype=torch.bfloat16,
                local_files_only=True  # This ensures we only load local files
            )
        else:
            # Treat as Hugging Face model ID
            logger.info(f"Loading model from Hugging Face Hub: {model_path}")
            return HiggsAudioModel.from_pretrained(
                model_path,
                device_map=self._device,
                torch_dtype=torch.bfloat16,
            )
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
```

Then replace the existing model loading code with a call to this method:

```python
# Replace this existing code in the __init__ method:
# self.model = HiggsAudioModel.from_pretrained(
#     model_path,
#     device_map=self._device,
#     torch_dtype=torch.bfloat16,
# )

# With this:
self.model = self._load_model_safely(model_path)
```

## Usage Examples

After implementing these changes, users will be able to run the script with the new argument:

```bash
# Process only the first 10 samples
python3 arb_inference.py --chatml_file data.json --num_of_samples 10

# Process all samples (default behavior)
python3 arb_inference.py --chatml_file data.json

# Process with a local model path (fixed)
python3 arb_inference.py --chatml_file data.json --model_path /path/to/local/model
```

## Testing Plan

1. Test with a local model path to verify the model loading fix
2. Test with a Hugging Face model ID to ensure backward compatibility
3. Test the `--num_of_samples` argument with various values:
   - No limit (default behavior)
   - Limit set to 0 (should process no samples)
   - Limit set to a number less than total samples
   - Limit set to a number greater than total samples (should process all samples)

## Backward Compatibility

These changes maintain full backward compatibility:
- The default behavior for model loading remains the same
- The `--num_of_samples` argument defaults to None, preserving existing behavior
- All existing command-line arguments continue to work as before

## Error Handling

The solution includes proper error handling:
- Clear error messages when model loading fails
- Graceful handling of invalid `--num_of_samples` values
- Logging to help diagnose issues

## Performance Impact

The changes have minimal performance impact:
- The model loading fix adds a simple file system check
- The sample limiting feature reduces processing time when used

## Conclusion

These changes will resolve the model loading issue when using local paths and provide users with greater control over the inference process by allowing them to limit the number of samples processed. The implementation maintains full backward compatibility while adding valuable functionality for testing and development workflows.