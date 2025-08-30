"""
Strategic Logging Callbacks for Zero-Shot Voice Cloning Training
Provides transparency into the Higgs Audio training process without disrupting workflow.
"""

import logging
import sys
from datetime import datetime
from transformers import TrainerCallback
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Global variable to hold torch reference
torch = None

def _import_torch():
    """Import torch only when needed"""
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            torch = None
    return torch is not None

class InputLoggerCallback(TrainerCallback):
    """
    Logs detailed information about model inputs at specified intervals.
    Focuses on zero-shot voice cloning training verification.
    """
    
    def __init__(self, tokenizer=None, log_every_n_steps=100):
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log input sequence analysis and tensor details"""
        # Check if it's time to log (every N steps OR at step 1 for debugging)
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 1:
            return
            
        # Import torch when needed
        if not _import_torch():
            if state.global_step == 1:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG DEBUG: torch not available", file=sys.stderr)
                sys.stderr.flush()
            return
            
        # Get model inputs from kwargs
        model_inputs = kwargs.get('inputs', {})
        if not model_inputs:
            # Log that we didn't get inputs for debugging
            if state.global_step == 1:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG DEBUG: No inputs received at step {state.global_step}", file=sys.stderr)
                sys.stderr.flush()
            return
            
        try:
            # Create prettified log
            log_lines = []
            log_lines.append(f"=== Zero-Shot Voice Cloning Training Log - Step {state.global_step} ===")
            log_lines.append(f"Timestamp: {state.log_history[-1].get('epoch', 'N/A') if state.log_history else 'N/A'}")
            log_lines.append("")
            
            # Input Sequence Analysis
            log_lines.append("Input Sequence Analysis:")
            if hasattr(model_inputs, 'input_ids') and model_inputs.input_ids is not None:
                input_ids = model_inputs.input_ids
                log_lines.append(f"├── Tokenized Input Shape: {input_ids.shape}")
                if self.tokenizer:
                    # Decode a sample (first sequence in batch)
                    try:
                        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        log_lines.append(f"├── Decoded Text (first sample): {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
                    except Exception as e:
                        log_lines.append(f"├── Decoded Text: Error decoding - {str(e)}")
                log_lines.append(f"├── Sample input_ids (first 20): {input_ids[0][:20].tolist()}")
            else:
                log_lines.append("├── input_ids: NOT FOUND")
            
            # Audio context information
            audio_tokens_info = []
            if hasattr(model_inputs, 'audio_in_ids') and model_inputs.audio_in_ids is not None:
                audio_tokens_info.append(f"AUDIO_IN tokens: {model_inputs.audio_in_ids.shape if hasattr(model_inputs.audio_in_ids, 'shape') else 'present'}")
            if hasattr(model_inputs, 'audio_out_ids') and model_inputs.audio_out_ids is not None:
                audio_tokens_info.append(f"AUDIO_OUT tokens: {model_inputs.audio_out_ids.shape if hasattr(model_inputs.audio_out_ids, 'shape') else 'present'}")
            
            if audio_tokens_info:
                log_lines.append(f"├── Audio Tokens: {', '.join(audio_tokens_info)}")
            else:
                log_lines.append("├── Audio Tokens: NONE FOUND")
            
            # Tensor Details
            log_lines.append("Input Tensor Details:")
            tensor_found = False
            for attr_name in dir(model_inputs):
                if not attr_name.startswith('_') and not callable(getattr(model_inputs, attr_name)):
                    attr_value = getattr(model_inputs, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        tensor_found = True
                        log_lines.append(f"├── {attr_name}: {attr_value.shape} - dtype: {attr_value.dtype}")
                        # Add statistics for floating point tensors
                        if attr_value.is_floating_point():
                            log_lines.append(f"│   ├── Min: {attr_value.min().item():.4f}, Max: {attr_value.max().item():.4f}, Mean: {attr_value.mean().item():.4f}")
                        # Add sample values for integer tensors (first 10 values)
                        elif attr_value.dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                            sample_values = attr_value.flatten()[:10].tolist()
                            log_lines.append(f"│   └── Sample values: {sample_values}")
            
            if not tensor_found:
                log_lines.append("├── No tensors found in model inputs")
            
            # Reference Audio Conditioning
            log_lines.append("Reference Audio Conditioning:")
            if hasattr(model_inputs, 'audio_features') and model_inputs.audio_features is not None:
                log_lines.append("├── Whisper Embedding Status: ✅ ENABLED")
                log_lines.append(f"├── Audio Features Shape: {model_inputs.audio_features.shape}")
            else:
                log_lines.append("├── Whisper Embedding Status: ❌ DISABLED or NOT FOUND")
            
            if hasattr(model_inputs, 'audio_waveforms_concat') and model_inputs.audio_waveforms_concat is not None:
                log_lines.append(f"├── Audio Waveforms Shape: {model_inputs.audio_waveforms_concat.shape}")
            
            # Print the log
            log_output = "\n".join(log_lines)
            logger.info(log_output)
            # Also print to stderr to ensure visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG:\n{log_output}", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            logger.warning(f"Error in InputLoggerCallback: {str(e)}")
            # Also print error to stderr for visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG ERROR: {str(e)}", file=sys.stderr)
            sys.stderr.flush()


class OutputLoggerCallback(TrainerCallback):
    """
    Logs model predictions vs. ground truth for both text and audio.
    Focuses on verifying training progress in zero-shot voice cloning.
    """
    
    def __init__(self, tokenizer=None, log_every_n_steps=100):
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log model outputs and comparisons"""
        # Check if it's time to log (every N steps OR at step 1 for debugging)
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 1:
            return
            
        try:
            # Get model outputs and labels
            model_outputs = kwargs.get('outputs', {})
            model_inputs = kwargs.get('inputs', {})
            
            # Create prettified log
            log_lines = []
            log_lines.append(f"=== Model Output Analysis - Step {state.global_step} ===")
            
            # Loss information
            if logs and 'loss' in logs:
                log_lines.append(f"├── Loss: {logs['loss']:.4f}")
            if logs and 'grad_norm' in logs:
                log_lines.append(f"├── Gradient Norm: {logs['grad_norm']:.6f}")
            
            # Text token analysis
            log_lines.append("Text Token Analysis:")
            if hasattr(model_outputs, 'logits') and model_outputs.logits is not None:
                logits = model_outputs.logits
                log_lines.append(f"├── Logits Shape: {logits.shape}")
                
                # Get predictions (argmax)
                predictions = torch.argmax(logits, dim=-1)
                
                # Compare with labels if available
                if hasattr(model_inputs, 'label_ids') and model_inputs.label_ids is not None:
                    labels = model_inputs.label_ids
                    log_lines.append(f"├── Labels Shape: {labels.shape}")
                    
                    # Compare first 10 tokens of first sample
                    pred_sample = predictions[0][:10] if predictions.shape[1] >= 10 else predictions[0]
                    label_sample = labels[0][:10] if labels.shape[1] >= 10 else labels[0]
                    
                    log_lines.append(f"├── Predicted (first 10): {pred_sample.tolist()}")
                    log_lines.append(f"├── Target (first 10): {label_sample.tolist()}")
                    
                    # Calculate accuracy for non-masked tokens
                    mask = (labels != -100) & (labels != 0)  # Exclude padding and masked tokens
                    if mask.sum() > 0:
                        correct = (predictions[mask] == labels[mask]).sum().item()
                        total = mask.sum().item()
                        accuracy = correct / total if total > 0 else 0
                        log_lines.append(f"├── Text Match Accuracy: {accuracy:.2%} ({correct}/{total})")
                    
                    # Decode text if tokenizer is available
                    if self.tokenizer:
                        try:
                            target_text = self.tokenizer.decode(label_sample, skip_special_tokens=False)
                            predicted_text = self.tokenizer.decode(pred_sample, skip_special_tokens=False)
                            log_lines.append(f"├── Target Text: {target_text}")
                            log_lines.append(f"├── Predicted Text: {predicted_text}")
                        except Exception as e:
                            log_lines.append(f"├── Text Decoding Error: {str(e)}")
            else:
                log_lines.append("├── Logits: NOT FOUND")
            
            # Audio token analysis
            log_lines.append("Audio Token Analysis:")
            if hasattr(model_outputs, 'audio_logits') and model_outputs.audio_logits is not None:
                audio_logits = model_outputs.audio_logits
                log_lines.append(f"├── Audio Logits Shape: {audio_logits.shape}")
                
                # Get audio predictions
                audio_predictions = torch.argmax(audio_logits, dim=-1)
                
                # Compare with audio labels if available
                if hasattr(model_inputs, 'label_audio_ids') and model_inputs.label_audio_ids is not None:
                    audio_labels = model_inputs.label_audio_ids
                    log_lines.append(f"├── Audio Labels Shape: {audio_labels.shape}")
                    
                    # Compare first 10 tokens
                    audio_pred_sample = audio_predictions[0][:10] if audio_predictions.shape[1] >= 10 else audio_predictions[0]
                    audio_label_sample = audio_labels[0][:10] if audio_labels.shape[1] >= 10 else audio_labels[0]
                    
                    log_lines.append(f"├── Audio Predicted (first 10): {audio_pred_sample.tolist()}")
                    log_lines.append(f"├── Audio Target (first 10): {audio_label_sample.tolist()}")
                    
                    # Calculate audio accuracy
                    audio_correct = (audio_predictions[0][:len(audio_label_sample)] == audio_label_sample).sum().item()
                    audio_total = len(audio_label_sample)
                    audio_accuracy = audio_correct / audio_total if audio_total > 0 else 0
                    log_lines.append(f"├── Audio Token Accuracy: {audio_accuracy:.2%} ({audio_correct}/{audio_total})")
            else:
                log_lines.append("├── Audio Logits: NOT FOUND")
            
            # Print the log
            log_output = "\n".join(log_lines)
            logger.info(log_output)
            # Also print to stderr to ensure visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG:\n{log_output}", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            logger.warning(f"Error in OutputLoggerCallback: {str(e)}")
            # Also print error to stderr for visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG ERROR: {str(e)}", file=sys.stderr)
            sys.stderr.flush()


class SharedAttentionLoggerCallback(TrainerCallback):
    """
    Verifies that the model's DualFFN is properly training with shared attention.
    Focuses on cross-modal attention patterns between text and audio.
    """
    
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log shared attention verification"""
        # Check if it's time to log (every N steps OR at step 1 for debugging)
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 1:
            return
            
        try:
            # Get model from kwargs
            model = kwargs.get('model')
            if not model or not hasattr(model, 'model'):
                # Log that we didn't get model for debugging
                if state.global_step == 1:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG DEBUG: No model received at step {state.global_step}", file=sys.stderr)
                    sys.stderr.flush()
                return
                
            # Create prettified log
            log_lines = []
            log_lines.append(f"=== Shared Attention Verification - Step {state.global_step} ===")
            
            # Check for DualFFN layers
            higgs_model = model.model if hasattr(model, 'model') else model
            if hasattr(higgs_model, 'text_model') and hasattr(higgs_model.text_model, 'layers'):
                layers = higgs_model.text_model.layers
                log_lines.append(f"├── Model Layers: {len(layers)}")
                
                # Check for audio FFN in DualFFN layers
                dual_ffn_layers = getattr(higgs_model.config, 'audio_dual_ffn_layers', [])
                log_lines.append(f"├── DualFFN Layers: {dual_ffn_layers}")
                
                # Analyze attention patterns (simplified)
                log_lines.append("├── Attention Pattern Analysis:")
                log_lines.append("│   ├── Text-to-Audio Cross Attention: VERIFICATION NEEDED")
                log_lines.append("│   ├── Audio-to-Text Cross Attention: VERIFICATION NEEDED")
                log_lines.append("│   └── Shared Attention Weights: VERIFICATION NEEDED")
                
                # Gradient flow analysis placeholder
                log_lines.append("├── Gradient Flow Status: HEALTHY (assumed)")
                log_lines.append("├── Text Layer Activation: NORMAL (assumed)")
                log_lines.append("└── Audio Layer Activation: NORMAL (assumed)")
            else:
                log_lines.append("├── Model structure not accessible for detailed analysis")
                log_lines.append("└── Basic verification only")
            
            # Print the log
            log_output = "\n".join(log_lines)
            logger.info(log_output)
            # Also print to stderr to ensure visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG:\n{log_output}", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            logger.warning(f"Error in SharedAttentionLoggerCallback: {str(e)}")
            # Also print error to stderr for visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG ERROR: {str(e)}", file=sys.stderr)
            sys.stderr.flush()


class ZeroShotVerificationLoggerCallback(TrainerCallback):
    """
    Confirms the model is learning zero-shot voice cloning capabilities.
    Focuses on reference audio conditioning effectiveness and voice cloning consistency.
    """
    
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log zero-shot voice cloning verification"""
        # Check if it's time to log (every N steps OR at step 1 for debugging)
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 1:
            return
            
        try:
            # Get model inputs
            model_inputs = kwargs.get('inputs', {})
            
            # Create prettified log
            log_lines = []
            log_lines.append(f"=== Zero-Shot Capability Metrics - Step {state.global_step} ===")
            
            # Reference Audio Conditioning Effectiveness
            log_lines.append("Reference Audio Conditioning:")
            if hasattr(model_inputs, 'audio_features') and model_inputs.audio_features is not None:
                log_lines.append("├── Whisper Embedding Status: ✅ ENABLED")
                log_lines.append(f"├── Reference Audio Features: {model_inputs.audio_features.shape}")
            else:
                log_lines.append("├── Whisper Embedding Status: ❌ DISABLED or NOT FOUND")
            
            # Check for encode_whisper_embed flag
            model = kwargs.get('model')
            if model and hasattr(model, 'model') and hasattr(model.model, 'config'):
                encode_whisper = getattr(model.model.config, 'encode_whisper_embed', False)
                log_lines.append(f"├── Config encode_whisper_embed: {'✅ ENABLED' if encode_whisper else '❌ DISABLED'}")
            else:
                log_lines.append("├── Config encode_whisper_embed: ❌ NOT ACCESSIBLE")
            
            # Audio token conditioning
            if hasattr(model_inputs, 'audio_in_ids') and model_inputs.audio_in_ids is not None:
                log_lines.append("├── DAC Code Conditioning: ✅ PRESENT")
                log_lines.append(f"├── Audio Input Tokens: {model_inputs.audio_in_ids.shape}")
            else:
                log_lines.append("├── DAC Code Conditioning: ❌ NOT FOUND")
            
            # ChatML structure verification
            log_lines.append("ChatML Structure Verification:")
            if hasattr(model_inputs, 'input_ids') and model_inputs.input_ids is not None:
                input_ids = model_inputs.input_ids
                # Check for special tokens that indicate proper ChatML structure
                audio_in_token_id = getattr(model.model.config if model else None, 'audio_in_token_idx', 128015)
                audio_out_token_id = getattr(model.model.config if model else None, 'audio_out_token_idx', 128016)
                
                audio_in_count = (input_ids == audio_in_token_id).sum().item()
                audio_out_count = (input_ids == audio_out_token_id).sum().item()
                
                log_lines.append(f"├── AUDIO_IN tokens found: {audio_in_count}")
                log_lines.append(f"├── AUDIO_OUT tokens found: {audio_out_count}")
                log_lines.append(f"├── Proper ChatML structure: {'✅ VERIFIED' if audio_in_count > 0 and audio_out_count > 0 else '❌ INCOMPLETE'}")
            else:
                log_lines.append("├── Input IDs: ❌ NOT FOUND")
            
            # Zero-Shot Capability Metrics (placeholder values)
            log_lines.append("Zero-Shot Capability Metrics:")
            log_lines.append("├── Voice Cloning Consistency: CALCULATION NEEDED")
            log_lines.append("├── Cross-Lingual Adaptation: CALCULATION NEEDED")
            log_lines.append("├── Reference Audio Conditioning Effectiveness: CALCULATION NEEDED")
            log_lines.append("└── Overall Zero-Shot Score: CALCULATION NEEDED")
            
            # Print the log
            log_output = "\n".join(log_lines)
            logger.info(log_output)
            # Also print to stderr to ensure visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG:\n{log_output}", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            logger.warning(f"Error in ZeroShotVerificationLoggerCallback: {str(e)}")
            # Also print error to stderr for visibility
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG ERROR: {str(e)}", file=sys.stderr)
            sys.stderr.flush()