#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script with LoRA Support
Based on the Higgs Audio v2 architecture from Boson AI
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import re

# 尝试导入 Higgs Audio 相关模块
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False
    logging.warning("Higgs Audio modules not available. Using fallback implementation.")

    # A simple fallback class is sufficient. It does NOT need a .to() method.
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加常量定义
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class ExtendedHiggsAudioBatchInput:
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and self.input_ids is not None:
            return self.input_ids.shape[0]
        else:
            return 0

    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key)

    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)

    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """
    Extended collator that returns our custom batch input class
    """
    def __init__(self, **kwargs):
        if HIGGS_AVAILABLE:
            # Assuming HiggsAudioSampleCollator can be initialized with these kwargs
            self.base_collator = HiggsAudioSampleCollator(**kwargs)
        else:
            # Fallback collator
            self.pad_token_id = kwargs.get('pad_token_id', 0)

    def __call__(self, batch: List[ChatMLDatasetSample]):
        if HIGGS_AVAILABLE and hasattr(self, 'base_collator'):
            batch_input = self.base_collator(batch)
            label_audio_ids = batch_input.audio_out_ids
            extended_batch = ExtendedHiggsAudioBatchInput(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                audio_features=batch_input.audio_features,
                audio_feature_attention_mask=batch_input.audio_feature_attention_mask,
                audio_out_ids=batch_input.audio_out_ids,
                audio_out_ids_start=batch_input.audio_out_ids_start,
                audio_out_ids_start_group_loc=batch_input.audio_out_ids_start_group_loc,
                audio_in_ids=batch_input.audio_in_ids,
                audio_in_ids_start=batch_input.audio_in_ids_start,
                label_ids=batch_input.label_ids,
                label_audio_ids=label_audio_ids,
                reward=batch_input.reward,
            )
            return extended_batch
        else:
            # Fallback implementation
            input_ids_list, attention_mask_list, label_ids_list = [], [], []
            for sample in batch:
                input_ids_list.append(sample.input_ids)
                attention_mask_list.append(torch.ones_like(sample.input_ids))
                label_ids_list.append(getattr(sample, 'label_ids', sample.input_ids))

            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids, padded_attention_mask, padded_label_ids = [], [], []

            for i in range(len(input_ids_list)):
                pad_len = max_len - len(input_ids_list[i])
                padded_input_ids.append(torch.cat([input_ids_list[i], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
                padded_attention_mask.append(torch.cat([attention_mask_list[i], torch.zeros(pad_len, dtype=torch.long)]))
                padded_label_ids.append(torch.cat([label_ids_list[i], torch.full((pad_len,), -100, dtype=torch.long)]))

            return ExtendedHiggsAudioBatchInput(
                input_ids=torch.stack(padded_input_ids),
                attention_mask=torch.stack(padded_attention_mask),
                label_ids=torch.stack(padded_label_ids),
                audio_features=None, audio_feature_attention_mask=None,
                audio_out_ids=None, audio_out_ids_start=None, audio_out_ids_start_group_loc=None,
                audio_in_ids=None, audio_in_ids_start=None, label_audio_ids=None, reward=None,
            )

def normalize_chinese_punctuation(text):
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "——": "-", "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


class HiggsAudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        task_type: str = "zero_shot_voice_cloning",
        sample_rate: int = 24000,
        use_metadata: bool = True,
        ref_audio_in_system_message: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.use_metadata = use_metadata
        self.ref_audio_in_system_message = ref_audio_in_system_message

        valid_tasks = ["zero_shot_voice_cloning", "single_speaker_smart_voice", "multi_speaker_smart_voice", "multi_speaker_voice_cloning"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")

        self.actual_num_codebooks = self._detect_codebook_size()

        if use_metadata and (self.data_dir / "metadata.json").exists():
            self.samples = self._load_samples_from_metadata()
        else:
            logger.warning(f"metadata.json not found in {data_dir}. Scanning directory instead.")
            self.samples = self._scan_data_directory()

    def _detect_codebook_size(self):
        """Detect the actual number of codebooks in the audio tokenizer"""
        if self.audio_tokenizer is None:
            return 8  # Default fallback
        try:
            # Try to get the codebook size from the tokenizer configuration
            if hasattr(self.audio_tokenizer, 'config') and hasattr(self.audio_tokenizer.config, 'codebook_size'):
                return self.audio_tokenizer.config.codebook_size
            # Fallback to default
            return 8
        except Exception as e:
            logger.warning(f"Could not detect codebook size: {e}. Using default of 8.")
            return 8

    def _load_samples_from_metadata(self):
        """Load dataset samples from metadata.json"""
        metadata_path = self.data_dir / "metadata.json"
        logger.info(f"Loading samples from {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        samples = []
        for item in metadata:
            try:
                # Resolve paths relative to data directory
                audio_path = item.get("audio_path")
                if audio_path and not os.path.isabs(audio_path):
                    audio_path = os.path.normpath(self.data_dir / audio_path)
                    
                sample = {
                    "audio_path": audio_path,
                    "text": item.get("text", ""),
                    "speaker": item.get("speaker", "default"),
                    "task_type": item.get("task_type", self.task_type),
                }
                
                # Validate sample
                if self._validate_sample(sample):
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Skipping invalid sample: {e}")
                
        logger.info(f"Loaded {len(samples)} valid samples from metadata")
        return samples

    def _scan_data_directory(self):
        """Scan data directory for audio files and corresponding text files"""
        logger.info(f"Scanning directory {self.data_dir} for audio files")
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.m4a'}
        samples = []
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                try:
                    # Look for corresponding text file
                    text_file = file_path.with_suffix('.txt')
                    if not text_file.exists():
                        # Try alternative naming patterns
                        text_file = file_path.parent / f"{file_path.stem}.txt"
                        
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            
                        sample = {
                            "audio_path": str(file_path),
                            "text": text,
                            "speaker": "default",
                            "task_type": self.task_type,
                        }
                        
                        if self._validate_sample(sample):
                            samples.append(sample)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    
        logger.info(f"Found {len(samples)} valid samples by scanning directory")
        return samples

    def _validate_sample(self, sample):
        """Validate a dataset sample"""
        audio_path = sample.get("audio_path")
        text = sample.get("text", "")
        
        # Check if audio file exists
        if audio_path and not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return False
            
        # Check if text is not empty
        if not text.strip():
            logger.warning(f"Empty text for sample: {audio_path}")
            return False
            
        return True

    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and process audio waveform"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return torch.zeros(1), self.sample_rate

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        """Encode audio to tokens using audio tokenizer"""
        if not self.audio_tokenizer: 
            return None
        try:
            return self.audio_tokenizer.encode(audio_path)
        except Exception as e:
            logger.error(f"Failed to encode audio {audio_path}: {e}")
            return None

    def _prepare_multispeaker_smart_voice_sample(self, sample: Dict) -> ChatMLDatasetSample:
        """Prepare sample for multi-speaker smart voice task"""
        text = sample["text"]
        audio_path = sample["audio_path"]
        speaker = sample["speaker"]
        
        # Create system message with speaker information
        system_message = f"[SPEAKER_{speaker}] {MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}"
        system_content = _build_system_message_with_audio_prompt(system_message)
        
        # Load and encode audio
        waveform, sr = self._load_audio_waveform(audio_path)
        audio_tokens = self._encode_audio_tokens(audio_path)
        
        # Create messages
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]
        
        chatml_sample = ChatMLSample(messages=messages)
        
        # Process with prepare function
        input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(
            chatml_sample, self.tokenizer
        )
        
        # Process audio data
        context_audio_tokens = []
        for audio_content in (audio_contents or []):
            if audio_content.audio_url:
                tokens = self._encode_audio_tokens(audio_content.audio_url)
                if tokens is not None:
                    context_audio_tokens.append(tokens)

        label_audio_tokens = []
        for audio_label_content in (audio_label_contents or []):
            if audio_label_content and audio_label_content.audio_url:
                tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                if tokens is not None:
                    label_audio_tokens.append(tokens)

        # Concatenate tensors
        if context_audio_tokens:
            audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [t.shape[1] for t in context_audio_tokens[:-1]])),
                dtype=torch.long
            )
        else:
            audio_ids_concat = torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([0], dtype=torch.long)

        label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

        return ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            label_audio_ids=label_audio_ids,
            audio_waveforms_concat=waveform,
            audio_waveforms_start=torch.tensor([0], dtype=torch.long),
            audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([0], dtype=torch.long),
        )

    def _prepare_zero_shot_voice_cloning_sample(self, sample: Dict) -> ChatMLDatasetSample:
        """Prepare sample for zero-shot voice cloning task"""
        text = sample["text"]
        audio_path = sample["audio_path"]
        
        # Create system message for zero-shot voice cloning
        system_message = "Generate speech in the provided voice."
        system_content = Message(role="system", content=system_message)
        
        # Load and encode reference audio
        waveform, sr = self._load_audio_waveform(audio_path)
        audio_tokens = self._encode_audio_tokens(audio_path)
        
        # Create messages for zero-shot voice cloning
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]
        
        chatml_sample = ChatMLSample(messages=messages)
        
        # Process with prepare function
        input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(
            chatml_sample, self.tokenizer
        )
        
        # Process audio data
        context_audio_tokens = []
        for audio_content in (audio_contents or []):
            if audio_content.audio_url:
                tokens = self._encode_audio_tokens(audio_content.audio_url)
                if tokens is not None:
                    context_audio_tokens.append(tokens)

        label_audio_tokens = []
        for audio_label_content in (audio_label_contents or []):
            if audio_label_content and audio_label_content.audio_url:
                tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                if tokens is not None:
                    label_audio_tokens.append(tokens)

        # Concatenate tensors
        if context_audio_tokens:
            audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [t.shape[1] for t in context_audio_tokens[:-1]])),
                dtype=torch.long
            )
        else:
            audio_ids_concat = torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([0], dtype=torch.long)

        label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

        return ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            label_audio_ids=label_audio_ids,
            audio_waveforms_concat=waveform,
            audio_waveforms_start=torch.tensor([0], dtype=torch.long),
            audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([0], dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """Get a dataset sample by index"""
        sample = self.samples[idx]
        task_type = sample.get("task_type", self.task_type)
        
        try:
            if task_type == "multi_speaker_smart_voice":
                return self._prepare_multispeaker_smart_voice_sample(sample)
            elif task_type == "zero_shot_voice_cloning":
                return self._prepare_zero_shot_voice_cloning_sample(sample)
            else:
                # Default handling for other task types
                text = sample["text"]
                audio_path = sample["audio_path"]
                
                # Load and encode audio
                waveform, sr = self._load_audio_waveform(audio_path)
                audio_tokens = self._encode_audio_tokens(audio_path)
                
                # Simple message structure for other tasks
                messages = [
                    Message(role="user", content=text),
                    Message(role="assistant", content=AudioContent(audio_url=audio_path))
                ]
                
                chatml_sample = ChatMLSample(messages=messages)
                
                # Process with prepare function
                input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(
                    chatml_sample, self.tokenizer
                )
                
                # Process audio data
                context_audio_tokens = []
                for audio_content in (audio_contents or []):
                    if audio_content.audio_url:
                        tokens = self._encode_audio_tokens(audio_content.audio_url)
                        if tokens is not None:
                            context_audio_tokens.append(tokens)

                label_audio_tokens = []
                for audio_label_content in (audio_label_contents or []):
                    if audio_label_content and audio_label_content.audio_url:
                        tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                        if tokens is not None:
                            label_audio_tokens.append(tokens)

                # Concatenate tensors
                if context_audio_tokens:
                    audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                    audio_ids_start = torch.tensor(
                        np.cumsum(np.array([0] + [t.shape[1] for t in context_audio_tokens[:-1]])),
                        dtype=torch.long
                    )
                else:
                    audio_ids_concat = torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long)
                    audio_ids_start = torch.tensor([0], dtype=torch.long)

                label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

                return ChatMLDatasetSample(
                    input_ids=torch.tensor(input_tokens, dtype=torch.long),
                    label_ids=torch.tensor(label_tokens, dtype=torch.long),
                    audio_ids_concat=audio_ids_concat,
                    audio_ids_start=audio_ids_start,
                    label_audio_ids=label_audio_ids,
                    audio_waveforms_concat=waveform,
                    audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                    audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
                    audio_speaker_indices=torch.tensor([0], dtype=torch.long),
                )
                
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            # Return a simple fallback sample
            return ChatMLDatasetSample(
                input_ids=torch.tensor([0], dtype=torch.long),
                label_ids=torch.tensor([-100], dtype=torch.long),
                audio_ids_concat=torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long),
                audio_ids_start=torch.tensor([0], dtype=torch.long),
                label_audio_ids=None,
                audio_waveforms_concat=torch.zeros(1),
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([24000], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training"""
    def __init__(self, model_path: str, device: str = 'cuda', args=None):
        super().__init__()
        if HIGGS_AVAILABLE:
            self.model = HiggsAudioModel.from_pretrained(
                config=HiggsAudioConfig.from_pretrained(model_path),
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map={'': device},
            )
            self.config = self.model.config
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.config = self.model.config
        self.model = self.model.to(device)
        
        if args:
            if args.freeze_audio_tower: self.model.freeze_audio_tower()
            if args.freeze_audio_encoder_proj: self.model.freeze_audio_encoder_proj()
            if args.freeze_llm: self.model.freeze_llm()

    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        """
        Forward pass with proper type alignment.
        This follows the same pattern as the inference script.
        """
        # --- 开始终极修复 ---
        # 在数据进入底层模型之前，强制将所有浮点张量类型与模型权重对齐。
        # 这是解决顽固类型不匹配问题的最可靠方法。
        model_dtype = next(self.model.parameters()).dtype
        
        for key, value in kwargs.items():
            # 仅转换浮点类型的张量，忽略整数类型（如 input_ids）
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                kwargs[key] = value.to(model_dtype)
        # --- 结束终极修复 ---

        if HIGGS_AVAILABLE:
            return self.model(**kwargs)
        else:
            # Fallback 逻辑也受益于上面的类型转换
            outputs = self.model(input_ids=kwargs.get('input_ids'), attention_mask=kwargs.get('attention_mask'))
            loss = None
            if kwargs.get('label_ids') is not None:
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = kwargs.get('label_ids')[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": outputs.logits}

class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加这行，从model中获取config
        self.config = self.model.config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation"""
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
        
        # 确保所有输入都在相同设备上
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(model.device)
        
        outputs = model(**model_inputs)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation loop that ensures eval_loss is computed and returned
        """
        # Force prediction_loss_only to False to ensure loss is computed
        if prediction_loss_only is None:
            prediction_loss_only = False
            
        # Call the parent evaluation loop
        eval_result = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Ensure eval_loss is in the metrics
        if "eval_loss" not in eval_result.metrics and hasattr(eval_result, 'loss'):
            eval_result.metrics["eval_loss"] = eval_result.loss
            
        return eval_result


def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Setup LoRA configuration for the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        auto_mapping=True
    )
    
    model = model.to(device)
    
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 with LoRA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/root/code/higgs-audio-main/model_ckpt")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/root/code/higgs-audio-main/model_ckpt_tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, default="/root/code/higgs-audio-main/higgs_training_data_huo")
    parser.add_argument("--eval_data_dir", type=str, default="")

    # Task type arguments
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice",
                       choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", 
                               "multi_speaker_smart_voice", "multi_speaker_voice_cloning"],
                       help="Training task type")
    parser.add_argument("--ref_audio_in_system_message", action="store_true", default=False,
                       help="Whether to include reference audio in system message")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/huo_train-vxx")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--disable_evaluation", action="store_true", default=False,
                       help="Disable evaluation during training to avoid checkpoint/evaluation mismatch")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--logging_dir", type=str, default="./logs/huo_train-vxx")
    parser.add_argument("--report_to", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb", "none"])
    
    # Freeze model components
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False)
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False)
    parser.add_argument("--freeze_llm", action="store_true", default=False)


    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load audio tokenizer
    if HIGGS_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
        except Exception as e:
            logger.warning(f"Failed to load audio tokenizer: {e}")
            audio_tokenizer = None
    else:
        audio_tokenizer = None
        logger.warning("Audio tokenizer not available, using fallback")
    
    # Load model
    model = HiggsAudioModelWrapper(args.model_path, device='cuda', args=args)
    
    # Setup LoRA
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        model = setup_lora_config(model, lora_config)
        logger.info("LoRA configuration applied")
    
    # Load datasets
    train_dataset = HiggsAudioDataset(
        args.train_data_dir,
        tokenizer,
        audio_tokenizer,
        task_type=args.task_type,
        ref_audio_in_system_message=args.ref_audio_in_system_message
    )
    
    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = HiggsAudioDataset(
            args.eval_data_dir,
            tokenizer,
            audio_tokenizer,
            task_type=args.task_type,
            ref_audio_in_system_message=args.ref_audio_in_system_message
        )
    
    # Determine if evaluation should be enabled
    evaluation_enabled = eval_dataset is not None and not args.disable_evaluation
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if evaluation_enabled else None,
        evaluation_strategy="steps" if evaluation_enabled else "no",
        save_total_limit=3,
        load_best_model_at_end=evaluation_enabled,  # Only True when evaluation is enabled
        metric_for_best_model="eval_loss" if evaluation_enabled else None,
        fp16=args.fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else None,
        logging_dir=args.logging_dir,
    )

    # Setup data collator
    if HIGGS_AVAILABLE and hasattr(model.config, 'audio_in_token_idx'):
        try:
            from transformers import WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            
            data_collator = ExtendedHiggsAudioSampleCollator(
                whisper_processor=whisper_processor,
                audio_in_token_id=model.config.audio_in_token_idx,
                audio_out_token_id=model.config.audio_out_token_idx,
                audio_stream_bos_id=model.config.audio_stream_bos_id,
                audio_stream_eos_id=model.config.audio_stream_eos_id,
                encode_whisper_embed=True,
                pad_token_id=tokenizer.pad_token_id,
                return_audio_in_tokens=True,
                use_delay_pattern=False,
                round_to=8,
                audio_num_codebooks=8,
            )
        except Exception as e:
            logger.warning(f"Failed to setup Higgs collator: {e}. Using fallback.")
            data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
        logger.warning("Using fallback collator")
    config = AutoConfig.from_pretrained(args.model_path)
    # Initialize trainer
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info(f"Starting training for task: {args.task_type}")
    trainer.train()
    
    # Save the final model
    config.save_pretrained(args.output_dir)
    trainer.save_model()
    logger.info(f"Model checkpoints saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
            model.model.text_model.save_pretrained(lora_output_dir)
        elif hasattr(model, 'model'):
            model.model.save_pretrained(lora_output_dir)
        else:
            model.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters saved separately to {lora_output_dir}")
        logger.info("IMPORTANT: LoRA adapters are saved SEPARATELY from model checkpoints!")
        logger.info("To merge LoRA adapters with base model, use the merger.py script:")
        logger.info(f"  python trainer/merger.py --base_model_path {args.model_path} --lora_adapter_path {lora_output_dir} --output_path ./merged_model")
        logger.info("DO NOT try to use checkpoint directories with merger.py - they don't contain LoRA adapters!")


if __name__ == "__main__":
    main()