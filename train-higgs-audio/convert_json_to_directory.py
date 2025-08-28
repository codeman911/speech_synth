#!/usr/bin/env python3
"""
Convert ChatML JSON file to directory structure for DDP training
"""

import os
import json
import argparse
import shutil
from pathlib import Path

def convert_json_to_directory(json_file_path, output_dir):
    """
    Convert a ChatML JSON file to directory structure for DDP training
    
    Args:
        json_file_path (str): Path to the ChatML JSON file
        output_dir (str): Output directory path
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single sample and list of samples
    samples = data if isinstance(data, list) else [data]
    
    # Create metadata list
    metadata_samples = []
    
    # Process each sample
    for i, sample in enumerate(samples):
        # Create sample directory
        sample_dir = output_path / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Extract messages
        messages = sample.get("messages", [])
        
        # Find reference audio and target audio
        ref_audio_path = None
        target_audio_path = None
        ref_text = None
        target_text = None
        
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "audio":
                        if role == "user" and ref_audio_path is None:
                            ref_audio_path = item.get("audio_url")
                        elif role == "assistant" and target_audio_path is None:
                            target_audio_path = item.get("audio_url")
                    elif item.get("type") == "text":
                        if role == "user":
                            if ref_text is None:
                                ref_text = item.get("text")
                            else:
                                target_text = item.get("text")
        
        # Copy audio files if they exist
        if ref_audio_path and os.path.exists(ref_audio_path):
            ref_dest = sample_dir / f"ref_audio{Path(ref_audio_path).suffix}"
            shutil.copy2(ref_audio_path, ref_dest)
            
        if target_audio_path and os.path.exists(target_audio_path):
            target_dest = sample_dir / f"target_audio{Path(target_audio_path).suffix}"
            shutil.copy2(target_audio_path, target_dest)
        
        # Create transcript file
        transcript_content = target_text or "No transcript available"
        transcript_file = sample_dir / "transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        # Add to metadata
        sample_metadata = {
            "audio_file": str(target_dest.relative_to(output_path)) if target_audio_path else "target_audio.wav",
            "transcript_file": str(transcript_file.relative_to(output_path)),
            "audio_id": f"sample_{i:04d}"
        }
        
        if ref_audio_path:
            sample_metadata["ref_audio_file"] = str(ref_dest.relative_to(output_path))
            sample_metadata["ref_transcript"] = ref_text or "Reference transcript"
            
        metadata_samples.append(sample_metadata)
    
    # Create metadata.json
    metadata = {"samples": metadata_samples}
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(samples)} samples to directory structure in {output_dir}")
    print(f"Metadata file created at {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert ChatML JSON to directory structure for DDP training")
    parser.add_argument("json_file", help="Path to the ChatML JSON file")
    parser.add_argument("output_dir", help="Output directory path")
    
    args = parser.parse_args()
    
    convert_json_to_directory(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()