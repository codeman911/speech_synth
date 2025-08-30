#!/usr/bin/env python3
"""
Test script to verify logging visibility
"""

import sys
import os
from datetime import datetime

# Add the trainer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

def test_logging_visibility():
    """Test that logging is visible"""
    print("Testing logging visibility...")
    
    # Simulate what our callbacks will do
    log_lines = [
        "=== Zero-Shot Voice Cloning Training Log - Step 100 ===",
        "Timestamp: 2023-08-15 14:32:45",
        "",
        "Input Sequence Analysis:",
        "├── Tokenized Input Shape: torch.Size([4, 128])",
        "├── Decoded Text (first sample): Generate speech in the provided voice. Hello, how are you today?<|AUDIO_IN|><|AUDIO_OUT|>",
        "├── Sample input_ids (first 20): [1, 32006, 15875, 263, 304, 369, 264, 1700, 369, 32007, 25, 11770, 13, 7214, 318, 428, 318, 1937, 296, 1475]",
        "├── Audio Tokens: 1 AUDIO_IN tokens, 1 AUDIO_OUT tokens",
        "└── Sequence Length: 128 tokens"
    ]
    
    log_output = "\n".join(log_lines)
    
    # This is what our callbacks now do
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRATEGIC LOG:\n{log_output}", file=sys.stderr)
    sys.stderr.flush()
    
    print("Test completed. You should see the log output above.")

if __name__ == "__main__":
    test_logging_visibility()