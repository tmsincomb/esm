#!/usr/bin/env python3
"""
Fix deepspeed compatibility with newer PyTorch versions
"""

import sys
import os

# Create a mock torch._six module for compatibility
import torch
if not hasattr(torch, '_six'):
    class MockSix:
        inf = float('inf')
    
    sys.modules['torch._six'] = MockSix()
    torch._six = MockSix()

print("Applied deepspeed compatibility fix for newer PyTorch versions")