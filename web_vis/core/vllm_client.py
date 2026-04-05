# web_vis/core/vllm_client.py
"""
Import vLLM client from main core module.
"""
import sys
from pathlib import Path

# Add parent core to path
parent_core = Path(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(parent_core))

from vllm_client import *  # noqa
