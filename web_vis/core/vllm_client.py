"""
Wrapper for root core/vllm_client.py to avoid namespace conflicts.
Re-exports the vLLM client for web_vis usage.
"""
import sys
from pathlib import Path

# Locate root core directory
_root_core = Path(__file__).parent.parent.parent / "core"

if not _root_core.exists():
    raise ImportError(f"Root core/ directory not found at {_root_core}")

# Temporarily add to path for import
_original_path = sys.path.copy()
sys.path.insert(0, str(_root_core))

try:
    # Import root vllm_client with all its functions
    from vllm_client import (
        safe_call_llm,
        init_log_path,
    )
except ImportError as e:
    sys.path = _original_path
    raise ImportError(f"Failed to import root core/vllm_client: {e}")
finally:
    # Restore original path to avoid side effects
    sys.path = _original_path

# Re-export for web_vis
__all__ = ['safe_call_llm', 'init_log_path']
