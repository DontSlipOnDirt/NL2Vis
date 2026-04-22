# core/vllm_server.py
"""
Script to start vLLM server easily.
"""

import subprocess
import sys
import os

# Add parent directory to path for imports because this is a standalone script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    VLLM_MODEL_NAME,
    VLLM_TOKENIZER,
    VLLM_HOST,
    VLLM_PORT,
    VLLM_MAX_MODEL_LEN,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_QUANTIZATION,
    VLLM_DTYPE
)


def start_vllm_server():
    """Start vLLM OpenAI-compatible server."""
    
    print("=" * 60)
    print("Starting vLLM Server")
    print("=" * 60)
    print(f"Model: {VLLM_MODEL_NAME}")
    print(f"Tokenizer: {VLLM_TOKENIZER if VLLM_TOKENIZER else 'Auto (from model)'}")
    print(f"Host: {VLLM_HOST}")
    print(f"Port: {VLLM_PORT}")
    print(f"Max Model Length: {VLLM_MAX_MODEL_LEN}")
    print(f"GPU Memory Utilization: {VLLM_GPU_MEMORY_UTILIZATION}")
    print(f"Quantization: {VLLM_QUANTIZATION}")
    print(f"Dtype: {VLLM_DTYPE}")
    print("=" * 60)

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
        "--dtype", VLLM_DTYPE
    ]

    # Add tokenizer if specified (required for GGUF models)
    if VLLM_TOKENIZER:
        cmd.extend(["--tokenizer", VLLM_TOKENIZER])

        # For GGUF models, explicitly set load format
    if "GGUF" in VLLM_MODEL_NAME.upper() or ":" in VLLM_MODEL_NAME:
        cmd.extend(["--load-format", "gguf"])
    # Add quantization if specified
    elif VLLM_QUANTIZATION:
        cmd.extend(["--quantization", VLLM_QUANTIZATION])

    # Add quantization if specified
    if VLLM_QUANTIZATION:
        cmd.extend(["--quantization", VLLM_QUANTIZATION])
    
    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    print("Server is starting... (this may take a few minutes)")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_vllm_server()
