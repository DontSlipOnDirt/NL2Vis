# core/vision_vllm_server.py
"""
Script to start vision vLLM server for multimodal inference.
"""

import subprocess
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    VISION_VLLM_MODEL_NAME,
    VISION_VLLM_HOST,
    VISION_VLLM_PORT,
    VISION_VLLM_MAX_MODEL_LEN,
    VISION_VLLM_GPU_MEMORY_UTILIZATION,
    VISION_VLLM_QUANTIZATION,
    VISION_VLLM_DTYPE
)


def start_vision_vllm_server():
    """Start vLLM OpenAI-compatible server for vision models."""
    
    print("=" * 60)
    print("Starting Vision vLLM Server")
    print("=" * 60)
    print(f"Model: {VISION_VLLM_MODEL_NAME}")
    print(f"Host: {VISION_VLLM_HOST}")
    print(f"Port: {VISION_VLLM_PORT}")
    print(f"Max Model Length: {VISION_VLLM_MAX_MODEL_LEN}")
    print(f"GPU Memory Utilization: {VISION_VLLM_GPU_MEMORY_UTILIZATION}")
    print(f"Quantization: {VISION_VLLM_QUANTIZATION}")
    print(f"Dtype: {VISION_VLLM_DTYPE}")
    print("=" * 60)
    
    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VISION_VLLM_MODEL_NAME,
        "--host", VISION_VLLM_HOST,
        "--port", str(VISION_VLLM_PORT),
        "--max-model-len", str(VISION_VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(VISION_VLLM_GPU_MEMORY_UTILIZATION),
        "--dtype", VISION_VLLM_DTYPE,
        "--trust-remote-code"  # Required for vision models
    ]
    
    # Add quantization if specified
    if VISION_VLLM_QUANTIZATION:
        cmd.extend(["--quantization", VISION_VLLM_QUANTIZATION])
    
    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    print("Server is starting... (this may take a few minutes)")
    print("⚠️  Vision models require image input support")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nVision server stopped by user.")
    except Exception as e:
        print(f"\n\nError starting vision server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_vision_vllm_server()
