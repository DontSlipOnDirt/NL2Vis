"""
vLLM Server Startup Script
This script starts a vLLM OpenAI-compatible API server for local LLM inference
"""

import argparse
import subprocess
import sys

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 8192  # Context length

# Model recommendations for NL2Vis
RECOMMENDED_MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2-7B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server for NL2Vis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default Llama 3.1 8B model
  python start_vllm_server.py
  
  # Start with a specific model
  python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.3
  
  # Use a shorthand alias
  python start_vllm_server.py --model-alias mistral-7b
  
  # Adjust GPU memory and context length
  python start_vllm_server.py --gpu-memory-utilization 0.9 --max-model-len 16384
  
Recommended models:
  - llama-3.1-8b: Best overall performance (default)
  - mistral-7b: Good alternative, slightly faster
  - qwen-7b: Good for multilingual tasks
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Full model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        choices=list(RECOMMENDED_MODELS.keys()),
        help="Use a shorthand alias for recommended models"
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    
    # Performance tuning
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (0.0-1.0, default: 0.90)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum model context length (default: {DEFAULT_MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)"
    )
    
    # Advanced options
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Enable quantization for memory efficiency"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Resolve model name
    if args.model_alias:
        model_name = RECOMMENDED_MODELS[args.model_alias]
        print(f"Using model alias '{args.model_alias}' -> {model_name}")
    else:
        model_name = args.model
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    
    print("=" * 70)
    print("Starting vLLM OpenAI-compatible API Server")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Max Context Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization * 100}%")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    print("=" * 70)
    print("\nStarting server... (This may take a few minutes to load the model)")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down vLLM server...")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting vLLM server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
