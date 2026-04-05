#!/usr/bin/env python
"""
Comprehensive test suite for vLLM integration with NL2Vis.

Usage:
    python tests/test_vllm_suite.py --all              # Run all tests
    python tests/test_vllm_suite.py --setup            # Check setup/configuration
    python tests/test_vllm_suite.py --server           # Test vLLM server
    python tests/test_vllm_suite.py --integration      # Test NL2Vis integration
    python tests/test_vllm_suite.py --monitor          # Monitor server continuously
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# Setup & Configuration Tests
# ============================================================================

def print_header(title, width=60):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def check_imports():
    """Check if required modules can be imported."""
    print_header("Checking Python Packages")
    
    modules = {
        "vllm": "vLLM inference engine",
        "openai": "OpenAI client library",
        "torch": "PyTorch",
        "ray": "Ray (required by vLLM)",
        "requests": "HTTP library",
        "core.config": "central configuration",
        "core.vllm_client": "vLLM client wrapper"
    }
    
    results = {}
    for module_name, description in modules.items():
        try:
            __import__(module_name)
            results[module_name] = True
            print(f"✅ {module_name:20s} - {description}")
        except ImportError as e:
            results[module_name] = False
            print(f"❌ {module_name:20s} - {description} ({e})")
    
    return all(results.values())


def check_config():
    """Check vLLM configuration."""
    print_header("Checking vLLM Configuration")
    
    try:
        from core.config import (
            USE_VLLM,
            VLLM_MODEL_NAME,
            VLLM_HOST,
            VLLM_PORT,
            VLLM_MAX_MODEL_LEN,
            VLLM_MAX_TOKENS,
            VLLM_GPU_MEMORY_UTILIZATION,
            VLLM_TEMPERATURE
        )
        
        print(f"Backend Mode:     {'vLLM (Local)' if USE_VLLM else 'Azure OpenAI'}")
        print(f"Model:            {VLLM_MODEL_NAME}")
        print(f"Server:           {VLLM_HOST}:{VLLM_PORT}")
        print(f"Max Model Length: {VLLM_MAX_MODEL_LEN}")
        print(f"Max Tokens:       {VLLM_MAX_TOKENS}")
        print(f"GPU Memory:       {VLLM_GPU_MEMORY_UTILIZATION}")
        print(f"Temperature:      {VLLM_TEMPERATURE}")
        
        if not USE_VLLM:
            print("\n⚠️  vLLM is DISABLED - System will use Azure OpenAI API")
            print("   Set USE_VLLM=True in core/config.py to enable")
        else:
            print("\n✅ vLLM is ENABLED")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def check_gpu():
    """Check GPU availability and properties."""
    print_header("Checking GPU Resources")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is NOT available")
            print("   vLLM requires a CUDA-capable GPU")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available")
        print(f"   GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"\n   GPU {i}: {gpu_name}")
            print(f"   Total Memory:     {total_memory:.2f} GB")
            print(f"   Allocated:        {allocated:.2f} GB")
            print(f"   Reserved:         {reserved:.2f} GB")
            print(f"   Compute Cap:      {props.major}.{props.minor}")
        
        return True
            
    except Exception as e:
        print(f"❌ GPU check error: {e}")
        return False


# ============================================================================
# Server Tests
# ============================================================================

def check_server():
    """Check if vLLM server is running and responding."""
    print_header("Checking vLLM Server Status")
    
    try:
        import requests
        from core.config import VLLM_HOST, VLLM_PORT
        
        url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
        
        print(f"Connecting to: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Server is RUNNING")
            
            models = response.json()
            if 'data' in models:
                print(f"\nAvailable models:")
                for model in models['data']:
                    print(f"  - {model.get('id', 'unknown')}")
            else:
                print(f"Response: {models}")
            
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        if "ConnectionError" in str(type(e).__name__):
            print(f"❌ Server is NOT running")
            print(f"\nTo start the server:")
            print(f"  Windows: start_vllm.bat")
            print(f"  Linux:   ./start_vllm.sh")
            print(f"  Manual:  python core/vllm_server.py")
        else:
            print(f"❌ Server check error: {e}")
        return False


def test_server_basic():
    """Test basic vLLM server functionality."""
    print_header("Testing vLLM Server - Basic Response")
    
    try:
        from openai import OpenAI
        from core.config import VLLM_MODEL_NAME, VLLM_HOST, VLLM_PORT
        
        client = OpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="dummy"
        )
        
        print("Sending test prompt: 'What is 2+2? Answer briefly.'")
        
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            temperature=0.0,
            max_tokens=50
        )
        
        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        print(f"\n✅ Response received:")
        print(f"   {content}")
        print(f"\nToken usage:")
        print(f"   Prompt:     {prompt_tokens}")
        print(f"   Completion: {completion_tokens}")
        print(f"   Total:      {prompt_tokens + completion_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False


def test_server_nl2vis():
    """Test vLLM server with NL2VIS-relevant query."""
    print_header("Testing vLLM Server - SQL Generation")
    
    try:
        from openai import OpenAI
        from core.config import VLLM_MODEL_NAME, VLLM_HOST, VLLM_PORT
        
        client = OpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="dummy"
        )
        
        sql_prompt = """Given this table schema:
Table: Students (id, name, age, grade)

Write a SQL query to find students older than 18."""
        
        print("Testing SQL generation capability...")
        
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        
        print(f"\n✅ SQL Generation Response:")
        print("-" * 60)
        print(content)
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ SQL test failed: {e}")
        return False


# ============================================================================
# Integration Tests
# ============================================================================

def test_vllm_client():
    """Test vLLM client wrapper."""
    print_header("Testing vLLM Client Integration")
    
    try:
        from core.config import USE_VLLM
        
        if not USE_VLLM:
            print("⚠️  vLLM is disabled, skipping client test")
            return None
        
        from core.vllm_client import safe_call_llm
        
        print("Testing safe_call_llm()...")
        
        response = safe_call_llm("Hello! Please respond with a short greeting.")
        
        print(f"\n✅ Client wrapper working!")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Client test failed: {e}")
        print("\nMake sure:")
        print("  1. USE_VLLM=True in core/config.py")
        print("  2. vLLM server is running")
        return False


def test_agent_integration():
    """Test agent integration with vLLM."""
    print_header("Testing Agent Integration")
    
    try:
        from core.config import USE_VLLM
        
        if not USE_VLLM:
            print("⚠️  vLLM is disabled, skipping agent test")
            return None
        
        from core.agents import LLM_API_FUC
        
        if LLM_API_FUC is None:
            print("❌ LLM_API_FUC is None")
            return False
        
        print(f"LLM_API_FUC: {LLM_API_FUC.__module__}.{LLM_API_FUC.__name__}")
        
        # Check if using vLLM
        if "vllm" in LLM_API_FUC.__module__:
            print("✅ Using vLLM client")
        else:
            print("⚠️  Using Azure OpenAI API (not vLLM)")
        
        # Test the function
        print("\nTesting agent API call...")
        response = LLM_API_FUC("What is data visualization? Answer in one sentence.")
        
        print(f"\n✅ Agent integration working!")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False


# ============================================================================
# Monitoring
# ============================================================================

def monitor_server(interval=30):
    """Monitor vLLM server continuously."""
    print_header("vLLM Server Monitor")
    print(f"Monitoring interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        import requests
        from core.config import VLLM_HOST, VLLM_PORT
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                response = requests.get(
                    f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models",
                    timeout=5
                )
                status = "✅ ONLINE" if response.status_code == 200 else f"⚠️  Status {response.status_code}"
            except:
                status = "❌ OFFLINE"
            
            print(f"[{timestamp}] {status}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 60)
    print("vLLM Integration Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Setup tests
    results["Imports"] = check_imports()
    results["Configuration"] = check_config()
    results["GPU"] = check_gpu()
    
    # Server tests
    results["Server Status"] = check_server()
    if results["Server Status"]:
        results["Basic Response"] = test_server_basic()
        results["SQL Generation"] = test_server_nl2vis()
    
    # Integration tests
    results["vLLM Client"] = test_vllm_client()
    results["Agent Integration"] = test_agent_integration()
    
    # Summary
    print_header("Test Summary")
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{status:10s} - {test_name}")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    failed = [name for name, result in results.items() if result is False]
    skipped = [name for name, result in results.items() if result is None]
    
    if not failed and not skipped:
        print("✅ All tests passed! vLLM is fully operational.")
        print("\nYou can now:")
        print("  - Run evaluation: python run_evaluate.py")
        print("  - Test specific queries: python main.py")
    elif failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        print("\nTroubleshooting:")
        if "Server Status" in failed:
            print("  1. Start vLLM server: python core/vllm_server.py")
        if "GPU" in failed:
            print("  2. Check CUDA installation")
        if "Imports" in failed:
            print("  3. Install packages: pip install vllm ray openai")
    
    if skipped:
        print(f"\n⚠️  {len(skipped)} test(s) skipped: {', '.join(skipped)}")
        print("   Enable vLLM in core/config.py to run all tests")
    
    print("=" * 60 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test suite for vLLM integration with NL2Vis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_vllm_suite.py --all          # Run all tests
    python tests/test_vllm_suite.py --setup        # Check setup only
    python tests/test_vllm_suite.py --server       # Test server only
    python tests/test_vllm_suite.py --integration  # Test integration only
    python tests/test_vllm_suite.py --monitor      # Monitor server
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--setup', action='store_true', help='Check setup/configuration')
    parser.add_argument('--server', action='store_true', help='Test vLLM server')
    parser.add_argument('--integration', action='store_true', help='Test NL2Vis integration')
    parser.add_argument('--monitor', action='store_true', help='Monitor server continuously')
    parser.add_argument('--interval', type=int, default=30, help='Monitor interval in seconds')
    
    args = parser.parse_args()
    
    # Default to --all if no args
    if not any([args.all, args.setup, args.server, args.integration, args.monitor]):
        args.all = True
    
    try:
        if args.all:
            run_all_tests()
        else:
            if args.setup:
                check_imports()
                check_config()
                check_gpu()
            
            if args.server:
                check_server()
                test_server_basic()
                test_server_nl2vis()
            
            if args.integration:
                test_vllm_client()
                test_agent_integration()
            
            if args.monitor:
                monitor_server(args.interval)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
