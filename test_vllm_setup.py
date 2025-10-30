"""
Quick test script to verify vLLM integration
Run this after starting the vLLM server to ensure everything works
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vllm_connection():
    """Test basic connection to vLLM server"""
    print("=" * 70)
    print("Testing vLLM Connection")
    print("=" * 70)
    
    try:
        from core.vllm_client import safe_call_llm, VLLM_BASE_URL, VLLM_MODEL_NAME
        
        print(f"✓ vLLM client imported successfully")
        print(f"  Base URL: {VLLM_BASE_URL}")
        print(f"  Model: {VLLM_MODEL_NAME}")
        print()
        
        # Test simple prompt
        print("Sending test prompt: 'Hello! Please respond with a short greeting.'")
        print("-" * 70)
        
        response = safe_call_llm("Hello! Please respond with a short greeting.")
        
        print()
        print("✓ SUCCESS! Received response:")
        print(f"  {response}")
        print()
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        print("Make sure vLLM server is running:")
        print("  python start_vllm_server.py")
        print()
        return False


def test_agents_integration():
    """Test that agents can load vLLM client"""
    print("=" * 70)
    print("Testing Agents Integration")
    print("=" * 70)
    
    try:
        from core.api_config import USE_LOCAL_VLLM
        print(f"✓ USE_LOCAL_VLLM = {USE_LOCAL_VLLM}")
        
        if not USE_LOCAL_VLLM:
            print("\n⚠ WARNING: USE_LOCAL_VLLM is False in api_config.py")
            print("  Set it to True to use vLLM")
            print()
            return False
        
        from core import agents
        print(f"✓ Agents module imported successfully")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return False


def test_chat_manager():
    """Test ChatManager initialization"""
    print("=" * 70)
    print("Testing ChatManager")
    print("=" * 70)
    
    try:
        # Create a minimal test
        print("Initializing ChatManager...")
        
        # We'll just test the import and basic setup
        from core.chat_manager import ChatManager, LLM_API_FUC
        
        print(f"✓ ChatManager imported successfully")
        print(f"✓ LLM_API_FUC assigned: {LLM_API_FUC.__module__}.{LLM_API_FUC.__name__}")
        print()
        
        # Test network ping with a simple prompt
        print("Testing network availability...")
        try:
            response = LLM_API_FUC("Test")
            print("✓ Network test passed!")
            print()
            return True
        except Exception as e:
            print(f"✗ Network test failed: {e}")
            return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return False


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "vLLM Integration Test Suite" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    results = []
    
    # Test 1: vLLM Connection
    results.append(("vLLM Connection", test_vllm_connection()))
    
    # Test 2: Agents Integration
    results.append(("Agents Integration", test_agents_integration()))
    
    # Test 3: ChatManager
    results.append(("ChatManager", test_chat_manager()))
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("🎉 All tests passed! Your vLLM integration is working correctly.")
        print()
        print("Next steps:")
        print("  1. Run your main application: python run_evaluate.py")
        print("  2. Or start the web app: python web_vis/app.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print()
        print("Common issues:")
        print("  1. vLLM server not running - Start it with: python start_vllm_server.py")
        print("  2. Wrong configuration - Check USE_LOCAL_VLLM in api_config.py")
        print("  3. Connection issues - Verify server is at http://localhost:8000")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
