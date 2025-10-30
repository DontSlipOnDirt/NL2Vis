# Quick Start Guide - vLLM Local LLMs

## ✅ What Was Done

Your NL2Vis system has been configured to use **vLLM** for local LLM inference instead of Azure OpenAI API.

### Files Created:
1. ✅ `core/vllm_client.py` - vLLM client module
2. ✅ `web_vis/core/vllm_client.py` - vLLM client for web app
3. ✅ `start_vllm_server.py` - Python server launcher
4. ✅ `start_vllm_server.ps1` - PowerShell server launcher (Windows)
5. ✅ `test_vllm_setup.py` - Integration test suite
6. ✅ `VLLM_SETUP.md` - Complete documentation

### Files Modified:
1. ✅ `core/api_config.py` - Added vLLM configuration
2. ✅ `web_vis/core/api_config.py` - Added vLLM configuration
3. ✅ `core/agents.py` - Updated to use vLLM
4. ✅ `web_vis/core/agents.py` - Updated to use vLLM
5. ✅ `core/chat_manager.py` - Updated to use vLLM
6. ✅ `web_vis/core/chat_manager.py` - Updated to use vLLM

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start vLLM Server

**Windows (PowerShell):**
```powershell
.\start_vllm_server.ps1
```

**Or using Python:**
```bash
python start_vllm_server.py
```

**Wait for this message:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Test the Setup (New Terminal)

```bash
python test_vllm_setup.py
```

You should see:
```
✓ PASS - vLLM Connection
✓ PASS - Agents Integration
✓ PASS - ChatManager
🎉 All tests passed!
```

### Step 3: Run Your Application

```bash
# For evaluation
python run_evaluate.py

# For web interface
python web_vis/app.py
```

---

## ⚙️ Configuration

Edit `core/api_config.py` (and `web_vis/core/api_config.py`):

```python
# Toggle between Azure OpenAI and vLLM
USE_LOCAL_VLLM = True  # True = vLLM, False = Azure

# vLLM settings (only used when USE_LOCAL_VLLM = True)
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
```

### Change Model:

```python
# In api_config.py
VLLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Faster
# or
VLLM_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Multilingual
```

Then restart the vLLM server.

---

## 🔧 Common Commands

### Start Server with Different Model
```bash
python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.3
```

### Use Shorthand Alias
```bash
python start_vllm_server.py --model-alias mistral-7b
```

### Reduce Memory Usage
```bash
python start_vllm_server.py --gpu-memory-utilization 0.7 --max-model-len 4096
```

### Check Server Status
```powershell
# In PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/v1/models" | ConvertFrom-Json
```

---

## 🐛 Troubleshooting

### Problem: "Connection refused"
**Solution:** Start the vLLM server first!
```bash
python start_vllm_server.py
```

### Problem: "CUDA out of memory"
**Solutions:**
1. Reduce memory usage:
   ```bash
   python start_vllm_server.py --gpu-memory-utilization 0.7
   ```
2. Reduce context length:
   ```bash
   python start_vllm_server.py --max-model-len 4096
   ```

### Problem: Slow responses
**Solution:** Models need time to load first time. Subsequent requests are much faster.

### Problem: Model download fails
**Solution:** Some models (like Llama) need HuggingFace login:
```bash
huggingface-cli login
```

---

## 📊 Expected Performance

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| Llama 3.1 8B | ~16GB | 20-50 tok/s | ⭐⭐⭐⭐⭐ |
| Mistral 7B | ~14GB | 30-60 tok/s | ⭐⭐⭐⭐ |
| Qwen 2 7B | ~14GB | 25-55 tok/s | ⭐⭐⭐⭐ |

*Speed depends on your GPU (tested on RTX 3090/4090)*

---

## 🔄 Switching Back to Azure OpenAI

1. Edit `core/api_config.py`:
   ```python
   USE_LOCAL_VLLM = False
   ```

2. Restart your application (no need to stop vLLM server)

---

## 📚 More Information

See `VLLM_SETUP.md` for:
- Detailed configuration options
- Advanced optimization techniques
- Multi-GPU setup
- Quantization options
- Complete troubleshooting guide

---

## ✨ Benefits of vLLM

- ✅ **Free**: No API costs
- ✅ **Fast**: 10-20x faster than transformers
- ✅ **Private**: All data stays local
- ✅ **Offline**: Works without internet
- ✅ **Compatible**: Drop-in replacement for OpenAI API

---

**Need Help?**
1. Check `VLLM_SETUP.md` for detailed docs
2. Run `python test_vllm_setup.py` to diagnose issues
3. Verify vLLM docs: https://docs.vllm.ai/
