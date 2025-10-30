# vLLM Local LLM Setup Guide

This guide explains how to use local LLMs with vLLM instead of Azure OpenAI API.

## 🚀 Quick Start

### 1. Install vLLM (Already Done)
```bash
uv add vllm
```

### 2. Start the vLLM Server

**Option A: Using the helper script (Recommended)**
```bash
python start_vllm_server.py
```

**Option B: Direct command**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

### 3. Configure NL2Vis

Edit `core/api_config.py` and `web_vis/core/api_config.py`:

```python
# Set this to True to use vLLM
USE_LOCAL_VLLM = True

# Configure vLLM connection
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
```

### 4. Run Your Application

The system will automatically use vLLM instead of Azure OpenAI:

```bash
python run_evaluate.py
# or
python web_vis/app.py
```

---

## 📦 Recommended Models

### **Llama 3.1 8B Instruct** (Default, Best Overall)
- **Model ID**: `meta-llama/Llama-3.1-8B-Instruct`
- **VRAM**: ~16GB
- **Best for**: General-purpose, SQL generation, reasoning
- **Start command**:
  ```bash
  python start_vllm_server.py --model-alias llama-3.1-8b
  ```

### **Mistral 7B Instruct** (Faster Alternative)
- **Model ID**: `mistralai/Mistral-7B-Instruct-v0.3`
- **VRAM**: ~14GB
- **Best for**: Faster inference, good reasoning
- **Start command**:
  ```bash
  python start_vllm_server.py --model-alias mistral-7b
  ```

### **Qwen 2 7B Instruct** (Multilingual)
- **Model ID**: `Qwen/Qwen2-7B-Instruct`
- **VRAM**: ~14GB
- **Best for**: Multilingual tasks, coding
- **Start command**:
  ```bash
  python start_vllm_server.py --model-alias qwen-7b
  ```

---

## ⚙️ Advanced Configuration

### Memory Optimization

If you have limited GPU memory, use quantization:

```bash
python start_vllm_server.py --quantization awq --gpu-memory-utilization 0.8
```

### Increase Context Length

For longer prompts (your system uses long prompts):

```bash
python start_vllm_server.py --max-model-len 16384
```

### Multi-GPU Setup

If you have multiple GPUs:

```bash
python start_vllm_server.py --tensor-parallel-size 2
```

---

## 🧪 Testing the Setup

### Test vLLM Client

```bash
python -c "from core.vllm_client import safe_call_llm; print(safe_call_llm('Hello!'))"
```

### Test with ChatManager

```python
from core.chat_manager import ChatManager

manager = ChatManager(data_path="path/to/data", log_path="./test.log")
# If server is running, you'll see: "Network is available"
```

---

## 🔧 Troubleshooting

### Error: "Connection refused" or "Failed to connect"

**Solution**: Make sure the vLLM server is running:
```bash
python start_vllm_server.py
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Error: "CUDA out of memory"

**Solutions**:
1. Reduce context length:
   ```bash
   python start_vllm_server.py --max-model-len 4096
   ```

2. Lower GPU memory utilization:
   ```bash
   python start_vllm_server.py --gpu-memory-utilization 0.7
   ```

3. Use quantization:
   ```bash
   python start_vllm_server.py --quantization awq
   ```

4. Try a smaller model or use CPU (very slow):
   ```bash
   python start_vllm_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```

### Slow Response Times

**Solutions**:
1. Use a model with better hardware support
2. Enable tensor parallelism if you have multiple GPUs
3. Reduce `max-model-len` to free up memory for larger batch sizes

### Model Download Issues

Models are downloaded from Hugging Face automatically. If you have issues:

1. **Set up authentication** (for gated models like Llama):
   ```bash
   huggingface-cli login
   ```

2. **Pre-download models**:
   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
   ```

3. **Use local model path**:
   ```bash
   python start_vllm_server.py --model /path/to/local/model
   ```

---

## 🔄 Switching Back to Azure OpenAI

To switch back to Azure OpenAI API:

1. Edit `core/api_config.py` and `web_vis/core/api_config.py`:
   ```python
   USE_LOCAL_VLLM = False
   ```

2. Restart your application

---

## 📊 Performance Comparison

| Metric | Azure OpenAI | vLLM (8B Model) |
|--------|--------------|-----------------|
| **Latency** | ~2-5s (network) | ~0.5-2s (local) |
| **Throughput** | API limited | 20-100 tokens/s |
| **Cost** | $$$ per request | Free (hardware) |
| **Quality** | GPT-4o (best) | Good (8B models) |
| **Privacy** | Cloud | Local (100% private) |

---

## 💡 Tips for Best Results

1. **First run is slow**: Model downloads (~5-15GB) and initialization takes time
2. **Keep server running**: Start server once, make multiple requests
3. **Monitor GPU usage**: Use `nvidia-smi` to check VRAM usage
4. **Tune for your hardware**: Adjust `--gpu-memory-utilization` based on your GPU
5. **Test model quality**: Try different models to find the best for your prompts

---

## 📝 Configuration Files Modified

The following files have been modified to support vLLM:

- ✅ `core/api_config.py` - Added vLLM configuration
- ✅ `core/vllm_client.py` - New vLLM client module
- ✅ `core/agents.py` - Updated to use vLLM when configured
- ✅ `core/chat_manager.py` - Updated to use vLLM when configured
- ✅ `web_vis/core/api_config.py` - Added vLLM configuration
- ✅ `web_vis/core/vllm_client.py` - New vLLM client module
- ✅ `web_vis/core/agents.py` - Updated to use vLLM when configured
- ✅ `web_vis/core/chat_manager.py` - Updated to use vLLM when configured

All changes are backward compatible - you can switch between Azure OpenAI and vLLM by changing one flag!

---

## 🆘 Support

If you encounter issues:

1. Check the [vLLM documentation](https://docs.vllm.ai/)
2. Verify your GPU has enough VRAM (16GB+ recommended)
3. Ensure CUDA is properly installed
4. Test with the minimal example in `core/vllm_client.py`

---

**Happy local LLM inference! 🚀**
