import os

# ============================================================
# CONFIGURATION: Choose between Azure OpenAI API or Local vLLM
# ============================================================
USE_LOCAL_VLLM = True  # Set to True for local vLLM, False for Azure OpenAI API

# ============================================================
# Azure OpenAI API Configuration (when USE_LOCAL_VLLM = False)
# ============================================================
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY

AZURE_OPENAI_ENDPOINT = "https://xxxxxxxxxxxxxx.openai.azure.com/"
OPENAI_API_VERSION = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

MODEL_NAME="gpt-4o"

# ============================================================
# vLLM Configuration (when USE_LOCAL_VLLM = True)
# ============================================================
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Alternative models you can try:
# VLLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# VLLM_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
# VLLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

os.environ["VLLM_BASE_URL"] = VLLM_BASE_URL
os.environ["VLLM_MODEL_NAME"] = VLLM_MODEL_NAME
