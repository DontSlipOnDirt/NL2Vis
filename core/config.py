"""
Centralized configuration for the project.
Replaces dispersed config files in core/.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Global Project Settings
# =============================================================================
DATASET_FOLDER = "visEval_dataset"
LIBRARY = "matplotlib"
LOG_FOLDER = "evaluate_logs"
RESULTS_FOLDER = "results"
AGENT_LOG_FILE = "agent_logs.txt"
# Relative to project root
WEBDRIVER_PATH = "./chrome/chromedriver-linux64/chromedriver.exe" 

# =============================================================================
# Text Model Configuration
# =============================================================================
# Toggle between Azure API and local vLLM
USE_VLLM = True  # Set to False to use Azure API

# --- Azure OpenAI Settings ---
# Set your AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_KEY here!
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
AZURE_OPENAI_ENDPOINT = "https://XXXXXXXX.openai.azure.com/"
OPENAI_API_VERSION = "2024-02-01"
# Default Azure model
AZURE_MODEL_NAME = "gpt-4o"
# Set Environment Variables for Azure/OpenAI clients
# os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
# os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
# os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

# --- vLLM Text Model Settings ---
VLLM_HOST = "localhost"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
# VLLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
# VLLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
# VLLM_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
# VLLM_MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

# VLLM_MODEL_NAME = "google/gemma-4-31B-it"

VLLM_MAX_MODEL_LEN = 8192
VLLM_MAX_TOKENS = 1024
VLLM_TEMPERATURE = 0.0
VLLM_GPU_MEMORY_UTILIZATION = 0.90
# Options: None, "awq", "gptq", "squeezellm", "awq_marlin"
VLLM_QUANTIZATION = None
# Options: "auto", "float16", "bfloat16"
VLLM_DTYPE = "auto"

# =============================================================================
# Vision Model Configuration
# =============================================================================
# Toggle for Vision Models for either visual critic or inference in NL2Vis
# toggle to use OpenAI Vision API (if False, tries to use vLLM vision model as visual critic) 
# Both can be True, in which case vLLM vision is used for NL2Vis and OpenAI vision is used for visual critic
USE_OPENAI_VISION = False
USE_VISION_VLLM = False

# Toggle for Reviewer Agent
ENABLE_REVIEWER_AGENT = False

# --- OpenAI Vision Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL_NAME = "gpt-4o-mini"
OPENAI_VISION_TEMPERATURE = 0.0
OPENAI_VISION_MAX_TOKENS = 1024
OPENAI_VISION_MAX_RETRIES = 10
OPENAI_VISION_TIMEOUT = 60

# --- vLLM Vision Settings ---
VISION_VLLM_HOST = "localhost"
VISION_VLLM_PORT = 8000
VISION_VLLM_BASE_URL = f"http://{VISION_VLLM_HOST}:{VISION_VLLM_PORT}/v1"

# VISION_VLLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
# VISION_VLLM_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
VISION_VLLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
if USE_VISION_VLLM:
    VLLM_MODEL_NAME = VISION_VLLM_MODEL_NAME

# --- Active Text Model Logic ---
if USE_VLLM:
    MODEL_NAME = VLLM_MODEL_NAME
    print(f"[CONFIG] Using vLLM with model: {MODEL_NAME}")
# else:
#     MODEL_NAME = AZURE_MODEL_NAME
#     print("[CONFIG] vLLM disabled, using Azure OpenAI API")

VISION_VLLM_MAX_MODEL_LEN = 8192
VISION_VLLM_MAX_TOKENS = 512
VISION_VLLM_TEMPERATURE = 0.0
VISION_VLLM_GPU_MEMORY_UTILIZATION = 0.90
VISION_VLLM_QUANTIZATION = "awq_marlin"
VISION_VLLM_DTYPE = "auto"
