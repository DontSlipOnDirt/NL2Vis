import os
# set your OPENAI_API_BASE, OPENAI_API_KEY here!
# API_KEY = "sk-VTK6uXZvIG2S0BYj5aAI345JUZ9bNrSm6dOFA8UXRndor0Z5"#免费key
API_KEY = "sk-wrdbEoFKz2IIEMky3yuH5HcPIyKA050aJCK847QeafnrXQ4L" #付费key
# API_KEY = "sk-mYx208c58b27de4a65129c1c1f9611035dae103e8f0ffVRv"
os.environ["OPENAI_API_KEY"] = API_KEY
API_BASE = "https://api.chatanywhere.tech/v1"
# API_BASE = "https://api.gptsapi.net"
os.environ["OPENAI_API_BASE"] = API_BASE

# MODEL_NAME = 'gpt-4-1106-preview' # 128k 版本
# MODEL_NAME = 'CodeLlama-7b-hf'
# MODEL_NAME = 'gpt-4-32k' # 0613版本
# MODEL_NAME = 'gpt-4' # 0613版本
MODEL_NAME = 'gpt-3.5-turbo' # 0613版本