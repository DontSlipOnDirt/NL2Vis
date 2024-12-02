import os
# set your OPENAI_API_BASE, OPENAI_API_KEY here!
API_KEY = "5FGygbhO86laoGOvGJOST59kIYNJfpvLPr5pVBl4eexMHbkgl56DJQQJ99AKACi0881XJ3w3AAABACOGNMr3"
# API_KEY = "sk-wrdbEoFKz2IIEMky3yuH5HcPIyKA050aJCK847QeafnrXQ4L" #付费key
os.environ["OPENAI_API_KEY"] = API_KEY

# API_BASE = "https://api.chatanywhere.tech/v1"
# os.environ["OPENAI_API_BASE"] = API_BASE

AZURE_OPENAI_ENDPOINT = "https://nv-agent.openai.azure.com/"
OPENAI_API_VERSION = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

MODEL_NAME="gpt-4o"