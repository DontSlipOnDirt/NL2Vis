import os
# set your AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_KEY here!
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY

AZURE_OPENAI_ENDPOINT = "https://xxxxxxxxxxxxxx.openai.azure.com/"
OPENAI_API_VERSION = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

MODEL_NAME="gpt-4o"
