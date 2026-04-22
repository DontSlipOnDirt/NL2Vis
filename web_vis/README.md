# nvAgent Web Interface

This web app provides a simple UI to upload CSV files and generate visualizations from natural-language prompts.

## Prerequisites

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Web App

Start from repository root:

```bash
streamlit run web_vis/app.py
```

The app will open in your browser at `http://localhost:8501`.

**Prerequisites:**
- vLLM server running (start with `python -m core.vllm_server`) OR
- Azure OpenAI credentials configured (set in UI under Server & Config tab)

## How It Works

- upload one or more CSV files
- submit a natural-language visualization request
- app calls `ChatManager` to generate chart code
- generated SVG can be downloaded from the app

## Notes

- Uploaded files are stored in `uploads/`.
- Runtime logs are written by the app and core components.
- For model setup, first complete:
  - `README.md` quickstart
  - `README.md` -> `Text Model Setup (vLLM)`
  - `README.md` -> `Vision Model Setup`
