# 👾nvAgent

## 🙌Introduction

This is a fork of official repository for the paper ["nvAgent: Automated Data Visualization from Natural Language via Collaborative Agent Workflow"](https://arxiv.org/abs/2502.05036).

### Abstract
*Natural Language to Visualization* (NL2Vis) seeks to convert natural-language descriptions into visual representations of given tables, empowering users to derive insights from large-scale data. Recent advancements in Large Language Models (LLMs) show promise in automating code generation to transform tabular data into accessible visualizations. However, they often struggle with complex queries that require reasoning across multiple tables. 

To address this limitation, we propose a collaborative agent workflow, termed **nvAgent**, for NL2Vis. Specifically, **nvAgent** comprises three agents: a processor agent for database processing and context filtering, a composer agent for planning visualization generation, and a validator agent for code translation and output verification. 

Comprehensive evaluations on the new VisEval benchmark demonstrate that **nvAgent** consistently surpasses state-of-the-art baselines, achieving a 7.88% improvement in single-table and a 9.23% improvement in multi-table scenarios. Qualitative analyses further highlight that **nvAgent** maintains nearly a 20% performance margin over previous models, underscoring its capacity to produce high-quality visual representations from complex, heterogeneous data sources.

(pipeline in ./assets/pipeline.png)

<img src="./assets/pipeline.png" align="middle" width="95%">

## 🎮Demo

We conduct a web interface to demonstrate how to use ***nvAgent*** to generate visualizations from natural language descriptions. Upload .csv files and enter your requirements to generate visualizations simply.

We implement the interface in `web_vis`, and here is a demonstration. (./assets/tinywow_web_70526330.gif)

<img src="./assets/tinywow_web_70526330.gif" width="50%">

## 🎉Updates

## ⚙️Project Structure

This repo is organized as follows:

```txt
├─core
|  ├─agents.py       # define three agents class
|  ├─config.py       # centralized runtime/model configuration
|  ├─chat_manager.py # manage the communication between agents
|  ├─const.py        # prompt templates
|  ├─llm.py          # config llm api call and write logs
|  ├─utils.py        # contains utils functions
├─tests
|  ├─test_vllm_suite.py   # vLLM setup and integration checks
|  ├─test_visual_agent.py # reviewer/validator visual pipeline checks
|  ├─test_chromedriver.py # selenium/chromedriver compatibility checks
├─web_vis # the interface for nvAgent
|  ├─core
|  ├─templates
|  ├─app.py
├─visEval # the evaluation framework
|  ├─check # contains different check aspects
|  ├─dataset.py # generate the dataset path mapping
|  ├─evaluate.py # evaluate the score of agent
├─run_evaluate.py # evaluation script
├─run_evaluate_test.py # quick subset evaluation script
├─README.md
├─requirements.txt
├─visEval_dataset.zip # the dataset used for evaluation
```

## ⚡Quick Start (Linux)

This section is Linux-first because the primary runtime target is a Linux cluster.

### 1) Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure runtime settings

- Main runtime config: `core/config.py`

Important values to check before running:

- `DATASET_FOLDER` (defaults to `visEval_dataset`)
- `USE_VLLM` / `USE_VISION_VLLM`
- model names and GPU memory utilization
- API credentials when using OpenAI/Azure-backed vision

### 3) Start model servers

```bash
# Terminal 1: text model server
python -m core.vllm_server

# Terminal 2: vision model server (only if using local vision model)
python -m core.vision_vllm_server
```

### 4) Validate setup

```bash
# vLLM checks
python tests/test_vllm_suite.py --all

# Optional browser/visual checks
python tests/test_chromedriver.py
python tests/test_visual_agent.py
```

### 5) Run evaluation

```bash
# Quick subset test
python run_evaluate_test.py 50

# Full evaluation
python run_evaluate.py
```

## 🎰Evaluation Notes

- Results are written under `results/`.
- Web app usage is documented in `web_vis/README.md`.

## 🧠Text Model Setup (vLLM)

Use this section when running local text generation with vLLM.

### Prerequisites

- Python 3.9+
- NVIDIA GPU + CUDA runtime
- Linux environment

### Configure text backend

Update values in:

- `core/config.py`

Typical values to verify:

- `USE_VLLM = True`
- `VLLM_MODEL_NAME`
- `VLLM_GPU_MEMORY_UTILIZATION`
- `VLLM_QUANTIZATION`

### Start and validate text server

```bash
# Terminal 1
python -m core.vllm_server

# Terminal 2
python tests/test_vllm_suite.py --setup
python tests/test_vllm_suite.py --server
python tests/test_vllm_suite.py --integration
```

Or run all checks:

```bash
python tests/test_vllm_suite.py --all
```

### Text model troubleshooting

Check server endpoint:

```bash
curl http://localhost:8000/v1/models
```

If memory is tight, reduce `VLLM_GPU_MEMORY_UTILIZATION` and/or enable quantization.

## 👁️Vision Model Setup

Use this section when scoring chart readability/quality with a vision model.

### Vision modes

- OpenAI vision model configured in `core/config.py`
- Local vLLM vision model served via `core.vision_vllm_server`

### Configure vision backend

Update values in:

- `core/config.py`

All runtime settings are centralized in `core/config.py`.

Typical values to verify:

- `USE_OPENAI_VISION`
- `USE_VISION_VLLM`
- `OPENAI_VISION_MODEL_NAME`
- `VISION_VLLM_MODEL_NAME`

### Start and validate vision path

```bash
# Terminal 1: text server
python -m core.vllm_server

# Terminal 2: vision server (required only if USE_VISION_VLLM=True)
python -m core.vision_vllm_server

# Terminal 3: vision checks
python -m core.vision_vllm_client
python tests/test_visual_agent.py
```

### Vision troubleshooting

Check vision endpoint:

```bash
curl http://localhost:8001/v1/models
```

If vision is not being used, verify `USE_VISION_VLLM` and `USE_OPENAI_VISION` in `core/config.py` and confirm the configured host/port values.

## Windows Notes (Optional)

If you run locally on Windows, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```
