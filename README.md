# 👾nvAgent

## 🙌Introduction

HI! This is the official repository for the paper ["nvAgent: Automated Visualization from Natural Language via Collaborative Agent Workflow"](https://xxxxxxxxxxxxxx).

In this paper, we propose a novel multi-agent framework that integrates preprocessing, *Visualization Query Language* (VQL) generation, and a third stage for output validation and refinement. This comprehensive approach enables *nvAgent* to manage the full complexity of the NL2Vis process, ensuring accuracy and relevance at each step.

<img src="./assets/pipeline_1.jpg" align="middle" width="95%">

## 🎮Demo

We conduct an web interface to demonstrate how to use ***nvAgent*** to generate visualizations from natural language description. Upload .csv files and enter your requirement to generate visualizations simply.

We implement the interface in `web_vis`, and here is an demonstration.

<img src="https://github.com/Ouyangliangge/nvAgent/blob/main/assets/tinywow_web_70526330.gif" width="50%">


## 🎉Updates

## ⚙️Project Structure

This repo is organized as follows:

```txt
├─core
|  ├─agents.py       # define three agents class
|  ├─api_config.py   # config API key and base
|  ├─chat_manager.py # manage the communication between agents
|  ├─const.py        # prompt templates
|  ├─llm.py          # config llm api call and write logs
|  ├─utils.py        # contains utils functions
├─web_vis # the interface for nvAgent
|  ├─core
|  ├─templates
|  ├─app.py
├─visEval # the evaluation framework
|  ├─check # contains different check aspects
|  ├─dataset.py # generate the dataset path mapping
|  ├─evaluate.py # evaluate the score of agent
├─run_evaluate.py # evaluation script
├─README.md
├─requirements.txt
├─visEval_dataset.zip # the dataset used for evaluation
```

## ⚡Start



## 🎰Evaluation

## 💡Citation

## 🪶Contributing
