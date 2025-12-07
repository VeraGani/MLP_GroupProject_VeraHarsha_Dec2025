# MLP_GroupProject_VeraHarsha_Dec2025

Final project for **AIDI 1002 – Machine Learning Programming**  
Topic: **Sentiment Analysis in the Era of Large Language Models – Reproduction + Contribution**

## 1. Project Overview

This repository contains our implementation and small extension of the paper:

> **“Sentiment Analysis in the Era of Large Language Models: A Reality Check”**  
> Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, Lidong Bing, 2023.

The original paper evaluates Large Language Models (LLMs) on a wide range of **sentiment analysis** and **aspect-based sentiment analysis** tasks, using **zero-shot prompts**.

In this project we:

1. **Reproduce** the authors’ zero-shot evaluation pipeline using OpenAI models via the public repo `LLM-Sentiment`.
2. **Add our own contribution:** a **classical TF-IDF + Logistic Regression baseline** on the IMDB sentiment dataset and compare it with the LLM zero-shot results.

## 2. Repository Structure
.
├── notebooks/
│   └── phase1_env_setup.ipynb               # Initial Colab environment setup
│
├── outputs/
│   └── contrib_logreg/
│       └── imdb/
│           ├── .gitkeep                     
│           └── prediction.csv               # Logistic Regression predictions (contribution)
│
├── LLM_Sentiment_Contribution.ipynb         # Contribution: TF-IDF + Logistic Regression (IMDB)
├── LLM_sentiment_zero_shot.ipynb            # Zero-shot evaluation pipeline reproduction
├── README.md

