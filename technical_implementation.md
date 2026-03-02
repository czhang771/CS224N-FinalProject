# CS224N Final Project
# Technical Implementation Document
## Hallucination Detection via Uncertainty Signals

---

# 1. Project Overview

This project investigates whether token-level uncertainty signals can predict hallucinations in long-form biomedical QA responses.

We use:

- Dataset: PubMedQA (1000 labeled examples)
- Generator Model: Qwen2-3B-Instruct
- LLM Judge: GPT-4o or Gemini 2.5 Flash
- Classifier: Logistic Regression
- Features: Token-level uncertainty + evidence alignment signals

Goal:
Train an early classifier that predicts hallucination from intermediate generation signals.

---

# 2. System Architecture

Pipeline:

PubMedQA
    ↓
Generator Model (Qwen2-3B)
    ↓
Log Uncertainty Signals
    ↓
LLM Judge (Hallucination Label)
    ↓
Feature Extraction
    ↓
Train Logistic Regression Classifier
    ↓
Evaluation + Ablations

---
