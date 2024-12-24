# AI Model Benchmark Tool

## Overview
A robust and flexible tool for evaluating AI models using various performance metrics. This tool is designed to work seamlessly with PyTorch models and provides comprehensive evaluation capabilities with detailed error handling and reporting.

## Features
- 🚀 Easy-to-use interface for model evaluation
- 📊 Support for multiple evaluation metrics
- 💪 Robust error handling and validation
- 🔄 CUDA support with automatic device selection
- 📈 Progress tracking during evaluation
- 🛡️ Type checking and input validation

## Supported Metrics
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- AUC-ROC (One-vs-Rest)

## Installation
bash
pip install ai-benchmark-tool
```

## Quick Start
```python
from ai_benchmark import ModelEvaluator
import torch
```

# Initialize the evaluator
```python
evaluator = ModelEvaluator(device='cuda') # or 'cpu'
```

# Prepare your model and data

model = YourModel()
dataloader = YourDataLoader()

# Evaluate the model
```python
metrics = ['accuracy', 'precision', 'f1']
results = evaluator.evaluate_model(model, dataloader, metrics)
print(results)
```
