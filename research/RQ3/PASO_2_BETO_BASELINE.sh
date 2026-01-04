#!/bin/bash

# ============================================
# PASO 2: BETO BASELINE TRAINING (SIMPLIFIED)
# Jetson Orin NX | Real TASS 2019 Data
# ============================================

set -e

echo "========================================"
echo "PASO 2: BETO BASELINE TRAINING"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

echo -e "\nStarting training..."

python << 'PYTHON_EOF'
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Configuration
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
DATA_DIR = './data/raw'
OUTPUT_DIR = './models/PASO_2_BETO_BASELINE'
RESULTS_DIR = './results'

# Valid labels from TASS 2019
LABEL_MAP = {'N': 0, 'P': 1, 'NEU': 2, 'NONE': 3}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_data_from_tsv(file_path):
    """Load TSV and return texts and labels"""
    texts = []
    labels = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label_str, text = parts
                
                if label_str in LABEL_MAP and text.strip():
                    texts.append(text.strip())
                    labels.append(LABEL_MAP[label_str])
                else:
                    skipped += 1
    
    return texts, labels, skipped

print("\n" + "="*70)
print("PASO 2: BETO BASELINE TRAINING")
print("="*70)

# Device check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load data
print(f"\n" + "="*70)
print("LOADING DATA")
print("="*70)

train_texts, train_labels, train_skip = load_data_from_tsv(f'{DATA_DIR}/train.tsv')
test_texts, test_labels, test_skip = load_data_from_tsv(f'{DATA_DIR}/test.tsv')

print(f"\nTrain: {len(train_texts)} examples (skipped {train_skip})")
print(f"Test: {len(test_texts)} examples (skipped {test_skip})")

if len(train_texts) == 0:
    print("✗ No training data loaded!")
    exit(1)

# Load model and tokenizer
print(f"\n" + "="*70)
print("LOADING MODEL")
print("="*70)

print(f"\nModel: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_MAP),
)

print(f"✓ Model loaded: {model.num_parameters():,} parameters")

# Prepare datasets
print(f"\n" + "="*70)
print("PREPARING DATASETS")
print("="*70)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
    )

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

print(f"✓ Train dataset prepared: {len(train_dataset)}")

# Test dataset (optional)
test_dataset = None
if len(test_texts) > 0:
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    print(f"✓ Test dataset prepared: {len(test_dataset)}")

# Compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': (predictions == labels).mean(),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0),
    }

# Training arguments
print(f"\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy='no',  # No evaluation during training (only test at end)
    save_strategy='no',
    seed=42,
    fp16=True,
    optim='adamw_torch',
)

print(f"Batch size: 32")
print(f"Learning rate: 2e-5")
print(f"Epochs: 5")

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

print(f"\n" + "="*70)
print("TRAINING...")
print("="*70)

start = datetime.now()
train_result = trainer.train()
end = datetime.now()
training_time = (end - start).total_seconds() / 3600

print(f"\n✓ Training complete ({training_time:.2f} hours)")
print(f"Final loss: {train_result.training_loss:.4f}")

# Evaluate on test
print(f"\n" + "="*70)
print("EVALUATION")
print("="*70)

results = {
    'model': MODEL_NAME,
    'timestamp': datetime.now().isoformat(),
    'training_time_hours': training_time,
    'training_loss': float(train_result.training_loss),
    'data_stats': {
        'train_examples': len(train_texts),
        'test_examples': len(test_texts),
    }
}

if test_dataset:
    print(f"\nEvaluating on test set ({len(test_dataset)} examples)...")
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = np.array(test_labels)
    
    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=list(REVERSE_LABEL_MAP.values()),
        output_dict=True,
        zero_division=0,
    )
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=list(REVERSE_LABEL_MAP.values()),
        zero_division=0,
    ))
    
    results['test_results'] = report
    results['f1_macro'] = report['macro avg']['f1-score']
    results['f1_weighted'] = report['weighted avg']['f1-score']

# Save results
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
results_file = Path(RESULTS_DIR) / 'PASO_2_BASELINE_RESULTS.json'

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved: {results_file}")

# Save model
model_save_path = Path(OUTPUT_DIR) / 'best_model'
trainer.save_model(model_save_path)
print(f"✓ Model saved: {model_save_path}")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTraining time: {training_time:.2f} hours")
print(f"Final loss: {train_result.training_loss:.4f}")

if 'f1_macro' in results:
    print(f"\nTest F1 (macro): {results['f1_macro']:.4f}")
    print(f"Test F1 (weighted): {results['f1_weighted']:.4f}")
    
    if 'test_results' in results:
        for label in REVERSE_LABEL_MAP.values():
            if label in results['test_results']:
                f1 = results['test_results'][label]['f1-score']
                print(f"  {label}: F1={f1:.4f}")

print(f"\n✓ PASO 2: COMPLETE")

PYTHON_EOF

echo ""
echo "========================================"
echo "PASO 2: COMPLETE ✓"
echo "========================================"
echo ""
echo "Results: ~/projects/RQ3_TASS/results/PASO_2_BASELINE_RESULTS.json"
echo "Model: ~/projects/RQ3_TASS/models/PASO_2_BETO_BASELINE/best_model"
echo "========================================"
