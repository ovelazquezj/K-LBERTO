#!/bin/bash

# ============================================
# PASO 6 - T2: K-BERT WITHOUT KNOWLEDGE
# Test if K-BERT architecture alone helps
# ============================================

set -e

echo "========================================"
echo "PASO 6 - T2: K-BERT WITHOUT KNOWLEDGE"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

echo -e "\nTraining K-BERT without knowledge injection...\n"

python << 'PYTHON_EOF'
"""
PASO 6 T2: K-BERT without Knowledge
Test K-BERT architecture WITHOUT knowledge injection
Baseline for comparing knowledge impact
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import csv

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

print("\n" + "="*70)
print("PASO 6 T2: K-BERT WITHOUT KNOWLEDGE")
print("="*70)

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
DATA_DIR = './data/raw'
OUTPUT_DIR = './models/PASO_6_T2_KBERT_NO_KG'
RESULTS_DIR = './results'

LABEL_MAP = {'N': 0, 'P': 1, 'NEU': 2, 'NONE': 3}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Load data
print("\nLoading data...")

def load_data_from_tsv(file_path):
    texts = []
    labels = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
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

train_texts, train_labels, _ = load_data_from_tsv(f'{DATA_DIR}/train.tsv')
test_texts, test_labels, _ = load_data_from_tsv(f'{DATA_DIR}/test.tsv')

print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

# Load model
print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_MAP),
)

# Prepare datasets (NO knowledge enhancement)
print("\nTokenizing datasets (NO knowledge)...")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128,  # Standard length, no knowledge
    )

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': (predictions == labels).mean(),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
    }

# Training
print(f"\n" + "="*70)
print("TRAINING K-BERT (NO KNOWLEDGE)")
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
    eval_strategy='no',
    save_strategy='no',
    seed=42,
    fp16=True,
    optim='adamw_torch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

start = datetime.now()
train_result = trainer.train()
end = datetime.now()
training_time = (end - start).total_seconds() / 3600

print(f"\n✓ Training complete ({training_time:.2f} hours)")

# Evaluate
print(f"\n" + "="*70)
print("EVALUATION T2 (NO KNOWLEDGE)")
print("="*70)

results = {
    'treatment': 'T2: K-BERT without Knowledge',
    'timestamp': datetime.now().isoformat(),
    'training_time_hours': training_time,
    'training_loss': float(train_result.training_loss),
    'data_stats': {
        'train_examples': len(train_texts),
        'test_examples': len(test_texts),
        'knowledge_coverage': '0.0% (no knowledge)',
    }
}

predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = np.array(test_labels)

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
results_file = Path(RESULTS_DIR) / 'PASO_6_T2_RESULTS.json'

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

trainer.save_model(Path(OUTPUT_DIR) / 'best_model')

# Summary
print(f"\n" + "="*70)
print("SUMMARY T2")
print("="*70)

baseline_f1 = 0.4577
current_f1 = results['f1_macro']
improvement = ((current_f1 - baseline_f1) / baseline_f1 * 100)

print(f"\nBaseline (T1): {baseline_f1:.4f}")
print(f"T2 (K-BERT no KG): {current_f1:.4f}")
print(f"Difference: {improvement:+.2f}%")

if current_f1 > baseline_f1:
    print("✓ K-BERT architecture alone improves baseline")
else:
    print("✗ K-BERT without knowledge does not improve")

print(f"\n✓ PASO 6 T2: COMPLETE")

PYTHON_EOF

echo ""
echo "========================================"
echo "✓ PASO 6 T2: COMPLETE"
echo "========================================"
echo ""
cat ~/projects/RQ3_TASS/results/PASO_6_T2_RESULTS.json | python -m json.tool | grep -A2 "f1_macro"
