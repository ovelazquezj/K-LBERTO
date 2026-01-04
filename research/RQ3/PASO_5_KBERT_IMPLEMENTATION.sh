#!/bin/bash

# ============================================
# PASO 5: K-BERT IMPLEMENTATION
# Knowledge-enhanced BERT for Spanish sentiment
# Inject 88 curated triplets as knowledge
# ============================================

set -e

echo "========================================"
echo "PASO 5: K-BERT IMPLEMENTATION"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

echo -e "\nImplementing K-BERT with knowledge injection...\n"

python << 'PYTHON_EOF'
"""
PASO 5: K-BERT Implementation
Implement knowledge-enhanced BERT for Spanish sentiment analysis
Strategy: Word-level knowledge injection from curated triplets
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

print("\n" + "="*70)
print("PASO 5: K-BERT IMPLEMENTATION")
print("="*70)

# Configuration
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
DATA_DIR = './data/raw'
OUTPUT_DIR = './models/PASO_5_KBERT_IMPLEMENTATION'
RESULTS_DIR = './results'
KG_FILE = './results/PASO_4_KNOWLEDGE_GRAPH_50K.json'

LABEL_MAP = {'N': 0, 'P': 1, 'NEU': 2, 'NONE': 3}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Load knowledge graph
print("\nLoading knowledge graph...")

with open(KG_FILE, 'r', encoding='utf-8') as f:
    kg_data = json.load(f)

triplets = kg_data['triplets']
print(f"✓ Loaded {len(triplets)} triplets")

# Build knowledge index for word lookup
kg_index = defaultdict(list)
for triplet in triplets:
    subject = triplet['subject'].lower()
    kg_index[subject].append({
        'relation': triplet['relation'],
        'object': triplet['object'],
    })

print(f"✓ Built index for {len(kg_index)} unique subjects")

# Load data
print(f"\n" + "="*70)
print("LOADING DATA")
print("="*70)

import csv

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

train_texts, train_labels, train_skip = load_data_from_tsv(f'{DATA_DIR}/train.tsv')
test_texts, test_labels, test_skip = load_data_from_tsv(f'{DATA_DIR}/test.tsv')

print(f"\nTrain: {len(train_texts)} examples (skipped {train_skip})")
print(f"Test: {len(test_texts)} examples (skipped {test_skip})")

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

# Create knowledge-enhanced dataset
print(f"\n" + "="*70)
print("CREATING KNOWLEDGE-ENHANCED DATASET")
print("="*70)

def add_knowledge_context(text, kg_index, tokenizer):
    """
    Add knowledge context to text based on matching entities
    Strategy: Append knowledge relations as additional context tokens
    """
    words = text.lower().split()
    knowledge_context = []
    
    # Find matching entities in KG
    for word in words:
        word_clean = word.replace(',', '').replace('.', '').replace('!', '').replace('?', '')
        
        if word_clean in kg_index:
            relations = kg_index[word_clean]
            for rel in relations[:2]:  # Limit to 2 relations per word
                # Format: [word relation:object]
                knowledge_context.append(f"[{word_clean} {rel['relation']}:{rel['object']}]")
    
    # Append knowledge context to original text
    if knowledge_context:
        enhanced_text = text + " " + " ".join(knowledge_context)
    else:
        enhanced_text = text
    
    return enhanced_text

print("\nEnhancing training texts with knowledge...")

enhanced_train_texts = []
knowledge_coverage = 0

for i, text in enumerate(train_texts):
    enhanced = add_knowledge_context(text, kg_index, tokenizer)
    enhanced_train_texts.append(enhanced)
    
    if len(enhanced) > len(text):
        knowledge_coverage += 1
    
    if (i + 1) % 200 == 0:
        print(f"  Enhanced {i + 1}/{len(train_texts)}")

print(f"\n✓ Knowledge coverage: {knowledge_coverage}/{len(train_texts)} ({knowledge_coverage/len(train_texts)*100:.1f}%)")

# Prepare datasets with knowledge
print(f"\n" + "="*70)
print("TOKENIZING DATASETS")
print("="*70)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256,  # Increased to accommodate knowledge
    )

train_dataset = Dataset.from_dict({'text': enhanced_train_texts, 'label': train_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

print(f"✓ Train dataset prepared: {len(train_dataset)}")

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
    per_device_train_batch_size=16,  # Reduced due to longer sequences
    per_device_eval_batch_size=16,
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

print(f"Batch size: 16")
print(f"Learning rate: 2e-5")
print(f"Epochs: 5")
print(f"Max length: 256 (with knowledge)")
print(f"Knowledge coverage: {knowledge_coverage/len(train_texts)*100:.1f}%")

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

print(f"\n" + "="*70)
print("TRAINING K-BERT WITH KNOWLEDGE INJECTION...")
print("="*70)

start = datetime.now()
train_result = trainer.train()
end = datetime.now()
training_time = (end - start).total_seconds() / 3600

print(f"\n✓ Training complete ({training_time:.2f} hours)")
print(f"Final loss: {train_result.training_loss:.4f}")

# Evaluate on test
print(f"\n" + "="*70)
print("EVALUATION - K-BERT WITH KNOWLEDGE")
print("="*70)

results = {
    'model': 'K-BERT (BETO base)',
    'timestamp': datetime.now().isoformat(),
    'training_time_hours': training_time,
    'training_loss': float(train_result.training_loss),
    'data_stats': {
        'train_examples': len(train_texts),
        'test_examples': len(test_texts),
        'knowledge_coverage': f"{knowledge_coverage/len(train_texts)*100:.1f}%",
    },
    'knowledge_graph': {
        'total_triplets': len(triplets),
        'unique_subjects': len(kg_index),
    }
}

if len(test_dataset) > 0:
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
results_file = Path(RESULTS_DIR) / 'PASO_5_KBERT_RESULTS.json'

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Results saved: {results_file}")

# Save model
model_save_path = Path(OUTPUT_DIR) / 'best_model'
trainer.save_model(model_save_path)
print(f"✓ Model saved: {model_save_path}")

# Comparison with baseline
print(f"\n" + "="*70)
print("COMPARISON WITH BASELINE")
print("="*70)

baseline_f1 = 0.4577
current_f1 = results.get('f1_macro', 0)
improvement = ((current_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0

print(f"\nBaseline (BETO vanilla): {baseline_f1:.4f}")
print(f"K-BERT with knowledge: {current_f1:.4f}")
print(f"Improvement: {improvement:+.2f}%")

if current_f1 > baseline_f1:
    print(f"\n✓✓✓ K-BERT IMPROVED over baseline!")
else:
    print(f"\n⚠ K-BERT did not improve (this is OK, will try different approaches in PASO 6)")

# Summary
print(f"\n" + "="*70)
print("SUMMARY - PASO 5")
print("="*70)

print(f"""
Model: K-BERT (BETO base with knowledge injection)
Training time: {training_time:.2f} hours
Final loss: {train_result.training_loss:.4f}

Knowledge Graph:
  - Triplets: {len(triplets)}
  - Coverage: {knowledge_coverage/len(train_texts)*100:.1f}% of training tweets

Results:
  - F1 macro: {current_f1:.4f}
  - F1 weighted: {results.get('f1_weighted', 0):.4f}
  - Improvement vs baseline: {improvement:+.2f}%

Next step (PASO 6):
- Test T2: K-BERT + 0 KG
- Test T4: K-BERT + 500k generic KG
- Compare T1 vs T2 vs T3 vs T4
- Validate: Curation > Scale
""")

print(f"\n✓ PASO 5: K-BERT IMPLEMENTATION COMPLETE")

PYTHON_EOF

echo ""
echo "========================================"
echo "✓ PASO 5: COMPLETE"
echo "========================================"
echo ""
echo "K-BERT Model saved:"
ls -lh ~/projects/RQ3_TASS/models/PASO_5_KBERT_IMPLEMENTATION/best_model/
echo ""
echo "Results:"
cat ~/projects/RQ3_TASS/results/PASO_5_KBERT_RESULTS.json | python -m json.tool | head -30
echo ""
echo "Next: PASO 6 - Experiments (4 treatments comparison)"
echo "========================================"
