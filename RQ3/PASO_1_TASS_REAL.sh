#!/bin/bash

echo "========================================"
echo "PASO 1: TASS 2019 DOWNLOAD & SPLIT"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

# Clean old data
rm -f ./data/raw/*.tsv

# Run Python download script
python << 'PYTHON_EOF'
from datasets import load_dataset
from pathlib import Path
import json
from collections import Counter

print("\n" + "="*70)
print("PASO 1: TASS 2019 DOWNLOAD & SPLIT")
print("="*70)

# Load dataset
print("\nLoading TASS 2019 from HuggingFace...")
dataset = load_dataset('mrm8488/tass-2019')

print(f"Original splits: {list(dataset.keys())}")

# Create 80/20 split from 'train'
print("\nCreating 80/20 train/test split...")
full_train = dataset['train']

split_dataset = full_train.train_test_split(test_size=0.2, seed=42)

print(f"  Train: {len(split_dataset['train'])} examples")
print(f"  Test: {len(split_dataset['test'])} examples")

# Output directory
output_dir = Path('./data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

stats = {}

# Save splits as TSV
for split_name in ['train', 'test']:
    split_data = split_dataset[split_name]
    
    output_file = output_dir / f"{split_name}.tsv"
    
    labels = Counter()
    tweet_lengths = []
    
    print(f"\nProcessing {split_name}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("label\ttext\n")
        
        for example in split_data:
            # Extract text and label (TASS 2019 uses 'sentence' and 'sentiments')
            text = example.get('sentence', '')
            label = example.get('sentiments', '')
            
            # Clean text
            text = str(text).replace('\n', ' ').replace('\t', ' ').strip()
            
            if text and label:
                f.write(f"{label}\t{text}\n")
                labels[label] += 1
                tweet_lengths.append(len(text.split()))
    
    if labels:
        stats[split_name] = {
            'examples': sum(labels.values()),
            'labels': dict(labels),
            'avg_length': sum(tweet_lengths) / len(tweet_lengths) if tweet_lengths else 0,
            'min_length': min(tweet_lengths) if tweet_lengths else 0,
            'max_length': max(tweet_lengths) if tweet_lengths else 0,
        }
        
        print(f"  File: {output_file}")
        print(f"  Examples: {stats[split_name]['examples']}")
        print(f"  Labels: {stats[split_name]['labels']}")
        print(f"  Avg length: {stats[split_name]['avg_length']:.1f} words")

# Save statistics
stats_file = output_dir / "TASS_STATS.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✓ Statistics saved: {stats_file}")

# Create summary
summary_file = output_dir / "TASS_SUMMARY.txt"
with open(summary_file, 'w') as f:
    f.write("TASS 2019 DATASET SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("Source: HuggingFace mrm8488/tass-2019 (Real Data)\n")
    f.write("Task: Spanish Sentiment Analysis on Twitter\n")
    f.write("Labels: P (Positive), N (Negative), NONE (None), NEU (Neutral)\n")
    f.write("Split: 80% train, 20% test (seed=42)\n\n")
    
    for split in sorted(stats.keys()):
        s = stats[split]
        f.write(f"{split.upper()}:\n")
        f.write(f"  Examples: {s['examples']}\n")
        f.write(f"  Labels: {s['labels']}\n")
        f.write(f"  Avg tweet length: {s['avg_length']:.1f} words\n")
        f.write(f"  Min/Max: {s['min_length']}/{s['max_length']} words\n\n")

# Print summary
print("\n" + "="*70)
print("DATASET SUMMARY:")
print("="*70)
with open(summary_file) as f:
    print(f.read())

print("✓ PASO 1: DATA PREPARATION COMPLETE\n")

PYTHON_EOF

echo "========================================"
echo "✓ PASO 1: COMPLETE"
echo "========================================"
echo ""
echo "Verifying files:"
wc -l ~/projects/RQ3_TASS/data/raw/*.tsv
