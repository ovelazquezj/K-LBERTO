#!/usr/bin/env python3
"""
K-BERT Ablation: 0 triplets (SIN Knowledge Graph)
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime

output_dir = Path("./outputs/kbert_cls")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"training_ablation_0_triplets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("\n" + "="*80)
print("🚀 K-BERT Ablation: 0 triplets (SIN Knowledge Graph)")
print("="*80 + "\n")

cmd = (
    "python3 -u run_kbert_cls_spanish.py "
    "--pretrained_model_path ./models/beto_uer_model/pytorch_model.bin "
    "--config_path ./models/beto_uer_model/config.json "
    "--vocab_path ./models/beto_uer_model/vocab.txt "
    "--train_path ./datasets/paws_x_spanish/train_kbert.tsv "
    "--dev_path ./datasets/paws_x_spanish/validation_kbert.tsv "
    "--test_path ./datasets/paws_x_spanish/test_kbert.tsv "
    "--epochs_num 5 "
    "--batch_size 32 "
    "--learning_rate 2e-05 "
    "--kg_name ./brain/kgs/ablation/WikidataES_0_triplets.spo "
    "--output_model_path ./outputs/kbert_cls/kbert_ablation_0_triplets.bin "
    "--seq_length 128"
)

start = time.time()
print(f"📝 Log: {log_file}\n")

with open(log_file, 'w') as log, open(log_file, 'r') as read_log:
    process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    process.wait()

elapsed = time.time() - start

print(f"\n✓ Ablation 0 completed in {elapsed/3600:.2f}h")
print(f"Model: ./outputs/kbert_cls/kbert_ablation_0_triplets.bin\n")
