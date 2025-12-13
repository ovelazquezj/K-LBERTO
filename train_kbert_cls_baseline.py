#!/usr/bin/env python3
import subprocess
import os
import time
from datetime import datetime
from pathlib import Path

class KBERTClassificationBaseline:
    def __init__(self):
        self.setup_directories()

    def setup_directories(self):
        Path('./outputs/kbert_cls').mkdir(parents=True, exist_ok=True)
        Path('./outputs/kbert_cls/monitoring').mkdir(exist_ok=True)
        print("✓ Directories ready")

    def build_training_command(self):
        cmd = [
            "python3 -u run_kbert_cls_spanish.py",
            "--pretrained_model_path ./models/beto_uer_model/pytorch_model.bin",
            "--config_path ./models/beto_uer_model/config.json",
            "--vocab_path ./models/beto_uer_model/vocab.txt",
            "--train_path ./datasets/paws_x_spanish/train_kbert.tsv",
            "--dev_path ./datasets/paws_x_spanish/validation_kbert.tsv",
            "--test_path ./datasets/paws_x_spanish/test_kbert.tsv",
            "--epochs_num 5",
            "--batch_size 32",
            "--learning_rate 2e-05",
            "--kg_name ./brain/kgs/WikidataES_CLEAN_v251109.spo",
            "--output_model_path ./outputs/kbert_cls/kbert_beto_cls_baseline.bin",
            "--seq_length 128"
        ]
        return " ".join(cmd)

    def run_training(self):
        print("\n" + "="*80)
        print("🚀 K-BERT Classification BASELINE Training (Spanish PAWS-X)")
        print("="*80)
        cmd = self.build_training_command()
        start_time = time.time()
        log_file = f"./outputs/kbert_cls/training_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print(f"📝 Logs: {log_file}\n")
        with open(log_file, 'w') as log:
            process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
            process.wait()
        
        elapsed = time.time() - start_time
        if process.returncode == 0:
            print(f"✓ Training completed ({elapsed/3600:.1f}h)")
        else:
            print(f"✗ Training failed (code {process.returncode})")

if __name__ == "__main__":
    trainer = KBERTClassificationBaseline()
    trainer.run_training()
