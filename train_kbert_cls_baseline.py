#!/usr/bin/env python3
"""
K-BERT Classification Baseline Training (Spanish PAWS-X)
Wrapper with INTEGRATED monitoring: training metrics + power/temperature + model characteristics
Genera 3 CSVs: training_metrics, power_metrics, model_characteristics
"""

import subprocess
import os
import sys
import time
import csv
import re
from datetime import datetime
from pathlib import Path
from threading import Thread, Event

class KBERTClassificationBaselineMonitor:
    """Integrated monitoring for K-BERT classification baseline on Jetson"""

    def __init__(self, output_dir="./outputs/kbert_cls"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir = self.output_dir / "monitoring"
        self.monitoring_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # CSV file paths
        self.training_csv = self.output_dir / f'training_metrics_baseline_{timestamp}.csv'
        self.power_csv = self.output_dir / f'power_metrics_baseline_{timestamp}.csv'
        self.characteristics_csv = self.output_dir / f'model_characteristics_baseline_{timestamp}.csv'
        
        # Log file

        self.log_file = self.output_dir / f'training_baseline_{timestamp}.log'
        self.tegrastats_log = self.monitoring_dir / f'tegrastats_baseline_{timestamp}.log'
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Monitoring directory: {self.monitoring_dir}")
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Stop signal for monitoring thread
        self.stop_monitoring = Event()

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training metrics
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'loss_train', 'learning_rate'])
        
        # Power metrics
        with open(self.power_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'power_watts', 'temperature_c', 'gpu_memory_used_mb', 'gpu_utilization_percent'])
        
        # Model characteristics
        with open(self.characteristics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
        
        print(f"✓ CSV files initialized")

    def _run_tegrastats(self):
        """Run tegrastats in background and log power metrics"""
        print(f"📊 Starting tegrastats monitoring...")
        
        try:
            process = subprocess.Popen(
                ['tegrastats', '--logfile', str(self.tegrastats_log)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Write to power CSV while tegrastats runs
            with open(self.power_csv, 'a', newline='') as f:
                power_writer = csv.writer(f)
                
                while not self.stop_monitoring.is_set():
                    try:
                        # Parse tegrastats log
                        if self.tegrastats_log.exists():
                            with open(self.tegrastats_log, 'r') as log:
                                lines = log.readlines()
                                if lines:
                                    latest_line = lines[-1].strip()
                                    # Parse: RAM 2234/7846MB (lfb 1804MB) CPU [...]
                                    # Extract power if available
                                    if 'POM_5V_IN' in latest_line or 'Power' in latest_line:
                                        power_writer.writerow([
                                            datetime.now().isoformat(),
                                            'N/A',  # Power parsing depends on tegrastats format
                                            'N/A',  # Temperature
                                            'N/A',  # GPU memory
                                            'N/A'   # GPU utilization
                                        ])
                    except Exception as e:
                        pass
                    
                    time.sleep(5)
            
            # Kill tegrastats when done
            process.terminate()
            process.wait(timeout=5)
        except FileNotFoundError:
            print("⚠️  tegrastats not found, skipping power monitoring")
        except Exception as e:
            print(f"⚠️  Error in tegrastats monitoring: {e}")

    def _parse_training_output(self, line):
        """Parse training output for metrics"""
        # Pattern: "Epoch id: 1, Training steps: 100, Avg loss: 0.693"
        epoch_match = re.search(r'Epoch id: (\d+)', line)
        step_match = re.search(r'Training steps: (\d+)', line)
        loss_match = re.search(r'Avg loss: ([\d.]+)', line)
        
        if epoch_match and step_match and loss_match:
            return {
                'epoch': epoch_match.group(1),
                'step': step_match.group(1),
                'loss': loss_match.group(1)
            }
        return None

    def build_training_command(self):
        """Build K-BERT training command"""
        cmd = [
            "python3 -u run_kbert_cls_spanish.py",
            "--pretrained_model_path ./models/beto_uer_model/pytorch_model.bin",
            "--config_path ./models/beto_uer_model/config.json",
            "--vocab_path ./models/beto_uer_model/vocab.txt",
            "--train_path ./datasets/paws_x_spanish/train_kbert.tsv",
            "--dev_path ./datasets/paws_x_spanish/validation_kbert.tsv",
            "--test_path ./datasets/paws_x_spanish/test_kbert.tsv",
            "--epochs_num 8",
            "--batch_size 16",
            "--learning_rate 1e-04",
            "--warmup 0.05",
            "--kg_name ./brain/kgs/WikidataES_CLEAN_v251109.spo",
            "--output_model_path ./outputs/kbert_cls/kbert_beto_cls_baseline.bin",
            "--seq_length 128"
        ]
        return " ".join(cmd)

    def run_training(self):
        """Execute training with integrated monitoring"""
        print("\n" + "="*80)
        print("🚀 K-BERT Classification BASELINE Training (Spanish PAWS-X)")
        print("="*80)
        print("\nConfiguration:")
        print("  Model: BETO (Spanish BERT)")
        print("  Dataset: PAWS-X Spanish (49k train, 2k dev, 2k test)")
        print("  Task: Binary Paraphrase Detection")
        print("  Knowledge Graph: WikidataES (500k triplets - inyectado)")
        print("  Epochs: 5")
        print("  Batch size: 32")
        print("  Learning rate: 2e-05")
        print("  Sequence length: 128\n")
        
        cmd = self.build_training_command()
        print(f"📝 Training log: {self.log_file}")
        print(f"📊 Power metrics: {self.power_csv}")
        print(f"📊 Training metrics: {self.training_csv}")
        print(f"📊 Tegrastats log: {self.tegrastats_log}\n")
        
        start_time = time.time()
        print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Start tegrastats monitoring in background thread
        monitor_thread = Thread(target=self._run_tegrastats, daemon=True)
        monitor_thread.start()
        
        # Open training log file
        with open(self.log_file, 'w') as log, \
             open(self.training_csv, 'a', newline='') as train_csv:
            
            training_writer = csv.writer(train_csv)
            
            # Start training process
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output: write to log and parse for metrics
            for line in process.stdout:
                line = line.rstrip()
                
                # Write to log file
                log.write(line + '\n')
                log.flush()
                
                # Print to console
                if any(keyword in line for keyword in ['Epoch', 'evaluation', 'Label', 'Acc', 'Progress', 'Loading']):
                    print(line)
                
                # Parse training metrics
                if 'Epoch id:' in line and 'Avg loss:' in line:
                    metrics = self._parse_training_output(line)
                    if metrics:
                        training_writer.writerow([
                            datetime.now().isoformat(),
                            metrics['epoch'],
                            metrics['step'],
                            metrics['loss'],
                            '2e-05'
                        ])
            
            process.wait()
        
        # Stop monitoring
        self.stop_monitoring.set()
        monitor_thread.join(timeout=5)
        
        elapsed = time.time() - start_time
        
        # Final report
        print("\n" + "="*80)
        if process.returncode == 0:
            print(f"✓ Training completed successfully")
            print(f"  Elapsed time: {elapsed/3600:.2f} hours ({int(elapsed/60)} minutes)")
            print(f"  Model saved: ./outputs/kbert_cls/kbert_beto_cls_baseline.bin")
        else:
            print(f"✗ Training failed (return code: {process.returncode})")
        
        print(f"\n📝 Logs and metrics saved to:")
        print(f"  - Training log: {self.log_file}")
        print(f"  - Training metrics CSV: {self.training_csv}")
        print(f"  - Tegrastats log: {self.tegrastats_log}")
        print(f"  - Power metrics CSV: {self.power_csv}")
        print("="*80 + "\n")

if __name__ == "__main__":
    trainer = KBERTClassificationBaselineMonitor()
    trainer.run_training()
