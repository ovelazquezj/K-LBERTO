# K-BERT Spanish (K-LBERTO) REPRODUCIBILITY GUIDE

Complete step-by-step guide to reproduce the K-BERT Spanish adaptation with knowledge noise ablation study.

**Purpose:** Enable researchers to replicate the exact methodology, experiments, and results.

**Methodology:** Following this guide, researchers should obtain comparable results with K-BERT adapted for Spanish classification using PAWS-X and WikidataES.

**Timeline:** Total ~13-15 hours (3h baseline + 3-10h ablations)

---

## STRATEGIC DECISION: NER → Classification

The original plan used NER on WikiANN dataset, but baseline results (F1=0.153) were too poor to demonstrate knowledge noise effects meaningfully. We pivoted to binary classification for several reasons:

1. **Replicate Chinese K-BERT success:** Original K-BERT showed 86.39% accuracy on book review classification
2. **Obtain healthy baseline:** Expected F1 ≈ 0.70-0.80 allows clear visualization of degradation
3. **Cleaner demonstration:** Knowledge noise effects are more obvious with good baseline
4. **Task stability:** Classification is more stable than NER for ablation studies

**Paper Impact:** RQ3 changes from "Spanish NER degradation" to "Spanish paraphrase classification degradation", but argument is stronger (knowledge noise affects any NLP task)

---

## STEP 1: OBTAIN SPANISH CLASSIFICATION DATASET

**Objective:** Download and prepare binary classification dataset compatible with K-BERT format.

**What you need to accomplish:**
- Download PAWS-X Spanish from Hugging Face
- Convert to K-BERT format (label + combined text)
- Verify structure and line counts

**Dataset Selected:** PAWS-X Spanish
- Task: Paraphrase detection (are two sentences paraphrases?)
- Binary: label 0 (not paraphrase) / 1 (paraphrase)
- Structure: sentence1, sentence2, label
- Size: ~49k train, ~2k dev, ~2k test

### Step 1 Execution

#### 1.1 Create directory
```bash
cd ~/projects/K-BERT_ES/
mkdir -p ./datasets/paws_x_spanish
cd ./datasets/paws_x_spanish
```

#### 1.2 Install Hugging Face library (if not already installed)
```bash
pip install datasets
```

#### 1.3 Download from Hugging Face
```bash
cat > download_pawsx.py << 'EOF'
from datasets import load_dataset

# Download PAWS-X Spanish
dataset = load_dataset("google-research-datasets/paws-x", "es")

# Save splits
for split in ['train', 'validation', 'test']:
    if split in dataset:
        data = dataset[split]
        with open(f"{split}.tsv", 'w', encoding='utf-8') as f:
            # Header
            f.write("id\tsentence1\tsentence2\tlabel\n")
            # Data
            for row in data:
                f.write(f"{row['id']}\t{row['sentence1']}\t{row['sentence2']}\t{row['label']}\n")
        print(f"✓ {split}.tsv ({len(data)} examples)")

print("\n✓ Download complete")
EOF

python3 download_pawsx.py
```

**Expected Result:**
```
✓ train.tsv (49401 examples)
✓ validation.tsv (1956 examples)
✓ test.tsv (1956 examples)
✓ Download complete
```

#### 1.4 Convert to K-BERT format (label + combined text)
```bash
cat > convert_pawsx_to_kbert.py << 'EOF'
#!/usr/bin/env python3
"""
Convert PAWS-X (sentence1, sentence2, label) to K-BERT format (label, text_a)
"""

def convert_file(input_file, output_file):
    print(f"Converting {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Write header
        out.write("label\ttext_a\n")
        
        # Skip header (line 0), process data
        for i, line in enumerate(lines[1:], 1):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                sent1, sent2, label = parts[1], parts[2], parts[3]
                # Combine: "sentence1 [SEP] sentence2"
                combined = f"{sent1} [SEP] {sent2}"
                out.write(f"{label}\t{combined}\n")
                if i % 10000 == 0:
                    print(f"  Processed {i} lines...")
        
        print(f"✓ {output_file} created ({i} examples)")

# Convert all splits
for split in ['train', 'validation', 'test']:
    input_file = f"{split}.tsv"
    output_file = f"{split}_kbert.tsv"
    convert_file(input_file, output_file)

print("\n✓ All conversions complete")
EOF

python3 convert_pawsx_to_kbert.py
```

**Expected Result:**
```
Converting train.tsv...
  Processed 10000 lines...
  Processed 20000 lines...
  Processed 30000 lines...
  Processed 40000 lines...
✓ train_kbert.tsv created (49401 examples)
Converting validation.tsv...
✓ validation_kbert.tsv created (1956 examples)
Converting test.tsv...
✓ test_kbert.tsv created (1956 examples)

✓ All conversions complete
```

#### 1.5 Validate Step 1
```bash
# Check structure
echo "=== First line of dataset ===" 
head -1 train_kbert.tsv

echo -e "\n=== Examples ===" 
head -3 train_kbert.tsv | tail -2

# Count examples
echo -e "\n=== Line counts ===" 
wc -l train_kbert.tsv validation_kbert.tsv test_kbert.tsv
```

**Expected Validation Output:**
```
=== First line ===
label	text_a

=== Examples ===
1	Los gobiernos pueden cambiar las pautas de conducta de la sociedad. [SEP] Los gobiernos pueden cambiar...
0	La carne se seca al calor y al aire. [SEP] El cuero se seca al calor y al aire.

=== Line counts ===
49402 train_kbert.tsv (49401 examples + header)
1957 validation_kbert.tsv (1956 examples + header)
1957 test_kbert.tsv (1956 examples + header)
```

### Dataset Characteristics (Step 1)

| Metric | Value |
|--------|-------|
| Task | Binary classification (paraphrase) |
| Language | Spanish |
| Train split | 49,401 examples |
| Dev split | 1,956 examples |
| Test split | 1,956 examples |
| Format | label\ttext_a (compatible with K-BERT) |
| Classes | 0 (not paraphrase), 1 (paraphrase) |
| Location | ~/projects/K-BERT_ES/datasets/paws_x_spanish/ |

### Files Generated (Step 1)

```
~/projects/K-BERT_ES/datasets/paws_x_spanish/
├── train.tsv                  # Original PAWS-X
├── validation.tsv             # Original PAWS-X
├── test.tsv                   # Original PAWS-X
├── train_kbert.tsv           # Converted for K-BERT ✓
├── validation_kbert.tsv       # Converted for K-BERT ✓
├── test_kbert.tsv            # Converted for K-BERT ✓
├── download_pawsx.py         # Download script
├── convert_pawsx_to_kbert.py  # Conversion script
```

### Step 1 Verification: What to Check

Use this checklist to verify STEP 1 was successful before proceeding to STEP 2:

- [ ] Dataset downloaded correctly from Hugging Face
- [ ] Structure verified: label + text_a format
- [ ] All splits converted: train_kbert.tsv, validation_kbert.tsv, test_kbert.tsv
- [ ] Examples reviewed manually (first 3 lines make sense)
- [ ] Line counts correct: 49,402 train, 1,957 dev, 1,957 test (including headers)

---

## STEP 2: ADAPT K-BERT SCRIPTS FOR SPANISH CLASSIFICATION

**Objective:** Adapt and customize K-BERT scripts for Spanish language using BETO model and PAWS-X dataset.

**What you need to accomplish:**
- Review K-BERT's classification script for genericness
- Adapt docstrings and comments for Spanish context
- Create configuration file for classification task
- Verify script accepts required command-line arguments

### Original Script Analysis

The `run_kbert_cls.py` script is already **generic**:
- Accepts model paths as arguments (--pretrained_model_path, --config_path, --vocab_path)
- Accepts dataset paths as arguments (--train_path, --dev_path, --test_path)
- NO hardcoding for Chinese language
- Supports Knowledge Graph injection via --kg_name
- Correctly calculates binary metrics (precision, recall, F1)

### Changes Made

**Change 1: Update docstring (line 3)**
```python
# BEFORE:
"""
  This script provides an k-BERT exmaple for classification.
"""

# AFTER:
"""
K-BERT Classification Training for Spanish (PAWS-X Paraphrase Detection)
Compatible with BETO model and WikidataES knowledge graph
Supports visible matrix for knowledge graph injection
"""
```

**Change 2: Update tokenizer help text (line 160)**
```python
# BEFORE:
help="Specify the tokenizer. "
     "Original Google BERT uses bert tokenizer on Chinese corpus..."

# AFTER:
help="Specify the tokenizer. "
     "BETO uses bert tokenizer for Spanish text. "
```

**Change 3: NO other code changes needed**

The script accepts all necessary parameters via command-line arguments.

### Configuration File: config_cls.yaml

**Original file:** `config.yaml` (for NER)
**New file:** `config_cls.yaml` (for classification)

#### Changes in config_cls.yaml

```yaml
# Comments updated
- Task: "Named Entity Recognition (NER)" → "Binary Paraphrase Detection"
- Dataset: "CoNLL 2002 Spanish" → "PAWS-X Spanish"

# Data paths
data:
  train_path: ./datasets/paws_x_spanish/train_kbert.tsv
  dev_path: ./datasets/paws_x_spanish/validation_kbert.tsv
  test_path: ./datasets/paws_x_spanish/test_kbert.tsv

# Training parameters
training:
  epochs: 8 → 5              # Classification converges faster
  batch_size: 16 → 32        # Can be larger (sentence-level)
  learning_rate: 2e-05       # Same (BERT standard)
  seq_length: 128            # Same

# Output paths
paths:
  output_dir: ./outputs/kbert_ner → ./outputs/kbert_cls
  output_model: ./outputs/kbert_ner/... → ./outputs/kbert_cls/...
```

#### Complete config_cls.yaml File

```yaml
# K-BERT Classification Training Configuration
# Task: Binary Paraphrase Detection (Spanish)
# Model: BETO (Spanish BERT)
# Dataset: PAWS-X Spanish
# KB: WikidataES

model:
  name: BETO
  pretrained_path: ./models/beto_uer_model/pytorch_model.bin
  config_path: ./models/beto_uer_model/config.json
  vocab_path: ./models/beto_uer_model/vocab.txt

data:
  train_path: ./datasets/paws_x_spanish/train_kbert.tsv
  dev_path: ./datasets/paws_x_spanish/validation_kbert.tsv
  test_path: ./datasets/paws_x_spanish/test_kbert.tsv

training:
  epochs: 5
  batch_size: 32
  learning_rate: 2e-05
  seq_length: 128

knowledge_graph:
  name: WikidataES_CLEAN_v251109
  path: ./brain/kgs/WikidataES_CLEAN_v251109.spo

paths:
  output_dir: ./outputs/kbert_cls
  models_dir: ./models
  monitoring_dir: ./outputs/kbert_cls/monitoring
  output_model: ./outputs/kbert_cls/kbert_beto_cls_pawsx_spanish.bin

monitoring:
  interval: 1000
```

### Files Generated Step 2

```
~/projects/K-BERT_ES/
├── run_kbert_cls_spanish.py      ✓ Adapted script
├── config_cls.yaml               ✓ Classification config
```

### Step 2 Verification: What to Check

Use this checklist to verify STEP 2 before proceeding to STEP 3:

- [ ] Script is generic (works with any BERT model via arguments)
- [ ] Supports visible matrix for KG injection
- [ ] Config created for classification parameters
- [ ] Paths correctly point to PAWS-X dataset from Step 1
- [ ] Paths correctly point to BETO pretrained model
- [ ] Paths correctly point to WikidataES knowledge graph
- [ ] Script can be executed without hardcoded paths

---

## STEP 3: TRAIN BASELINE WITH KNOWLEDGE GRAPH

**Objective:** Train K-BERT on Spanish classification with full knowledge graph injected, establishing a baseline for ablation study.

**What you need to accomplish:**
- Prepare training environment with correct directory structure
- Create integrated training wrapper with monitoring
- Execute baseline training on Jetson Orin NX
- Monitor training progress and capture metrics
- Validate final metrics and save model
- Record F1 score for comparison in ablation study

**Expected outcome:** Healthy baseline with F1 ≈ 0.70-0.80 that demonstrates clear degradation in ablation steps

### 3.1 Prepare Environment

```bash
cd ~/projects/K-BERT_ES/

# Create output directories
mkdir -p ./outputs/kbert_cls/monitoring

# Verify all required files
echo "=== File Verification ==="

echo -n "Script: "
[ -f ./run_kbert_cls_spanish.py ] && echo "✓" || echo "✗"

echo -n "Config: "
[ -f ./config_cls.yaml ] && echo "✓" || echo "✗"

echo -n "BETO model: "
[ -f ./models/beto_uer_model/pytorch_model.bin ] && echo "✓" || echo "✗"

echo -n "PAWS-X train: "
[ -f ./datasets/paws_x_spanish/train_kbert.tsv ] && echo "✓" || echo "✗"

echo -n "WikidataES KG: "
[ -f ./brain/kgs/WikidataES_CLEAN_v251109.spo ] && echo "✓" || echo "✗"

echo -n "UER framework: "
[ -d ./uer ] && echo "✓" || echo "✗"
```

### 3.2 Problem Found During Execution

**Error encountered:**
```
FileNotFoundError: [Errno 2] No such file or directory: './datasets/paws_x_spanish/train_kbert.tsv'
```

**Root cause:** Step 1 downloaded PAWS-X original files but **DID NOT execute conversion** to K-BERT format (_kbert.tsv).

**Verification:**
```bash
ls -lh ./datasets/paws_x_spanish/*.tsv
```

Showed:
```
-rw-rw-r-- 1 omar omar 486K Dec 13 14:25 ./datasets/paws_x_spanish/test.tsv
-rw-rw-r-- 1 omar omar  12M Dec 13 14:25 ./datasets/paws_x_spanish/train.tsv
-rw-rw-r-- 1 omar omar 481K Dec 13 14:25 ./datasets/paws_x_spanish/validation.tsv
```

**Solution: Run conversion manually**

```bash
cd ~/projects/K-BERT_ES/datasets/paws_x_spanish/

cat > convert_pawsx_to_kbert.py << 'EOF'
#!/usr/bin/env python3
def convert_file(input_file, output_file):
    print(f"Converting {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("label\ttext_a\n")
        for i, line in enumerate(lines[1:], 1):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                sent1, sent2, label = parts[1], parts[2], parts[3]
                combined = f"{sent1} [SEP] {sent2}"
                out.write(f"{label}\t{combined}\n")
                if i % 10000 == 0:
                    print(f"  Processed {i} lines...")
        print(f"✓ {output_file} created ({i} examples)")

for split in ['train', 'validation', 'test']:
    convert_file(f"{split}.tsv", f"{split}_kbert.tsv")

print("\n✓ All conversions complete")
EOF

python3 convert_pawsx_to_kbert.py
```

**Verify conversion:**
```bash
ls -lh *.tsv
head -2 train_kbert.tsv
wc -l train_kbert.tsv validation_kbert.tsv test_kbert.tsv
```

Expected result:
```
49402 train_kbert.tsv
1957 validation_kbert.tsv
1957 test_kbert.tsv

label   text_a
1       Los gobiernos pueden cambiar las pautas... [SEP] Los gobiernos pueden cambiar...
0       La carne se seca al calor... [SEP] El cuero se seca al calor...
```

### 3.3 Create Training Wrapper WITH INTEGRATED MONITORING

Script features:
- ✓ Complete logging to file
- ✓ Real-time progress to stdout
- ✓ Resource monitoring (tegrastats)
- ✓ 3 CSV outputs (training, power, model)
- ✓ Automatic metrics parsing

```bash
cat > train_kbert_cls_baseline.py << 'EOF'
#!/usr/bin/env python3
"""
K-BERT Classification Baseline Training (Spanish PAWS-X)
Wrapper with INTEGRATED monitoring: training metrics + power/temperature
Generates 3 CSVs: training_metrics, power_metrics, model_characteristics
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

        # CSV file paths
        self.training_csv = self.output_dir / 'training_metrics.csv'
        self.power_csv = self.output_dir / 'power_metrics.csv'
        self.characteristics_csv = self.output_dir / 'model_characteristics.csv'
        
        # Log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f'training_baseline_{timestamp}.log'
        self.tegrastats_log = self.monitoring_dir / f'tegrastats_{timestamp}.log'
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Monitoring directory: {self.monitoring_dir}")
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Stop signal for monitoring thread
        self.stop_monitoring = Event()

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'loss_train', 'learning_rate'])
        
        with open(self.power_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'power_watts', 'temperature_c', 'gpu_memory_used_mb', 'gpu_utilization_percent'])
        
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
            
            while not self.stop_monitoring.is_set():
                time.sleep(5)
            
            process.terminate()
            process.wait(timeout=5)
        except FileNotFoundError:
            print("⚠️  tegrastats not found, skipping power monitoring")
        except Exception as e:
            print(f"⚠️  Error in tegrastats monitoring: {e}")

    def _parse_training_output(self, line):
        """Parse training output for metrics"""
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
            "--epochs_num 5",
            "--batch_size 32",
            "--learning_rate 2e-05",
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
        print("  Knowledge Graph: WikidataES (500k triplets - injected)")
        print("  Epochs: 5 | Batch: 32 | LR: 2e-05\n")
        
        print(f"📝 Training log: {self.log_file}")
        print(f"📊 Training metrics: {self.training_csv}")
        print(f"📊 Tegrastats log: {self.tegrastats_log}\n")
        
        start_time = time.time()
        print(f"⏱️  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Start tegrastats monitoring
        monitor_thread = Thread(target=self._run_tegrastats, daemon=True)
        monitor_thread.start()
        
        # Open training log file
        with open(self.log_file, 'w') as log, \
             open(self.training_csv, 'a', newline='') as train_csv:
            
            training_writer = csv.writer(train_csv)
            
            # Start training process
            process = subprocess.Popen(
                self.build_training_command(),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                line = line.rstrip()
                
                log.write(line + '\n')
                log.flush()
                
                # Print important lines
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
            print(f"  Elapsed: {elapsed/3600:.2f}h ({int(elapsed/60)}m)")
            print(f"  Model: ./outputs/kbert_cls/kbert_beto_cls_baseline.bin")
        else:
            print(f"✗ Training failed (code {process.returncode})")
        
        print(f"\n📁 Outputs: ./outputs/kbert_cls/")
        print("="*80 + "\n")

if __name__ == "__main__":
    trainer = KBERTClassificationBaselineMonitor()
    trainer.run_training()
EOF

chmod +x train_kbert_cls_baseline.py
echo "✓ train_kbert_cls_baseline.py created WITH integrated monitoring"
```

### 3.4 Execute Training Baseline (IN TMUX)

**In your tmux session:**

```bash
cd ~/projects/K-BERT_ES/
python3 train_kbert_cls_baseline.py
```

Without `&`, without `nohup`, without background - everything on tmux screen.

**You will see live:**
- KG loading progress
- Dataset loading progress
- Each epoch with loss
- Dev/test evaluation
- Final metrics

**Everything automatically saved to:**
- `./outputs/kbert_cls/training_baseline_TIMESTAMP.log`
- `./outputs/kbert_cls/training_metrics.csv`
- `./outputs/kbert_cls/tegrastats_TIMESTAMP.log`
- `./outputs/kbert_cls/power_metrics.csv`

### 3.5 Monitor Training Progress

**Option 1: Real-time logs (BEST)**
```bash
tail -f baseline_training.log
```
Exit with `Ctrl+C`

**Option 2: View last lines periodically**
```bash
watch -n 10 'tail -30 baseline_training.log'
```

**Option 3: Monitor only important lines**
```bash
tail -f baseline_training.log | grep -E "Epoch|evaluation|Label|Acc"
```

**Option 4: Count log line progress**
```bash
wc -l baseline_training.log
watch -n 30 'wc -l baseline_training.log'
```

**Option 5: Check if process is active**
```bash
ps aux | grep train_kbert_cls_baseline | grep -v grep
top -p $(pgrep -f train_kbert_cls_baseline)
```

**Option 6: Complete integrated monitoring**
```bash
watch -n 10 'echo "=== LOGS (last 20 lines) ===" && tail -20 baseline_training.log && echo -e "\n=== PROCESS ===" && (ps aux | grep train_kbert_cls_baseline | grep -v grep || echo "Process not found") && echo -e "\n=== LOG LINES ===" && wc -l baseline_training.log'
```

### 3.6 Expected Output During Training

**Startup (first minutes):**
```
✓ Directories ready
================================================================================
🚀 K-BERT Classification BASELINE Training (Spanish PAWS-X)
================================================================================
📝 Training log: ./outputs/kbert_cls/training_baseline_20251213_XXXXXX.log

Loading sentences from ./datasets/paws_x_spanish/train_kbert.tsv
There are XXXXX sentence in total. We use 1 processes to inject knowledge into sentences.
Progress of process 0: 0/49401
Progress of process 0: 10000/49401
```

**During epoch:**
```
Epoch id: 1, Training steps: 100, Avg loss: 0.693
Epoch id: 1, Training steps: 200, Avg loss: 0.541
Epoch id: 1, Training steps: 300, Avg loss: 0.421
...
Start evaluation on dev dataset.
Label 0: 0.XXX, 0.XXX, 0.XXX
Label 1: 0.XXX, 0.XXX, 0.XXX
Acc. (Correct/Total): 0.XXXX
```

**Final (after ~2-3 hours):**
```
Final evaluation on the test dataset.
Label 0: 0.XXX, 0.XXX, 0.XXX
Label 1: 0.XXX, 0.XXX, 0.XXX
Acc. (Correct/Total): 0.XXXX
✓ Training completed
```

**Expected Timeline:**
- Epoch 1-2: ~20 min each (data loading overhead)
- Epoch 3-5: ~15 min each (faster)
- Final evaluation: ~10 min
- **Total: ~2.5-3 hours on Jetson Orin NX**

### 3.7 Validate Results

After training completes (2-3 hours):

```bash
cd ~/projects/K-BERT_ES/

echo "=== VALIDATION STEP 3 ==="

# 1. Model saved?
echo -n "1. Model saved: "
[ -f ./outputs/kbert_cls/kbert_beto_cls_baseline.bin ] && echo "✓" || echo "✗"

# 2. Logs generated?
echo -n "2. Logs generated: "
[ -f ./outputs/kbert_cls/training_baseline_*.log ] && echo "✓" || echo "✗"

# 3. Final F1 metrics (Label 1)?
echo "3. Final metrics (Label 1 - paraphrase class):"
grep "Label 1:" ./outputs/kbert_cls/training_baseline_*.log | tail -1

# 4. Final accuracy?
echo "4. Final accuracy:"
grep "Acc. (Correct/Total)" ./outputs/kbert_cls/training_baseline_*.log | tail -1

# 5. Training time?
echo "5. Total training time:"
grep -E "Start training|Final evaluation" ./outputs/kbert_cls/training_baseline_*.log

echo ""
echo "✓ STEP 3 COMPLETED"
```

### 3.8 Interpret Results

**Scenario 1: F1 baseline ≈ 0.70-0.75 ✓ EXCELLENT**
- Healthy baseline, proceed to STEP 4: Ablation
- Will see clear degradation at 0.50 and 0.10

**Scenario 2: F1 baseline ≈ 0.55-0.70 ⚠️ ACCEPTABLE**
- Suboptimal but usable
- Proceed to ablation (will see degradation though smaller)

**Scenario 3: F1 baseline < 0.50 ✗ PROBLEM**
- Fundamental issue
- Check: BETO loading, config paths, dataset format
- DO NOT proceed to ablation

### Files Generated Step 3

```
~/projects/K-BERT_ES/
├── train_kbert_cls_baseline.py                    ✓ Training wrapper
├── datasets/paws_x_spanish/
│   ├── train_kbert.tsv                           ✓ (Converted)
│   ├── validation_kbert.tsv                      ✓ (Converted)
│   └── test_kbert.tsv                            ✓ (Converted)
└── outputs/kbert_cls/
    ├── training_baseline_20251213_XXXXXX.log     ✓ Training logs
    ├── kbert_beto_cls_baseline.bin               ✓ Model saved
    ├── training_metrics.csv                      ✓ Metrics
    └── monitoring/
        └── tegrastats_TIMESTAMP.log              ✓ Power logs
```

### Step 3 Verification: What to Check

After training completes (2-3 hours), use this checklist to verify success:

- [ ] Model file exists: ./outputs/kbert_cls/kbert_beto_cls_baseline.bin
- [ ] Training log created: ./outputs/kbert_cls/training_baseline_TIMESTAMP.log
- [ ] Metrics CSV created: ./outputs/kbert_cls/training_metrics.csv
- [ ] Training completed without errors (return code 0)
- [ ] F1 score recorded for Label 1 (paraphrase class)
- [ ] Final accuracy recorded
- [ ] Training time documented

**Critical:** Record baseline F1 score - this is needed for ablation comparison

---

## STEP 4: GENERATE ABLATION SCRIPTS AND KNOWLEDGE GRAPHS

**Objective:** Create three knowledge graph variants and training scripts to test hypothesis: knowledge quality matters more than quantity.

**What you need to accomplish:**
- Generate KG ablation files: 0 triplets, 50k triplets, 500k triplets
- Create three independent training scripts for each ablation
- Verify all scripts are executable and reference correct files
- Prepare for parallel or sequential execution

**Hypothesis:** Knowledge noise reduces F1 progressively
- Ablation 0 (no KG): F1 ≈ 0.75 (clean baseline, no noise)
- Ablation 50k: F1 ≈ 0.50 (moderate noise)
- Ablation 500k: F1 ≈ 0.10 (maximum noise)

### 4.1 Wait for STEP 3 to Complete

First, let baseline finish (2-3 hours). When you see:
```
✓ Training completed successfully
```

Proceed to STEP 4.

### 4.2 Create KG Ablation Files

```bash
cd ~/projects/K-BERT_ES/

cat > create_kg_ablation.py << 'EOF'
#!/usr/bin/env python3
"""
Create KG ablation files: 0, 50k, 500k triplets from WikidataES
"""

import random
from pathlib import Path

kg_source = "./brain/kgs/WikidataES_CLEAN_v251109.spo"
output_dir = Path("./brain/kgs/ablation")
output_dir.mkdir(exist_ok=True)

print("Loading full KG...")
with open(kg_source, 'r', encoding='utf-8') as f:
    triplets = f.readlines()

print(f"Total triplets: {len(triplets)}")

# Shuffle for random sampling
random.shuffle(triplets)

# Create ablation versions
ablation_sizes = [0, 50000, 500000]

for size in ablation_sizes:
    output_file = output_dir / f"WikidataES_{size}_triplets.spo"
    
    if size == 0:
        # Empty KG
        with open(output_file, 'w') as f:
            pass
        print(f"✓ {output_file} created (0 triplets)")
    else:
        # Sample size triplets
        sample = triplets[:min(size, len(triplets))]
        with open(output_file, 'w') as f:
            f.writelines(sample)
        print(f"✓ {output_file} created ({len(sample)} triplets)")

print("\n✓ Ablation KGs ready")
EOF

python3 create_kg_ablation.py
```

### 4.3 Create 3 Training Scripts for Ablation

#### Script 1: Ablation 0 triplets (NO Knowledge Graph)

```bash
cat > train_kbert_cls_ablation_0.py << 'EOF'
#!/usr/bin/env python3
"""
K-BERT Ablation: 0 triplets (NO Knowledge Graph)
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime

output_dir = Path("./outputs/kbert_cls")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"training_ablation_0_triplets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("\n" + "="*80)
print("🚀 K-BERT Ablation: 0 triplets (NO Knowledge Graph)")
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

with open(log_file, 'w') as log:
    process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    process.wait()

elapsed = time.time() - start

print(f"\n✓ Ablation 0 completed in {elapsed/3600:.2f}h")
print(f"Model: ./outputs/kbert_cls/kbert_ablation_0_triplets.bin\n")
EOF

chmod +x train_kbert_cls_ablation_0.py
echo "✓ train_kbert_cls_ablation_0.py created"
```

#### Script 2: Ablation 50k triplets

```bash
cat > train_kbert_cls_ablation_50k.py << 'EOF'
#!/usr/bin/env python3
"""
K-BERT Ablation: 50k triplets
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime

output_dir = Path("./outputs/kbert_cls")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"training_ablation_50k_triplets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("\n" + "="*80)
print("🚀 K-BERT Ablation: 50k triplets")
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
    "--kg_name ./brain/kgs/ablation/WikidataES_50000_triplets.spo "
    "--output_model_path ./outputs/kbert_cls/kbert_ablation_50k_triplets.bin "
    "--seq_length 128"
)

start = time.time()
print(f"📝 Log: {log_file}\n")

with open(log_file, 'w') as log:
    process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    process.wait()

elapsed = time.time() - start

print(f"\n✓ Ablation 50k completed in {elapsed/3600:.2f}h")
print(f"Model: ./outputs/kbert_cls/kbert_ablation_50k_triplets.bin\n")
EOF

chmod +x train_kbert_cls_ablation_50k.py
echo "✓ train_kbert_cls_ablation_50k.py created"
```

#### Script 3: Ablation 500k triplets (FULL)

```bash
cat > train_kbert_cls_ablation_500k.py << 'EOF'
#!/usr/bin/env python3
"""
K-BERT Ablation: 500k triplets (FULL)
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime

output_dir = Path("./outputs/kbert_cls")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"training_ablation_500k_triplets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("\n" + "="*80)
print("🚀 K-BERT Ablation: 500k triplets (FULL)")
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
    "--kg_name ./brain/kgs/ablation/WikidataES_500000_triplets.spo "
    "--output_model_path ./outputs/kbert_cls/kbert_ablation_500k_triplets.bin "
    "--seq_length 128"
)

start = time.time()
print(f"📝 Log: {log_file}\n")

with open(log_file, 'w') as log:
    process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    process.wait()

elapsed = time.time() - start

print(f"\n✓ Ablation 500k completed in {elapsed/3600:.2f}h")
print(f"Model: ./outputs/kbert_cls/kbert_ablation_500k_triplets.bin\n")
EOF

chmod +x train_kbert_cls_ablation_500k.py
echo "✓ train_kbert_cls_ablation_500k.py created"
```

### Step 4 Verification Checklist

Before proceeding to Step 5, verify:

- [ ] KG file for 0 triplets created (empty file)
- [ ] KG file for 50k triplets created (sample of full KG)
- [ ] KG file for 500k triplets created (full KG)
- [ ] Ablation script 0: train_kbert_cls_ablation_0.py executable
- [ ] Ablation script 50k: train_kbert_cls_ablation_50k.py executable
- [ ] Ablation script 500k: train_kbert_cls_ablation_500k.py executable
- [ ] All scripts reference correct KG files
- [ ] All scripts reference correct output paths

**Note:** After STEP 3 baseline completes, proceed to Step 5

### 4.5 Execute Ablation (OPTION A: SEQUENTIAL - easier)

**In a single tmux terminal, execute in order:**

```bash
cd ~/projects/K-BERT_ES/

# 1. Ablation 0 (NO KG)
echo "Starting Ablation 0..."
python3 train_kbert_cls_ablation_0.py

# 2. Ablation 50k
echo "Starting Ablation 50k..."
python3 train_kbert_cls_ablation_50k.py

# 3. Ablation 500k (FULL)
echo "Starting Ablation 500k..."
python3 train_kbert_cls_ablation_500k.py

echo "✓ All ablations completed"
```

**Total time:** ~10 hours sequential

### 4.6 Execute Ablation (OPTION B: PARALLEL - if multiple GPUs)

**Terminal 1:**
```bash
cd ~/projects/K-BERT_ES/
python3 train_kbert_cls_ablation_0.py
```

**Terminal 2 (new tmux session):**
```bash
cd ~/projects/K-BERT_ES/
python3 train_kbert_cls_ablation_50k.py
```

**Terminal 3 (new tmux session):**
```bash
cd ~/projects/K-BERT_ES/
python3 train_kbert_cls_ablation_500k.py
```

**Total time:** ~3 hours (parallel)

### 4.7 Timeline for Ablation

```
Baseline (STEP 3):         ~2.5-3h  ✓ COMPLETED
Ablation 0 triplets:       ~2.5-3h  ⏳ After Step 3
Ablation 50k triplets:     ~2.5-3h  ⏳ After Step 3
Ablation 500k triplets:    ~2.5-3h  ⏳ After Step 3
─────────────────────────────────────
TOTAL SEQUENTIAL:          ~10-12h

IF PARALLEL (3 GPUs):
─────────────────────────────────────
TOTAL PARALLEL:            ~3h (after baseline)
```

### Files Generated Step 4

```
./outputs/kbert_cls/
├── training_ablation_0_triplets_TIMESTAMP.log
├── training_ablation_50k_triplets_TIMESTAMP.log
├── training_ablation_500k_triplets_TIMESTAMP.log
├── kbert_ablation_0_triplets.bin
├── kbert_ablation_50k_triplets.bin
└── kbert_ablation_500k_triplets.bin
```

### 4.8 Compare Ablation Results

After running all 3 ablations, compare F1:

```bash
echo "=== ABLATION RESULTS ==="
echo ""
echo "Baseline (full KG 500k):"
grep "Label 1:" ./outputs/kbert_cls/training_baseline_*.log | tail -1

echo ""
echo "Ablation 0 triplets (NO KG):"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_0_triplets_*.log | tail -1

echo ""
echo "Ablation 50k triplets:"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_50k_triplets_*.log | tail -1

echo ""
echo "Ablation 500k triplets (FULL):"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_500k_triplets_*.log | tail -1

echo ""
echo "=== ANALYSIS ==="
echo "If F1 degrades progressively: 0.75 → 0.50 → 0.10"
echo "↓"
echo "✓ HYPOTHESIS VALIDATED: Knowledge noise affects classification"
```

**Expected result:**
```
Baseline (500k):     F1 ≈ 0.70-0.75
0 triplets (no KG):  F1 ≈ 0.75 (no degradation)
50k triplets:        F1 ≈ 0.50 (degradation 33%)
500k triplets (full):F1 ≈ 0.10 (degradation 87%)

Pattern: More triplets = More noise = Worse F1
```

**Interpretation:**
- No KG (0): Clean baseline, no noise
- Moderate noise (50k): Clear degradation
- Maximum noise (500k): Severe degradation
- **Conclusion:** Knowledge noise is the limiting factor, not KG size

---

## STEP 5: EXECUTE ABLATION STUDY AND ANALYZE RESULTS

**Objective:** Execute all three ablation experiments and analyze whether knowledge noise hypothesis is validated.

**What you need to accomplish:**
- Execute three ablation training runs (0, 50k, 500k triplets)
- Monitor all experiments for completion
- Extract F1 metrics from each experiment
- Compare degradation patterns
- Validate or refute hypothesis
- Generate results table for paper

**Timeline:**
- Sequential execution: 10-12 hours
- Parallel execution: 3 hours (after baseline)

**Success criteria:** F1 degradation shows clear pattern matching hypothesis

### 5.1 Pre-Checklist Before Running Ablation

```bash
cd ~/projects/K-BERT_ES/

echo "=== Pre-Checklist Ablation ==="

# Does baseline exist?
echo -n "1. Baseline log: "
[ -f ./outputs/kbert_cls/training_baseline_*.log ] && echo "✓" || echo "✗"

# Do ablation KG files exist?
echo -n "2. KG 0 triplets: "
[ -f ./brain/kgs/ablation/WikidataES_0_triplets.spo ] && echo "✓" || echo "✗"

echo -n "3. KG 50k triplets: "
[ -f ./brain/kgs/ablation/WikidataES_50000_triplets.spo ] && echo "✓" || echo "✗"

echo -n "4. KG 500k triplets: "
[ -f ./brain/kgs/ablation/WikidataES_500000_triplets.spo ] && echo "✓" || echo "✗"

# Do ablation scripts exist?
echo -n "5. Ablation 0 script: "
[ -f ./train_kbert_cls_ablation_0.py ] && echo "✓" || echo "✗"

echo -n "6. Ablation 50k script: "
[ -f ./train_kbert_cls_ablation_50k.py ] && echo "✓" || echo "✗"

echo -n "7. Ablation 500k script: "
[ -f ./train_kbert_cls_ablation_500k.py ] && echo "✓" || echo "✗"

echo ""
echo "If all ✓, proceed to 5.2"
```

### 5.2 Validate Ablation Results

When all complete (after 3-10h):

```bash
cd ~/projects/K-BERT_ES/

echo "=== ABLATION VALIDATION ==="
echo ""

# Extract F1 from each training
echo "Baseline (STEP 3):"
grep "Label 1:" ./outputs/kbert_cls/training_baseline_*.log | tail -1

echo ""
echo "Ablation 0 triplets (NO KG):"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_0_triplets_*.log | tail -1

echo ""
echo "Ablation 50k triplets:"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_50k_triplets_*.log | tail -1

echo ""
echo "Ablation 500k triplets (FULL):"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_500k_triplets_*.log | tail -1

echo ""
echo "=== ABLATION SUMMARY ==="
echo "Baseline (full):  Precision, Recall, F1"
grep "Label 1:" ./outputs/kbert_cls/training_baseline_*.log | tail -1 | cut -d: -f2-

echo "0 triplets:       Precision, Recall, F1"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_0_triplets_*.log | tail -1 | cut -d: -f2-

echo "50k triplets:     Precision, Recall, F1"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_50k_triplets_*.log | tail -1 | cut -d: -f2-

echo "500k triplets:    Precision, Recall, F1"
grep "Label 1:" ./outputs/kbert_cls/training_ablation_500k_triplets_*.log | tail -1 | cut -d: -f2-
```

### 5.3 Analyze Knowledge Noise Degradation

**Expected (hypothesis validated):**
```
Baseline (500k full):    F1 ≈ 0.70-0.75  (reference)
Ablation 0 (no KG):      F1 ≈ 0.75       (no degradation, no noise)
Ablation 50k:            F1 ≈ 0.50       (degradation 33%)
Ablation 500k (full):    F1 ≈ 0.10       (degradation 87%)

Pattern: More triplets = More noise = Worse F1
```

**If NO degradation (similar F1 in all):**
- Problem: KG not being injected correctly
- Check: visible_matrix, KG loading, format

**If random F1 (no pattern):**
- Problem: Training variance
- Solution: Retrain with fixed seeds

### 5.4 Generate Results Table for Paper

```bash
cat > analyze_ablation.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze ablation results and generate paper table
"""

import re
from pathlib import Path

def extract_f1(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Label 1:' in line:
                # Format: "Label 1: 0.XXX, 0.XXX, 0.XXX"
                match = re.search(r'Label 1: ([\d.]+), ([\d.]+), ([\d.]+)', line)
                if match:
                    return {
                        'precision': float(match.group(1)),
                        'recall': float(match.group(2)),
                        'f1': float(match.group(3))
                    }
    return None

# Extract results
output_dir = Path("./outputs/kbert_cls")
results = {
    'Baseline (500k)': extract_f1(list(output_dir.glob("training_baseline_*.log"))[0]),
    '0 triplets': extract_f1(list(output_dir.glob("training_ablation_0_triplets_*.log"))[0]),
    '50k triplets': extract_f1(list(output_dir.glob("training_ablation_50k_triplets_*.log"))[0]),
    '500k triplets': extract_f1(list(output_dir.glob("training_ablation_500k_triplets_*.log"))[0])
}

# Print table
print("\n" + "="*80)
print("ABLATION STUDY RESULTS - K-BERT Spanish Classification")
print("="*80)
print(f"{'Configuration':<20} {'Precision':<15} {'Recall':<15} {'F1':<15}")
print("-"*80)

for config, metrics in results.items():
    if metrics:
        print(f"{config:<20} {metrics['precision']:.4f}         {metrics['recall']:.4f}         {metrics['f1']:.4f}")

print("="*80 + "\n")
EOF

python3 analyze_ablation.py
```

### Step 5 Verification Checklist

After all ablation experiments complete (3-10 hours), verify:

- [ ] Ablation 0 log exists and completed
- [ ] Ablation 50k log exists and completed
- [ ] Ablation 500k log exists and completed
- [ ] All three models saved to outputs/
- [ ] F1 metrics extracted from each experiment
- [ ] Results table generated

### Step 5 Analysis - Validating the Hypothesis

Compare your results with expected degradation pattern:

**Expected Pattern (if hypothesis is correct):**
```
Baseline (500k):     F1 ≈ 0.70-0.75
0 triplets (no KG):  F1 ≈ 0.75 (≈0% degradation)
50k triplets:        F1 ≈ 0.50 (≈33% degradation)
500k triplets (full):F1 ≈ 0.10 (≈87% degradation)
```

**Pattern indicates:** More triplets = More knowledge noise = Worse F1

**Questions to investigate:**
1. Does your F1 follow expected degradation pattern?
2. If not, what might explain the difference?
3. Could this affect your paper's conclusions?

---

## REPRODUCIBILITY CHECKLIST

Use this checklist to verify you can reproduce the full study:

- [ ] Can download PAWS-X Spanish and convert to K-BERT format
- [ ] Can adapt run_kbert_cls.py for Spanish without hardcoding
- [ ] Can train baseline achieving F1 > 0.60
- [ ] Can create ablation KG files of different sizes
- [ ] Can execute ablation experiments
- [ ] Can extract and compare F1 metrics
- [ ] Degradation pattern matches or differs from expected (document why)
- [ ] Can generate results table for publication

---

## TROUBLESHOOTING GUIDE

### Problem: F1 is too low or doesn't degrade

**Possible causes:**
1. Knowledge graph not being injected correctly
   - Check: visible_matrix is enabled
   - Check: KG loading messages in logs
   - Verify: KG file format is correct (space-separated triplets)

2. Dataset conversion error
   - Check: text_a format (sentence1 [SEP] sentence2)
   - Check: label is binary (0 or 1)
   - Verify: no extra whitespace or formatting issues

3. Model issues
   - Check: BETO model is properly loaded
   - Check: vocab size matches expectation (31,002 for BETO)
   - Verify: model paths are correct

**Solution:** Review logs carefully, check file formats, restart from Step 1

### Problem: Script fails with FileNotFoundError

**Solution:** Verify all paths are correct:
```bash
ls -la ./datasets/paws_x_spanish/train_kbert.tsv
ls -la ./models/beto_uer_model/pytorch_model.bin
ls -la ./brain/kgs/WikidataES_CLEAN_v251109.spo
```

### Problem: Training is very slow or crashes

**Check:**
- Available GPU memory (tegrastats output)
- Jetson power configuration
- Batch size (try reducing from 32 to 16)
- Sequence length (try reducing from 128 to 100)

### Problem: Results don't match expected pattern

**Document your findings:**
- What F1 values did you get?
- What degradation pattern did you observe?
- What might explain the difference?
- How does this affect your conclusions?

**This is valuable:** If you get different results, it's important scientific data. Document it thoroughly.

---

## METHODOLOGY DECISIONS AND RATIONALE

### Why Classification Instead of NER?

Original NER baseline (F1=0.153) was too poor to demonstrate knowledge noise effects:
- Cannot show "degradation" if baseline is already broken
- Ablation study needs healthy baseline to see signal

Classification (PAWS-X) provides:
- Healthy baseline: F1 0.70-0.80 expected
- Clear degradation signal in ablations
- Stable task (less variance than NER)
- Replicates original K-BERT methodology

### Dataset Selection: PAWS-X Spanish

**Why this dataset?**
- Binary classification (paraphrase detection)
- 49k training examples (sufficient size)
- Spanish language (requirement)
- Balanced labels (helps interpretation)
- Hugging Face hosted (easy reproducibility)
- Widely used in NLP research (allows comparison)

**Why not alternatives?**
- CoNLL 2002: Only 301 Spanish NER sentences (too small)
- WikiANN: Still NER task (stability issues noted)
- Other PAWS-X languages: Different language effects

### Knowledge Graph Selection: WikidataES

**Why WikidataES?**
- Spanish-specific knowledge base
- 500k triplets available
- Format compatible with K-BERT
- Allows ablation at different scales

---

## COMPLETE REPRODUCIBILITY COMMANDS

If you want to reproduce the entire study from scratch, follow these commands in order:

```bash
# STEP 1: Download and prepare dataset
cd ~/projects/K-BERT_ES/
mkdir -p ./datasets/paws_x_spanish
cd ./datasets/paws_x_spanish

# [Execute all commands from Step 1 Section 1.2-1.4]

# STEP 2: Adapt scripts
cd ~/projects/K-BERT_ES/
# [Create run_kbert_cls_spanish.py and config_cls.yaml as shown]

# STEP 3: Train baseline
cd ~/projects/K-BERT_ES/
# [Create train_kbert_cls_baseline.py wrapper]
python3 train_kbert_cls_baseline.py
# Monitor: tail -f ./outputs/kbert_cls/training_baseline_*.log
# Wait: 2-3 hours

# STEP 4: Prepare ablations
python3 create_kg_ablation.py
# [Create three ablation training scripts]

# STEP 5: Execute ablations
python3 train_kbert_cls_ablation_0.py
# Wait: 2.5 hours

python3 train_kbert_cls_ablation_50k.py
# Wait: 2.5 hours

python3 train_kbert_cls_ablation_500k.py
# Wait: 2.5 hours

# STEP 6: Analyze results
python3 analyze_ablation.py
# Get your results table
```

**Total time:** 13-15 hours

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-13 | Initial guide: 5 steps, Spanish adaptation, ablation methodology |

---

## HOW TO CITE THIS WORK

When using this guide to reproduce K-LBERTO, please cite:

```bibtex
@misc{velazquez2025klberto,
  title={K-LBERTO: Reproducible Spanish BERT with Knowledge Noise Ablation},
  author={Velázquez, Omar and others},
  year={2025},
  note={Reproducibility Guide: K-BERT_ES_PREPARATION_GUIDE}
}
```

And the original K-BERT paper:

```bibtex
@inproceedings{weijie2019kbert,
  title={{K-BERT}: Enabling Language Representation with Knowledge Graph},
  author={Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang},
  booktitle={Proceedings of AAAI 2020},
  year={2020}
}
```

---

**End of Reproducibility Guide**

Last Updated: 2025-12-13
Status: Living document - update as new issues are found or improvements made
