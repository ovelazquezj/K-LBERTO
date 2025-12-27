#!/bin/bash

set -e

cd ~/projects/K-LBERTO

echo "=========================================="
echo "PASO 5 v2: K-BERT WITH METRICS LOGGING"
echo "=========================================="
echo ""

# Crear directorio para logs
mkdir -p ./logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/paso5_v2_${TIMESTAMP}.log"
METRICS_FILE="./logs/paso5_v2_metrics_${TIMESTAMP}.json"

echo "Logging to: $LOG_FILE"
echo "Metrics to: $METRICS_FILE"
echo ""

# Ejecutar con captura de salida
python3 run_kbert_cls.py \
  --pretrained_model_path ./models/beto_uer_model/pytorch_model.bin \
  --config_path ./models/beto_uer_model/config.json \
  --vocab_path ./models/beto_uer_model/vocab.txt \
  --train_path ./datasets/tass_spanish/train.tsv \
  --dev_path ./datasets/tass_spanish/test.tsv \
  --test_path ./datasets/tass_spanish/test.tsv \
  --kg_name ./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo \
  --epochs_num 5 \
  --batch_size 8 \
  --learning_rate 5e-05 \
  --seq_length 128 \
  --output_model_path ./outputs/kbert_tass_sentiment_88_SPANISH_v2.bin \
  --workers_num 4 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "EXTRAYENDO MÉTRICAS..."
echo "=========================================="

# Script Python para extraer métricas del log
python3 << 'PYTHON_EOF'
import json
import re
import sys
from datetime import datetime

log_file = sys.argv[1]
metrics_file = sys.argv[2]

metrics = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "version": "PASO_5_v2",
    "hyperparameters": {
        "learning_rate": 5e-05,
        "batch_size": 8,
        "epochs": 5,
        "seq_length": 128,
        "kg_triplets": 61
    },
    "results": {
        "train": {},
        "dev": {},
        "test": {}
    }
}

with open(log_file, 'r') as f:
    content = f.read()

# Extraer Confusion Matrix
conf_match = re.search(r'Confusion matrix:\ntensor\(\[(.*?)\]\)', content, re.DOTALL)
if conf_match:
    conf_str = conf_match.group(1)
    metrics["results"]["test"]["confusion_matrix"] = conf_str

# Extraer métricas por label (buscar todas las líneas "Label X: ...")
label_pattern = r'Label (\d+): ([\d.]+), ([\d.]+), ([\d.]+)'
for match in re.finditer(label_pattern, content):
    label = int(match.group(1))
    precision = float(match.group(2))
    recall = float(match.group(3))
    f1 = float(match.group(4))
    
    metrics["results"]["test"][f"label_{label}"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Extraer Accuracy
acc_pattern = r'Acc\. \(Correct/Total\): ([\d.]+) \((\d+)/(\d+)\)'
for match in re.finditer(acc_pattern, content):
    acc = float(match.group(1))
    correct = int(match.group(2))
    total = int(match.group(3))
    metrics["results"]["test"]["accuracy"] = {
        "value": acc,
        "correct": correct,
        "total": total
    }

# Guardar JSON
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Métricas guardadas: {metrics_file}")
print(f"\n📊 RESULTADOS FINALES:")
print(json.dumps(metrics["results"]["test"], indent=2))

PYTHON_EOF "$LOG_FILE" "$METRICS_FILE"

echo ""
echo "=========================================="
echo "✓ PASO 5 v2 COMPLETADO"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"
echo ""
