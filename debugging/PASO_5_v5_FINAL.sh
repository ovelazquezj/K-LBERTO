#!/bin/bash
# PASO 5 v5: FINAL - TODOS LOS BUGS ARREGLADOS
# 
# Bugs arreglados:
#   v2: Segment embedding para single-sentence
#   v3: Special tags position agregado
#   v4: pos_idx prekalculado en lugar de [pos_idx + 1]
#
# Validación: positions correctas [0, 1, 2, ...]

TIMESTAMP=$(date +%s)
LOG_FILE="resultados/paso5_v5_final_${TIMESTAMP}.log"

echo "PASO 5 v5: FINAL - TODOS BUGS ARREGLADOS"
echo "Timestamp: $TIMESTAMP"
echo "Log: $LOG_FILE"

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
  --output_model_path ./outputs/kbert_tass_sentiment_88_SPANISH_v5_FINAL.bin \
  --workers_num 4 2>&1 | tee "$LOG_FILE"

echo ""
echo "PASO 5 v5 COMPLETADO"
echo "Resultados en: $LOG_FILE"
