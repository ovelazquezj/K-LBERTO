#!/bin/bash
################################################################################
# PASO 5 v6: TRAINING CON CAPTURA DE MÉTRICAS
# 
# Objetivo: Entrenar K-BERT con weights balanceados por clase
# Captura: Logits, predicciones, loss, accuracy, confusion matrix, F1
# 
# Bugs arreglados:
#   - Segment embedding [0,0,0...]
#   - Position index pos_idx_tree[i][0]
#   - Class imbalance con NLLLoss(weight=class_weights)
#
# Cambios v6:
#   - Captura de DEBUG EMBEDDINGS, ENCODER OUTPUT, LOGITS
#   - Captura de loss durante entrenamiento
#   - Captura de matriz de confusión
#   - Captura de F1 por clase
#   - Análisis post-training
#
# Date: Diciembre 27, 2025 23:55 UTC
################################################################################

set -e

cd ~/projects/K-LBERTO

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./resultados"
mkdir -p "$LOG_DIR"

# Archivos de salida
FULL_LOG="$LOG_DIR/paso5_v6_training_${TIMESTAMP}.log"
METRICS_SUMMARY="$LOG_DIR/paso5_v6_metrics_${TIMESTAMP}.txt"
DEBUG_LOG="$LOG_DIR/paso5_v6_debug_${TIMESTAMP}.log"
LOSS_CSV="$LOG_DIR/paso5_v6_loss_${TIMESTAMP}.csv"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         PASO 5 v6: TRAINING CON MÉTRICAS CAPTURADAS       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Full log:       $FULL_LOG"
echo "Metrics:        $METRICS_SUMMARY"
echo "Debug:          $DEBUG_LOG"
echo "Loss CSV:       $LOSS_CSV"
echo ""
echo "CONFIGURACIÓN:"
echo "  Epochs:        5"
echo "  Batch size:    8"
echo "  Learning rate: 5e-05"
echo "  Seq length:    128"
echo "  Class weights: Inverse frequency"
echo ""
echo "INICIO: $(date)"
echo ""

# Iniciar entrenamiento y capturar TODAS las salidas
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
  --output_model_path ./outputs/kbert_tass_balanced_final.bin \
  --workers_num 4 2>&1 | tee "$FULL_LOG"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ANÁLISIS DE RESULTADOS                        ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Extraer DEBUG logs
{
    echo "═════════════════════════════════════════════════════════"
    echo "DEBUG LOGS - Primeros 50 ejemplos (evaluación)"
    echo "═════════════════════════════════════════════════════════"
    echo ""
    grep -A 20 "DEBUG EMBEDDINGS" "$FULL_LOG" | head -100
    echo ""
    echo "═════════════════════════════════════════════════════════"
    echo "POST-OUTPUT_LAYER_2 (Logits finales)"
    echo "═════════════════════════════════════════════════════════"
    echo ""
    grep "DEBUG POST-OUTPUT_LAYER_2" -A 2 "$FULL_LOG" | head -60
} > "$DEBUG_LOG"

# Extraer LOSS durante entrenamiento
{
    echo "epoch,step,avg_loss"
    grep "Training steps:" "$FULL_LOG" | sed 's/.*Epoch id: \([0-9]*\).* Training steps: \([0-9]*\).* Avg loss: \([0-9.]*\).*/\1,\2,\3/'
} > "$LOSS_CSV"

# Extraer métricas finales
{
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              MÉTRICAS FINALES - RESUMEN                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo "Fecha: $(date)"
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "CONFIGURACIÓN DEL ENTRENAMIENTO"
    echo "════════════════════════════════════════════════════════"
    echo "  Epochs: 5"
    echo "  Batch size: 8"
    echo "  Learning rate: 5e-05"
    echo "  Seq length: 128"
    echo "  Optimizer: BertAdam"
    echo "  Loss function: NLLLoss con class_weights"
    echo "  Class weights: [inverse frequency de cada clase]"
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "DATASET"
    echo "════════════════════════════════════════════════════════"
    echo "  Train: 900 samples (Clase 0: 41.6%, 1: 31.3%, 2: 12.5%, 3: 14.4%)"
    echo "  Test: 225 samples (misma distribución)"
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "LOSS DURANTE ENTRENAMIENTO"
    echo "════════════════════════════════════════════════════════"
    grep "Training steps:" "$FULL_LOG" | tail -5
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "MATRIZ DE CONFUSIÓN - FINAL"
    echo "════════════════════════════════════════════════════════"
    grep -A 5 "Confusion matrix:" "$FULL_LOG" | tail -8
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "F1 SCORES POR CLASE - FINAL"
    echo "════════════════════════════════════════════════════════"
    grep "Label.*:" "$FULL_LOG" | tail -6
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "ACCURACY FINAL"
    echo "════════════════════════════════════════════════════════"
    grep "Acc\. (Correct/Total):" "$FULL_LOG" | tail -3
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "LOGITS EN EVALUACIÓN (Primeros 3 ejemplos)"
    echo "════════════════════════════════════════════════════════"
    grep "DEBUG POST-OUTPUT_LAYER_2" -A 2 "$FULL_LOG" | head -9
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "ANÁLISIS"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "✓ Logits deberían variar significativamente entre ejemplos"
    echo "  (diferencia > 0.5 después de 5 epochs)"
    echo ""
    echo "✓ Modelo debería hacer predicciones en múltiples clases"
    echo "  (no solo clase 0)"
    echo ""
    echo "✓ F1 scores debería mejorar para clases 1, 2, 3"
    echo "  (no solo F1=0 como antes)"
    echo ""
    echo "✓ Accuracy debería superar baseline 41.6%"
    echo "  (o estar cerca si clase 0 es muy fuerte)"
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "ARCHIVOS GENERADOS"
    echo "════════════════════════════════════════════════════════"
    echo "  Full log:  $FULL_LOG"
    echo "  Debug:     $DEBUG_LOG"
    echo "  CSV loss:  $LOSS_CSV"
    echo ""
    
    echo "════════════════════════════════════════════════════════"
    echo "PRÓXIMOS PASOS"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "1. Revisar LOSS CSV para ver convergencia:"
    echo "   head -20 $LOSS_CSV"
    echo ""
    echo "2. Revisar logits finales:"
    echo "   tail -50 $DEBUG_LOG"
    echo ""
    echo "3. Si F1 > 0.45 para cualquier clase:"
    echo "   ✓ El modelo está aprendiendo"
    echo "   ✓ Pasar a PASO 6 (ablation studies)"
    echo ""
    echo "4. Si F1 aún bajo:"
    echo "   ✗ Necesita investigación adicional"
    echo "   → Revisar knowledge graph injection"
    echo "   → Revisar visible_matrix application"
    echo ""
    
} > "$METRICS_SUMMARY"

cat "$METRICS_SUMMARY"

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "TRAINING COMPLETADO"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Ver resultados:"
echo "  cat $METRICS_SUMMARY"
echo ""
echo "Ver loss progression:"
echo "  cat $LOSS_CSV"
echo ""
echo "Ver debug detallado:"
echo "  tail -100 $DEBUG_LOG"
echo ""
