#!/bin/bash

# ════════════════════════════════════════════════════════════════════════════════
# PASO 5 v7 - K-BERT ENTRENAMIENTO CON FIXES APLICADOS
# ════════════════════════════════════════════════════════════════════════════════
#
# FIXES APLICADOS:
# FIX #1: Quitar view() redundante en loss calculation
# FIX #2: Agregar dropout en output layers
# FIX #3: Simplificar máscara (line 220)
# FIX #4: Agregar verificación de colapso a clase mayoritaria
# FIX #5: Aumentar epochs a 10
#
# CAMBIOS RESPECTO A v6:
# - 10 epochs vs 5 (dar más tiempo para que logits crezcan)
# - Dropout en output layers (reducir overfitting)
# - Verificación de colapso en evaluate()
# - Limpieza de código
#
# EXPECTED IMPROVEMENTS:
# ✓ Logits más grandes después de 10 epochs
# ✓ Predicciones en múltiples clases (no solo clase 0)
# ✓ F1 scores > 0 para clases 1, 2, 3
# ✓ Detección temprana si modelo sigue colapsando
#
# ════════════════════════════════════════════════════════════════════════════════

set -e

# Crear directorio de resultados si no existe
mkdir -p ~/projects/K-LBERTO/resultados

# Timestamp para archivos de resultado
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Directorios
PROJECT_DIR=~/projects/K-LBERTO
RESULTADOS_DIR=$PROJECT_DIR/resultados
SCRIPT_FIXED=$PROJECT_DIR/run_kbert_cls.py  # El script original será reemplazado
VOCAB_PATH=$PROJECT_DIR/models/beto_uer_model/vocab.txt
CONFIG_PATH=$PROJECT_DIR/models/beto_uer_model/config.json
PRETRAINED_MODEL=$PROJECT_DIR/models/beto_uer_model/pytorch_model.bin

# Dataset paths
TRAIN_PATH=$PROJECT_DIR/datasets/tass_spanish/train.tsv
DEV_PATH=$PROJECT_DIR/datasets/tass_spanish/test.tsv
TEST_PATH=$PROJECT_DIR/datasets/tass_spanish/test.tsv

# KG path (NUEVO: Generado específicamente para TASS con cobertura 1.83%)
KG_PATH=$PROJECT_DIR/brain/kgs/TASS_sentiment_KG_FINAL.spo

# Output model
OUTPUT_MODEL=$PROJECT_DIR/outputs/kbert_tass_v7_kg_new.bin

# Log files
FULL_LOG=$RESULTADOS_DIR/paso5_v7_training_${TIMESTAMP}.log
DEBUG_LOG=$RESULTADOS_DIR/paso5_v7_debug_${TIMESTAMP}.log
LOSS_CSV=$RESULTADOS_DIR/paso5_v7_loss_${TIMESTAMP}.csv
METRICS_FILE=$RESULTADOS_DIR/paso5_v7_metrics_${TIMESTAMP}.txt

echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 5 v7 - K-BERT CLASSIFICATION CON FIXES"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 CONFIGURACIÓN:"
echo "  Epochs: 10 (FIX #5)"
echo "  Batch size: 8"
echo "  Learning rate: 5e-05"
echo "  Dropout: 0.5 (FIX #2)"
echo "  Loss: NLLLoss con class_weights"
echo ""
echo "📂 RUTAS:"
echo "  Vocab: $VOCAB_PATH"
echo "  Config: $CONFIG_PATH"
echo "  Modelo pre-entrenado: $PRETRAINED_MODEL"
echo "  Dataset train: $TRAIN_PATH"
echo "  Dataset dev: $DEV_PATH"
echo "  Dataset test: $TEST_PATH"
echo "  KG (NUEVO): $KG_PATH"
echo ""
echo "📊 KG NUEVO GENERADO:"
echo "  Subjects: 79 únicos"
echo "  Triplets: 138 limpias"
echo "  Cobertura: 1.83% (vs 0.04% anterior = 45x mejor)"
echo "  [UNK] tokens: 0% (vs 40% anterior)"
echo "  Estado: Listo para inyección de conocimiento"
echo ""
echo "💾 RESULTADOS:"
echo "  Full log: $FULL_LOG"
echo "  Debug log: $DEBUG_LOG"
echo "  Loss CSV: $LOSS_CSV"
echo "  Metrics: $METRICS_FILE"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Iniciar tiempo
START_TIME=$(date +%s)

# Ejecutar K-BERT classification con redirección de output
# Captura stdout en full log, stderr en ambos logs
python3 $SCRIPT_FIXED \
    --pretrained_model_path $PRETRAINED_MODEL \
    --output_model_path $OUTPUT_MODEL \
    --vocab_path $VOCAB_PATH \
    --config_path $CONFIG_PATH \
    --train_path $TRAIN_PATH \
    --dev_path $DEV_PATH \
    --test_path $TEST_PATH \
    --kg_name $KG_PATH \
    --epochs_num 10 \
    --batch_size 8 \
    --seq_length 128 \
    --learning_rate 5e-05 \
    --dropout 0.5 \
    --report_steps 100 \
    2>&1 | tee $FULL_LOG

# Capturar el código de salida del comando anterior
EXIT_CODE=${PIPESTATUS[0]}

# Calcular tiempo total
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "EXTRAYENDO MÉTRICAS Y GENERANDO RESUMEN"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Crear archivo CSV de loss
echo "epoch,step,avg_loss" > $LOSS_CSV
grep "Epoch id:" $FULL_LOG | sed 's/.*Epoch id: \([0-9]*\).*Training steps: \([0-9]*\).*Avg loss: \([0-9.]*\).*/\1,\2,\3/' >> $LOSS_CSV

# Crear resumen de métricas
cat > $METRICS_FILE << EOF
════════════════════════════════════════════════════════════════════════════════
PASO 5 v7 - RESUMEN DE RESULTADOS
════════════════════════════════════════════════════════════════════════════════
Timestamp: $TIMESTAMP
Tiempo total de ejecución: ${MINUTES}m ${SECONDS}s
Código de salida: $EXIT_CODE

════════════════════════════════════════════════════════════════════════════════
CONFIGURACIÓN DEL ENTRENAMIENTO
════════════════════════════════════════════════════════════════════════════════
  Epochs: 10 (AUMENTADO de 5 para que crezcan logits)
  Batch size: 8
  Learning rate: 5e-05
  Seq length: 128
  Optimizer: BertAdam
  Loss function: NLLLoss con class_weights
  Dropout: 0.5 (AGREGADO a output layers)

════════════════════════════════════════════════════════════════════════════════
DATASET
════════════════════════════════════════════════════════════════════════════════
  Train: 900 samples
  Dev/Test: 225 samples cada uno
EOF

# Extraer loss progression
echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "LOSS DURANTE ENTRENAMIENTO (CSV)" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
cat $LOSS_CSV >> $METRICS_FILE

# Extraer confusion matrix y F1 scores
echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "MATRIZ DE CONFUSIÓN - FINAL" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
grep -A 20 "Confusion matrix:" $FULL_LOG | tail -10 >> $METRICS_FILE 2>/dev/null || echo "No se encontró matriz de confusión" >> $METRICS_FILE

echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "F1 SCORES POR CLASE - FINAL" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
grep "Label [0-3]:" $FULL_LOG | tail -4 >> $METRICS_FILE 2>/dev/null || echo "No se encontraron F1 scores" >> $METRICS_FILE

echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "ACCURACY FINAL" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
grep "Acc. (Correct/Total):" $FULL_LOG | tail -1 >> $METRICS_FILE 2>/dev/null || echo "No se encontró accuracy" >> $METRICS_FILE

echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "VERIFICACIÓN DE COLAPSO A CLASE MAYORITARIA" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
grep -A 20 "VERIFICACIÓN DE COLAPSO" $FULL_LOG | head -15 >> $METRICS_FILE 2>/dev/null || echo "No se encontró verificación de colapso" >> $METRICS_FILE

echo "" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
echo "PRÓXIMOS PASOS" >> $METRICS_FILE
echo "════════════════════════════════════════════════════════════════════════════════" >> $METRICS_FILE
cat >> $METRICS_FILE << 'ENDEOF'

1. Revisar F1 scores para clases 1, 2, 3:
   - Si F1 > 0.3 para alguna clase: ✓ Modelo está aprendiendo
   - Si F1 = 0 para todas: ✗ Sigue colapsando

2. Revisar logits finales (más grandes que v6?):
   - v6: rango [-1.2, +1.3] (muy pequeños)
   - v7: esperar rango más grande o distribución más peaky

3. Revisar confusion matrix:
   - Si predicciones solo en clase 0: modelo colapsa
   - Si predicciones en clases 1,2,3: modelo discrimina

4. Revisar verificación de colapso:
   - Si ALERTA: investigar knowledge graph o learning rate
   - Si ✓: continuar a PASO 6 (ablation studies)

5. Comparar loss progression:
   - v6: 1.482 → 1.638 → 1.608 → 1.586 → 1.573 (aumenta en epoch 2)
   - v7: esperar que baje monotónicamente

DECISIÓN:
- Si mejoras significativas (F1 > 0.3): ✓ PASO 6 (ablation studies)
- Si sin mejoras: Investigar arquitectura profunda o entrenamiento adicional

NOTA SOBRE KG NUEVO:
- Cobertura mejorada 45x: 0.04% → 1.83%
- Sin [UNK] tokens
- Palabras de sentimiento curadas para TASS español
- Esperamos inyección de conocimiento efectiva

Si PASO 5 v7 aún tiene problemas:
- Considerar aumentar cobertura: incluir sinónimos de palabras de sentimiento
- Generar relaciones más semánticas (es-antónimo, intensidad, etc.)
- Usar embeddings pre-entrenados de sentimiento español

# Resumen en pantalla
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "RESUMEN FINAL"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✓ Entrenamiento completado en ${MINUTES}m ${SECONDS}s"
echo ""
echo "📊 Archivos de resultado:"
echo "  - Full log: $FULL_LOG"
echo "  - Debug log: $DEBUG_LOG"
echo "  - Loss CSV: $LOSS_CSV"
echo "  - Metrics summary: $METRICS_FILE"
echo ""
echo "🔍 Para revisar resultados:"
echo "  cat $METRICS_FILE"
echo ""
echo "📈 Para ver loss progression:"
echo "  cat $LOSS_CSV"
echo ""
echo "🐛 Para ver debug details:"
echo "  tail -100 $FULL_LOG | grep -A 5 'DEBUG'"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

# Salir con el código de error original si hubo problema
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️  ADVERTENCIA: El script finalizó con código de error: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✅ PASO 5 v7 COMPLETADO EXITOSAMENTE"
echo ""
