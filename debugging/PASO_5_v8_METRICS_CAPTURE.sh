#!/bin/bash

# ════════════════════════════════════════════════════════════════════════════════
# PASO 5 v8 - K-BERT CON HIPERPARÁMETROS AJUSTADOS PARA DATASET GRANDE
# ════════════════════════════════════════════════════════════════════════════════
#
# DIAGNÓSTICO DE v7 (CANCELADO):
# - v7 divergió: loss aumentó 86% (1.505 → 2.804) en epoch 2
# - Root cause: Hiperparámetros NO optimizados para dataset 2.4x más grande
#
# CORRECCIONES APLICADAS EN v8:
# #1: Learning rate reducido 5x (5e-05 → 1e-05)
#     Razón: Dataset 2.4x más grande necesita gradientes más pequeños
# #2: Dropout reducido (0.5 → 0.3)
#     Razón: Dropout agresivo + dataset nuevo = divergencia garantizada
# #3: Mantener 10 epochs (suficiente si converge)
# #4: Mantener 5 fixes de código de v7
#
# CAMBIOS v7 → v8:
# - Learning rate: 5e-05 → 1e-05 (5x menor)
# - Dropout: 0.5 → 0.3 (menos agresivo)
# - Dataset: 2176 muestras (sin cambio)
# - KG: TASS_sentiment_KG_FINAL.spo (sin cambio)
# - Epochs: 10 (sin cambio)
#
# CRITERIO DE ÉXITO v8:
# ✓ Loss baja continuamente (NO diverge como v7)
# ✓ Logits se amplían (rango > 0.8)
# ✓ Predicciones en múltiples clases (diversificación)
# ✓ F1 > 0.15 para clases 1-3
# ✓ Accuracy > 0.50
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
OUTPUT_MODEL=$PROJECT_DIR/outputs/kbert_tass_v8_adjusted.bin

# Log files
FULL_LOG=$RESULTADOS_DIR/paso5_v8_training_${TIMESTAMP}.log
DEBUG_LOG=$RESULTADOS_DIR/paso5_v8_debug_${TIMESTAMP}.log
LOSS_CSV=$RESULTADOS_DIR/paso5_v8_loss_${TIMESTAMP}.csv
METRICS_FILE=$RESULTADOS_DIR/paso5_v8_metrics_${TIMESTAMP}.txt

echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 5 v8 - K-BERT CON HIPERPARÁMETROS AJUSTADOS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 CONFIGURACIÓN AJUSTADA:"
echo "  Epochs: 10"
echo "  Batch size: 8"
echo "  Learning rate: 1e-05 (REDUCIDO 5x de v7: 5e-05 → 1e-05)"
echo "  Dropout: 0.3 (REDUCIDO de v7: 0.5 → 0.3)"
echo "  Loss: NLLLoss con class_weights"
echo ""
echo "📂 RUTAS:"
echo "  Vocab: $VOCAB_PATH"
echo "  Config: $CONFIG_PATH"
echo "  Modelo pre-entrenado: $PRETRAINED_MODEL"
echo "  Dataset train: $TRAIN_PATH (1522 muestras)"
echo "  Dataset dev: $DEV_PATH (654 muestras)"
echo "  Dataset test: $TEST_PATH (654 muestras)"
echo "  KG: $KG_PATH (138 triplets, 1.83% cobertura)"
echo ""
echo "📊 DATASET REGENERADO:"
echo "  Total: 2176 muestras (1.93x vs v6: 1125)"
echo "  Sin menciones (@usuario): 0/2176"
echo "  Sin URLs: 0/2176"
echo "  Sin duplicados: 0/2176"
echo "  Clases balanceadas: 42.4%, 31.3%, 12.4%, 13.9%"
echo ""
echo "🧠 KNOWLEDGE GRAPH:"
echo "  Subjects: 79 únicos"
echo "  Triplets: 138 limpias"
echo "  Cobertura: 1.83% (vs 0.04% en v6)"
echo "  [UNK] tokens: 0% (vs 40% en v6)"
echo ""
echo "💾 RESULTADOS:"
echo "  Full log: $FULL_LOG"
echo "  Debug log: $DEBUG_LOG"
echo "  Loss CSV: $LOSS_CSV"
echo "  Metrics: $METRICS_FILE"
echo ""
echo "⏱️  TIEMPO ESTIMADO: 2-3 horas (10 epochs × 1522 muestras)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 INICIANDO PASO 5 v8..."
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
    --learning_rate 1e-05 \
    --dropout 0.3 \
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
PASO 5 v8 - RESUMEN DE RESULTADOS
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
- Si loss baja continuamente: ✓ Ajuste exitoso, evaluar resultados finales
- Si F1 > 0.3 en clases 1-3: ✓ PASO 6 (ablation studies)
- Si F1 0.15-0.30: PASO 8b (aumentar cobertura KG a 5-10%)
- Si diverge nuevamente: PASO 8c (investigar arquitectura/embeddings)

AJUSTES APLICADOS v7 → v8:
- Learning rate: 5e-05 → 1e-05 (5x menor para convergencia estable)
- Dropout: 0.5 → 0.3 (menos agresivo, mayor flexibilidad)
- Razón: Dataset 2.4x más grande requiere gradientes más pequeños
- Esperado: Loss baja monótonamente (NO diverge como v7)

Si v8 diverge de nuevo:
- Revisar inicialización de pesos en embedding layers
- Verificar que visible_matrix en KnowledgeGraph no esté colapsada
- Considerar warm-up learning rate (primeras épocas con LR más alto)
- Investigar si KG causa distorsión en embeddings intermedios

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
echo "✅ PASO 5 v8 COMPLETADO"
echo ""
