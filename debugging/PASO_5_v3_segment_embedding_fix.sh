#!/bin/bash
################################################################################
# PASO 5 v3: Segment Embedding Fix for Single-Sentence TASS Classification
# 
# CRITICAL FIX from v2:
#   Problem: segment embedding received [0,1,1,1...] 
#            told model "real tokens are in segment 1"
#   Solution: Changed to [0,0,0...] 
#             All tokens in same segment 0 (correct for single-sentence)
#
# Changes from PASO 5 v2:
#   - run_kbert_cls.py: Line 89-92 mask assignment corrected
#   - All other parameters identical to v2
#
# Expected Improvement:
#   - F1 should be > 0.4577 (baseline) if fix works
#   - If still collapses to class 0, character-level hypothesis needs revision
#
# Date: Diciembre 27, 2025
################################################################################

set -e

cd ~/projects/K-LBERTO

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./resultados"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/kbert_paso5_v3_${TIMESTAMP}.log"
METRICS_FILE="$LOG_DIR/metrics_paso5_v3_${TIMESTAMP}.txt"

echo "======================================"
echo "PASO 5 v3: SEGMENT EMBEDDING FIX"
echo "======================================"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"
echo ""
echo "CRITICAL CHANGES:"
echo "  - run_kbert_cls.py line 89-92 FIXED"
echo "  - mask: [0,1,1,1...] → [0,0,0...]"
echo "  - Reason: Single-sentence should use consistent segment"
echo ""
echo "Parameters:"
echo "  - Learning rate: 5e-05 (increased from 2e-05)"
echo "  - Batch size: 8 (reduced from 16)"
echo "  - seq_length: 128 (fixed)"
echo "  - Epochs: 5"
echo "  - Knowledge graph: 61 curated triplets (SPANISH CLEAN)"
echo "  - Baseline F1 to beat: 0.4577"
echo ""

# Start training with logging
{
    echo "Starting training at $(date)"
    
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
      --output_model_path ./outputs/kbert_tass_sentiment_88_SPANISH_v3.bin \
      --workers_num 4
    
    echo "Training completed at $(date)"
    
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "======================================"
echo "PASO 5 v3: TRAINING COMPLETE"
echo "======================================"

# Extract metrics from log
echo ""
echo "Extracting metrics..."

{
    echo "PASO 5 v3 - SEGMENT EMBEDDING FIX - METRICS"
    echo "==========================================="
    echo "Timestamp: $TIMESTAMP"
    echo ""
    echo "CHANGE SUMMARY:"
    echo "  File: run_kbert_cls.py line 89-92"
    echo "  Old: mask = [1 if t != PAD_TOKEN else 0 for t in tokens]"
    echo "  New: mask = [0 if t != PAD_TOKEN else 0 for t in tokens]"
    echo "  Impact: All tokens same segment (0) for single-sentence"
    echo ""
    echo "HYPOTHESIS BEING TESTED:"
    echo "  Segment embedding was incorrectly configured for TASS"
    echo "  Expected: Better model learning with correct segments"
    echo ""
    echo "==========================================="
    echo ""
    echo "FINAL EVALUATION RESULTS:"
    echo "==========================================="
    
    # Extract final test results from log
    grep -A 50 "Final evaluation on the test dataset" "$LOG_FILE" | head -30 || echo "No final results found"
    
    echo ""
    echo "==========================================="
    echo "CONFUSION MATRIX (if available):"
    echo "==========================================="
    grep -A 10 "Confusion matrix:" "$LOG_FILE" | head -15 || echo "No confusion matrix found"
    
    echo ""
    echo "==========================================="
    echo "F1 SCORES BY CLASS:"
    echo "==========================================="
    grep "Label.*:" "$LOG_FILE" | tail -10 || echo "No per-class metrics found"
    
    echo ""
    echo "==========================================="
    echo "OVERALL ACCURACY:"
    echo "==========================================="
    grep "Acc\. (Correct/Total):" "$LOG_FILE" | tail -5 || echo "No accuracy found"
    
    echo ""
    echo "==========================================="
    echo "BASELINE COMPARISON:"
    echo "==========================================="
    echo "BETO Baseline (T1): F1 macro = 0.4577"
    echo "K-BERT v2 (segment bug): F1 = ? (collapsed)"
    echo "K-BERT v3 (THIS RUN): F1 = ?"
    echo ""
    echo "SUCCESS CRITERIA:"
    echo "  ✓ F1 > 0.4577: Segment fix works, proceed to modification"
    echo "  ✗ F1 ≤ 0.4577: Character-level issue deeper, revise approach"
    
} > "$METRICS_FILE"

cat "$METRICS_FILE"

echo ""
echo "======================================"
echo "PASO 5 v3: COMPLETE"
echo "======================================"
echo "Results saved to: $METRICS_FILE"
echo "Full log saved to: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review metrics in $METRICS_FILE"
echo "  2. Compare F1 with baseline 0.4577"
echo "  3. If F1 > 0.4577: Segment fix validated ✓"
echo "  4. If F1 ≤ 0.4577: Problem is elsewhere, investigate further"
echo ""
