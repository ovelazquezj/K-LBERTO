# Quick Start Guide - K-LBERTO

Getting started with K-LBERTO model training in under 10 minutes.

---

## Step 1: Prerequisites Installation (2 minutes)

Verify Python version:
```bash
python3 --version  # Must be 3.8 or higher
```

Install required dependencies:
```bash
pip install torch transformers scikit-learn
```

---

## Step 2: Clone Repository (1 minute)

```bash
cd ~/projects/
git clone https://github.com/ovelazquezj/K-LBERTO.git
cd K-LBERTO
```

---

## Step 3: Run Training (5 minutes setup, 90 minutes execution)

Execute the pre-configured v8 training script (recommended):

```bash
chmod +x outputs/PASO_5_v8_METRICS_CAPTURE.sh
./outputs/PASO_5_v8_METRICS_CAPTURE.sh
```

The script will handle:
- Dataset loading
- Model initialization
- Training for 10 epochs
- Metrics computation
- Results logging

Expected training time:
- Jetson Orin NX: 1.5 hours
- Intel CPU: 3-4 hours

---

## Step 4: View Results

After training completes, examine the final metrics:

```bash
# View comprehensive metrics
cat outputs/resultados/paso5_v8_metrics_*.txt

# View loss progression per epoch
cat outputs/resultados/paso5_v8_loss_*.csv

# View training logs
tail -100 outputs/resultados/paso5_v8_training_*.log
```

---

## Step 5: Next Steps

Once training is complete:

1. Review the comprehensive README.md for detailed documentation
2. Read docs/RESEARCH.md for complete methodology and analysis
3. Examine the generated model at outputs/kbert_tass_v8_adjusted.bin
4. Consider training with custom datasets using --train_path parameter

For advanced customization, refer to the "Custom Training" section in README.md

---

## Troubleshooting

If you encounter issues:

1. Verify Python version is 3.8 or higher
2. Ensure all dependencies are installed with pip install -r requirements.txt
3. Check that BETO model exists in models/beto_uer_model/
4. Verify dataset files exist in datasets/tass_spanish/
5. Review logs in outputs/resultados/ for specific error messages

For detailed troubleshooting, see FAQ section in README.md

---

## Documentation References

Complete documentation: README.md
Research methodology: docs/RESEARCH.md
Detailed analysis: docs/BITACORA_RQ3_REFINADA_FINAL.md

---

## Support

For questions or issues:
Email: omar.velazquez@edu.uah.es
Repository: https://github.com/ovelazquezj/K-LBERTO
