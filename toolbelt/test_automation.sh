#!/bin/bash
# Test rápido del sistema de automatización
# Ejecuta solo 2 experimentos pequeños para validar

echo "========================================="
echo "  TEST DE AUTOMATIZACIÓN"
echo "========================================="
echo ""
echo "Este test ejecutará 2 experimentos pequeños:"
echo "  1. RQ1_D500_base_seed42 (1 época)"
echo "  2. RQ1_D500_sqrt_seed42 (1 época)"
echo ""
echo "Tiempo estimado: ~5-7 minutos"
echo ""
read -p "¿Continuar? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelado"
    exit 1
fi

# Crear versión temporal de experiments_config.json solo con test
cat > experiments_config_test.json << 'INNER_EOF'
{
  "base_config": {
    "pretrained_model_path": "models/beto_model/pytorch_model.bin",
    "vocab_path": "models/beto_model/vocab.txt",
    "config_path": "models/beto_model/config_uer.json",
    "dev_path": "data/wikiann_tsv/dev.tsv",
    "test_path": "data/wikiann_tsv/test.tsv",
    "kg_name": "brain/kgs/wikidata_ner_spanish.spo",
    "output_dir": "resultados",
    "epochs_num": 1,
    "batch_size": 4,
    "seq_length": 128,
    "report_steps": 25
  },
  
  "research_questions": {
    "RQ1": {
      "description": "Learning Rate Scaling Validation - TEST",
      "base_lr": 2e-5,
      "scaling_formula": "lr_new = lr_base / sqrt(N_new / N_base)",
      "datasets": [500],
      "seeds": [42],
      "configs": [
        {"name": "base", "lr_multiplier": 1.0},
        {"name": "sqrt", "lr_formula": "sqrt_scaled"}
      ]
    }
  },
  
  "hardware_limits": {
    "max_parallel": 1,
    "gpu_memory_limit": 7000,
    "timeout_minutes": 30
  }
}
INNER_EOF

echo ""
echo "✓ Config de test creado (2 experimentos × 1 época)"
echo ""
echo "Ejecutando experimentos..."
echo ""

# Ejecutar con config de test
python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, '.')
from run_experiments import ExperimentRunner

runner = ExperimentRunner('experiments_config_test.json')
runner.run_all_rq1()
PYTHON_EOF

# Verificar resultados
echo ""
echo "========================================="
echo "  VERIFICACIÓN DE RESULTADOS"
echo "========================================="
echo ""

if [ -f "resultados/RQ1_D500_base_seed42_metrics.csv" ]; then
    echo "✅ Experimento 1 completado"
    echo "   Métricas finales:"
    tail -1 resultados/RQ1_D500_base_seed42_metrics.csv | \
    awk -F',' '{printf "   Overall F1=%.3f | PER=%.3f | LOC=%.3f | ORG=%.3f\n", $5, $8, $11, $14}'
else
    echo "❌ Experimento 1 falló"
fi

echo ""

if [ -f "resultados/RQ1_D500_sqrt_seed42_metrics.csv" ]; then
    echo "✅ Experimento 2 completado"
    echo "   Métricas finales:"
    tail -1 resultados/RQ1_D500_sqrt_seed42_metrics.csv | \
    awk -F',' '{printf "   Overall F1=%.3f | PER=%.3f | LOC=%.3f | ORG=%.3f\n", $5, $8, $11, $14}'
else
    echo "❌ Experimento 2 falló"
fi

echo ""
echo "========================================="
echo "  CONSOLIDACIÓN"
echo "========================================="
echo ""

python consolidate_results.py

echo ""
echo "✓ Test completado"
echo ""
echo "Si ambos experimentos completaron exitosamente,"
echo "el sistema está listo para los 42 experimentos completos de RQ1."
echo ""
