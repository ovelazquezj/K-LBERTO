#!/bin/bash
# Validar fix con un solo experimento

echo "========================================="
echo "  VALIDACIÓN DEL FIX"
echo "========================================="
echo ""
echo "Ejecutará 1 experimento: D500_base_seed42 (1 época)"
echo "Tiempo: ~3-4 minutos"
echo ""

# Config temporal
cat > experiments_config_validate.json << 'INNER_EOF'
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
    "report_steps": 50
  },
  
  "research_questions": {
    "RQ1": {
      "description": "Validación fix",
      "base_lr": 2e-5,
      "datasets": [500],
      "seeds": [42],
      "configs": [
        {"name": "base"}
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

# Ejecutar
python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, '.')
from run_experiments import ExperimentRunner

runner = ExperimentRunner('experiments_config_validate.json')
runner.run_all_rq1()
PYTHON_EOF

# Verificar resultado
echo ""
echo "========================================="
echo "  VERIFICACIÓN"
echo "========================================="

if [ -f "experiments_progress.json" ]; then
    completed=$(grep -o '"completed"' experiments_progress.json | wc -l)
    failed=$(python3 -c "import json; f=open('experiments_progress.json'); d=json.load(f); print(len(d.get('failed', [])))")
    
    echo ""
    echo "Resultado del fix:"
    echo "  Completados: $completed"
    echo "  Fallados: $failed"
    echo ""
    
    if [ "$failed" -eq "0" ] && [ "$completed" -gt "0" ]; then
        echo "✅ FIX VALIDADO - Sistema funcionando correctamente"
        echo ""
        echo "Listo para ejecutar los 42 experimentos completos de RQ1"
    else
        echo "❌ FIX NO FUNCIONÓ - Revisar logs"
        echo ""
        cat experiments_progress.json
    fi
else
    echo "❌ No se encontró experiments_progress.json"
fi
