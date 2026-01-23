#!/bin/bash
# Safety Check - Grid Search LR
# Verifica cada 30 min

PROGRESS_FILE="experiments_progress_gridsearch.json"

while true; do
    echo "=== SAFETY CHECK - $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # Espacio en disco
    echo "💾 ESPACIO EN DISCO:"
    df -h . | tail -1 | awk '{printf "   Usado: %s de %s (%s)\n", $3, $2, $5}'
    SPACE_AVAIL=$(df . | tail -1 | awk '{print $4}')
    if [ $SPACE_AVAIL -lt 5242880 ]; then
        echo "   ⚠️  ADVERTENCIA: Menos de 5GB disponible"
    else
        echo "   ✅ Espacio OK"
    fi

    # Progreso
    echo "📊 PROGRESO DE EXPERIMENTOS:"
    if [ -f "$PROGRESS_FILE" ]; then
        python3 << 'PYTHON'
import json
with open('experiments_progress_gridsearch.json', 'r') as f:
    p = json.load(f)
    completed = len(p.get('completed', []))
    failed = len(p.get('failed', []))
    current = p.get('current', {})
    
    print(f"   Completados: {completed}/20")
    print(f"   Fallados: {failed}")
    
    if current and current.get('name'):
        print(f"   🚀 Actual: {current['name']}")
        print(f"      Inicio: {current.get('start_time', 'N/A')[:19]}")
PYTHON
    fi

    # GPU
    echo "🎮 GPU MEMORY:"
    echo "   ℹ️  nvidia-smi no disponible (Jetson usa jtop)"

    # Últimos resultados
    echo "📁 ÚLTIMOS RESULTADOS:"
    ls -lt resultados/GS_*.csv 2>/dev/null | head -3 | awk '{print "   "$9" ("$6" "$7" "$8")"}'

    echo "========================================="
    echo "Próxima verificación en 30 minutos..."
    echo "========================================="
    
    sleep 1800
done
