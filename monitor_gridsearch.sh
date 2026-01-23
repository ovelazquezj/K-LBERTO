#!/bin/bash
# Monitor de progreso - Grid Search LR
# Uso: ./monitor_gridsearch.sh [intervalo_segundos]

INTERVAL=${1:-10}
PROGRESS_FILE="experiments_progress_gridsearch.json"

while true; do
    clear
    echo "========================================="
    echo "  GRID SEARCH LR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="

    if [ -f "$PROGRESS_FILE" ]; then
        python3 << 'PYTHON'
import json
with open('experiments_progress_gridsearch.json', 'r') as f:
    p = json.load(f)
    completed = len(p.get('completed', []))
    failed = len(p.get('failed', []))
    current = p.get('current', {})

    print(f"\n📊 PROGRESO: {completed}/20 completados")
    if failed > 0:
        print(f"❌ Fallados: {failed}")
    
    # Mostrar completados por dataset
    comp = p.get('completed', [])
    for d in ['D500', 'D1000', 'D2000', 'D3000']:
        done = [c for c in comp if d in c]
        print(f"   {d}: {len(done)}/5 LRs")
    
    if current and current.get('name'):
        print(f"\n🚀 Ejecutando: {current['name']}")
        print(f"   Inicio: {current.get('start_time', 'unknown')[:19]}")
PYTHON

        # Mostrar última métrica del experimento actual
        current=$(python3 -c "import json; p=json.load(open('$PROGRESS_FILE')); c=p.get('current',{}); print(c.get('name',''))" 2>/dev/null)
        if [ ! -z "$current" ]; then
            metrics_file="resultados/${current}_metrics.csv"
            if [ -f "$metrics_file" ]; then
                echo ""
                echo "--- ÚLTIMA MÉTRICA ---"
                tail -1 "$metrics_file" | awk -F',' '{
                    if ($1 != "epoch") printf "Epoch %s: F1=%.4f\n", $1, $5
                }'
            fi
            
            # Última línea del log
            log_file="logs/${current}.log"
            if [ -f "$log_file" ]; then
                echo ""
                echo "--- LOG ---"
                tail -3 "$log_file" | grep -E "Epoch|loss|Report" | tail -2
            fi
        fi
    else
        echo "⚠️  $PROGRESS_FILE no encontrado"
    fi

    echo ""
    echo "--- RECURSOS ---"
    df -h . | tail -1 | awk '{printf "💾 Disco: %s usado de %s\n", $3, $2}'
    free -h | grep Mem | awk '{printf "🧠 RAM: %s / %s\n", $3, $2}'
    
    echo ""
    echo "========================================="
    echo "Actualiza cada ${INTERVAL}s | Ctrl+C salir"
    echo "========================================="
    
    sleep $INTERVAL
done
