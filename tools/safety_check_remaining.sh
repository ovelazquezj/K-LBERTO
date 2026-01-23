#!/bin/bash
# Safety check para experimentos restantes

while true; do
    clear
    echo "=== SAFETY CHECK - $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    echo "💾 ESPACIO EN DISCO:"
    df -h . | tail -1 | awk '{printf "   Usado: %s de %s (%s usado)\n", $3, $2, $5}'
    
    used_pct=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$used_pct" -gt 80 ]; then
        echo "   ⚠️  Espacio crítico"
    else
        echo "   ✅ Espacio OK"
    fi
    
    echo "📊 PROGRESO DE EXPERIMENTOS:"
    if [ -f experiments_progress_remaining.json ]; then
        python3 << 'PYTHON'
import json
with open('experiments_progress_remaining.json') as f:
    p = json.load(f)
    print(f"   Completados: {len(p['completed'])}/18")
    print(f"   Fallados: {len(p['failed'])}")
    print(f"   Saltados: {len(p.get('skipped', []))}")
    if p.get('current'):
        print(f"   🚀 Actual: {p['current']['name']}")
        print(f"      Inicio: {p['current']['start_time']}")
PYTHON
    else
        echo "   ⚠️  Archivo de progreso no encontrado"
    fi
    
    echo "🎮 GPU MEMORY:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "   Usado: %dMB / %dMB\n", $1, $2}'
    else
        echo "   ℹ️  nvidia-smi no disponible (Jetson usa jtop)"
    fi
    
    echo "📁 ÚLTIMOS RESULTADOS:"
    ls -lt resultados/*.csv 2>/dev/null | head -3 | \
    awk '{printf "   %s (%s %s %s)\n", $9, $6, $7, $8}'
    
    echo "========================================="
    echo "Próxima verificación en 30 minutos..."
    echo "========================================="
    
    sleep 1800  # 30 minutos
done
