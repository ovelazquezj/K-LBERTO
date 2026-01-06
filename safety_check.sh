#!/bin/bash
# Script de monitoreo de seguridad
# Verifica cada 30 min que todo funcione correctamente

echo "========================================="
echo "  SAFETY CHECK - Monitoreo Continuo"
echo "========================================="
echo "Verificando cada 30 minutos..."
echo "Presiona Ctrl+C para detener"
echo ""

while true; do
    clear
    echo "=== SAFETY CHECK - $(date +%Y-%m-%d\ %H:%M:%S) ==="
    echo ""
    
    # 1. Verificar espacio en disco
    echo "💾 ESPACIO EN DISCO:"
    df -h . | tail -1 | awk '{printf "   Usado: %s de %s (%s usado)\n", $3, $2, $5}'
    SPACE_AVAIL=$(df . | tail -1 | awk '{print $4}')
    if [ $SPACE_AVAIL -lt 5242880 ]; then  # Menos de 5GB
        echo "   ⚠️  ADVERTENCIA: Menos de 5GB disponible"
    else
        echo "   ✅ Espacio OK"
    fi
    
    echo ""
    
    # 2. Verificar progreso de experimentos
    echo "📊 PROGRESO DE EXPERIMENTOS:"
    if [ -f experiments_progress.json ]; then
        python3 << 'PYTHON'
import json
try:
    with open('experiments_progress.json', 'r') as f:
        p = json.load(f)
        completed = len(p.get('completed', []))
        failed = len(p.get('failed', []))
        skipped = len(p.get('skipped', []))
        current = p.get('current', {})
        
        print(f"   Completados: {completed}/42")
        print(f"   Fallados: {failed}")
        print(f"   Saltados: {skipped}")
        
        if current and current.get('name'):
            print(f"   🚀 Actual: {current['name']}")
            print(f"      Inicio: {current.get('started_at', 'unknown')}")
        else:
            print("   ⏸️  Sin experimento activo")
except Exception as e:
    print(f"   ❌ Error leyendo progreso: {e}")
PYTHON
    else
        echo "   ⚠️  experiments_progress.json no existe"
    fi
    
    echo ""
    
    # 3. Uso de GPU
    echo "🎮 GPU MEMORY:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk '{printf "   Memoria: %d / %d MB (%.1f%%)\n   Temperatura: %d°C\n", $1, $2, ($1/$2)*100, $3}'
    else
        echo "   ℹ️  nvidia-smi no disponible (Jetson usa jtop)"
    fi
    
    echo ""
    
    # 4. Últimos archivos generados
    echo "📁 ÚLTIMOS RESULTADOS:"
    ls -lt resultados/*.csv 2>/dev/null | head -3 | awk '{print "   "$9" ("$6" "$7" "$8")"}'
    
    echo ""
    echo "========================================="
    echo "Próxima verificación en 30 minutos..."
    echo "========================================="
    
    sleep 1800  # 30 minutos
done
