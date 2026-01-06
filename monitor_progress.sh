#!/bin/bash
# Monitor de progreso de experimentos
# Uso: ./monitor_progress.sh [intervalo_segundos]

INTERVAL=${1:-5}

clear

while true; do
    clear
    echo "========================================="
    echo "  MONITOR DE EXPERIMENTOS - $(date +%H:%M:%S)"
    echo "========================================="
    echo ""
    
    # Leer progreso
    if [ -f experiments_progress.json ]; then
        echo "--- PROGRESO GENERAL ---"
        completed=$(grep -o '"completed"' experiments_progress.json | wc -l)
        failed=$(grep -o '"failed"' experiments_progress.json | wc -l)
        
        echo "✅ Completados: $completed"
        echo "❌ Fallados: $failed"
        echo ""
        
        # Experimento actual
        echo "--- EXPERIMENTO ACTUAL ---"
        current=$(grep -A2 '"current":' experiments_progress.json | grep '"name"' | cut -d'"' -f4)
        if [ ! -z "$current" ]; then
            echo "🚀 Ejecutando: $current"
            
            # Buscar log del experimento actual
            log_file="logs/${current}.log"
            if [ -f "$log_file" ]; then
                echo ""
                echo "--- ÚLTIMAS LÍNEAS DEL LOG ---"
                tail -15 "$log_file" | grep -E "Epoch|Report precision|PER:|LOC:|ORG:" | tail -8
            fi
            
            # Buscar archivos de métricas
            metrics_file="resultados/${current}_metrics.csv"
            if [ -f "$metrics_file" ]; then
                echo ""
                echo "--- ÚLTIMA MÉTRICA ---"
                tail -1 "$metrics_file" | awk -F',' '{
                    printf "Epoch %s (%s): Overall F1=%.3f | PER=%.3f | LOC=%.3f | ORG=%.3f\n", 
                    $1, $2, $5, $8, $11, $14
                }'
            fi
        else
            echo "⏸️  Sin experimento activo"
        fi
    else
        echo "⚠️  No se encontró experiments_progress.json"
        echo "   Los experimentos no han iniciado"
    fi
    
    echo ""
    echo "--- RECURSOS (JETSON) ---"
    
    # GPU usage desde jtop (si está disponible)
    if command -v jtop &> /dev/null; then
        # Extraer info de jtop
        gpu_usage=$(jtop --health 2>/dev/null | grep -i "gpu" | head -1 | awk '{print $2}' || echo "N/A")
        echo "GPU Usage: ${gpu_usage}%"
        
        gpu_mem=$(jtop --health 2>/dev/null | grep -i "gpu" | head -1 | awk '{print $4"/"$5}' || echo "N/A")
        echo "GPU Memory: ${gpu_mem}"
    else
        # Fallback: nvidia-smi style
        echo "GPU Usage: [usa jtop en otra terminal para ver]"
    fi
    
    # CPU
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "CPU Usage: ${cpu_usage}%"
    
    # Memory
    mem_usage=$(free -h | grep Mem | awk '{printf "%s / %s", $3, $2}')
    echo "Memory: ${mem_usage}"
    
    echo ""
    echo "========================================="
    echo "Actualizando cada ${INTERVAL}s (Ctrl+C para salir)"
    echo "Para GPU detallado: jtop en otra terminal"
    echo "========================================="
    
    sleep $INTERVAL
done
