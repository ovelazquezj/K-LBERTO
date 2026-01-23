#!/bin/bash
# Safety check para experimentos de ablación KG
# Uso: ./safety_check_ablation.sh [--loop]

BASE_DIR="$HOME/projects/K-LBERTO"
PROGRESS_FILE="$BASE_DIR/ablation_progress.json"
LOG_FILE="$BASE_DIR/ablation_log.txt"
RESULTS_DIR="$BASE_DIR/resultados_ablation"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== ABLATION SAFETY CHECK - $(date '+%Y-%m-%d %H:%M:%S') ===${NC}"
}

check_disk() {
    echo -e "${YELLOW}💾 ESPACIO EN DISCO:${NC}"
    USED=$(df -h $BASE_DIR | awk 'NR==2 {print $3}')
    TOTAL=$(df -h $BASE_DIR | awk 'NR==2 {print $2}')
    PCT=$(df -h $BASE_DIR | awk 'NR==2 {print $5}' | tr -d '%')
    echo "   Usado: $USED de $TOTAL ($PCT%)"
    if [ "$PCT" -gt 90 ]; then
        echo -e "   ${RED}⚠️  ALERTA: Espacio bajo${NC}"
    else
        echo -e "   ${GREEN}✅ Espacio OK${NC}"
    fi
}

check_progress() {
    echo -e "${YELLOW}📊 PROGRESO DE EXPERIMENTOS:${NC}"
    if [ -f "$PROGRESS_FILE" ]; then
        COMPLETED=$(python3 -c "import json; d=json.load(open('$PROGRESS_FILE')); print(len(d.get('completed', [])))" 2>/dev/null || echo "0")
        FAILED=$(python3 -c "import json; d=json.load(open('$PROGRESS_FILE')); print(len(d.get('failed', [])))" 2>/dev/null || echo "0")
        echo "   Completados: $COMPLETED/18"
        echo "   Fallidos: $FAILED"
        
        if [ "$COMPLETED" -gt 0 ]; then
            echo -e "${YELLOW}   Últimos completados:${NC}"
            python3 -c "import json; d=json.load(open('$PROGRESS_FILE')); [print(f'     - {x}') for x in d.get('completed', [])[-3:]]" 2>/dev/null
        fi
        
        if [ "$FAILED" -gt 0 ]; then
            echo -e "${RED}   Fallidos:${NC}"
            python3 -c "import json; d=json.load(open('$PROGRESS_FILE')); [print(f'     - {x}') for x in d.get('failed', [])]" 2>/dev/null
        fi
    else
        echo "   Archivo de progreso no encontrado"
    fi
}

check_current_experiment() {
    echo -e "${YELLOW}🔬 EXPERIMENTO ACTUAL:${NC}"
    if [ -f "$LOG_FILE" ]; then
        LAST_EXP=$(grep "Ejecutando:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -n "$LAST_EXP" ]; then
            echo "   $LAST_EXP"
            # Tiempo desde inicio
            START_TIME=$(grep "$LAST_EXP" "$LOG_FILE" | head -1 | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
            if [ -n "$START_TIME" ]; then
                START_SEC=$(date -d "$START_TIME" +%s 2>/dev/null)
                NOW_SEC=$(date +%s)
                if [ -n "$START_SEC" ]; then
                    ELAPSED=$(( (NOW_SEC - START_SEC) / 60 ))
                    echo "   Tiempo transcurrido: ${ELAPSED} min"
                fi
            fi
        fi
    else
        echo "   Log no encontrado"
    fi
}

check_gpu() {
    echo -e "${YELLOW}🎮 GPU/MEMORIA:${NC}"
    if command -v jtop &> /dev/null; then
        # Jetson
        echo "   (Usar jtop para monitoreo detallado)"
    fi
    # Memoria RAM
    FREE_MEM=$(free -h | awk '/^Mem:/ {print $4}')
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    echo "   RAM libre: $FREE_MEM de $TOTAL_MEM"
}

check_process() {
    echo -e "${YELLOW}⚙️  PROCESO:${NC}"
    PID=$(pgrep -f "run_ablation_experiments.py" | head -1)
    if [ -n "$PID" ]; then
        echo -e "   ${GREEN}✅ Corriendo (PID: $PID)${NC}"
        # CPU usage
        CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null | tr -d ' ')
        MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null | tr -d ' ')
        echo "   CPU: ${CPU}% | MEM: ${MEM}%"
    else
        echo -e "   ${RED}❌ No está corriendo${NC}"
    fi
}

check_results_summary() {
    echo -e "${YELLOW}📁 RESUMEN DE RESULTADOS:${NC}"
    if [ -d "$RESULTS_DIR" ]; then
        COUNT=$(find "$RESULTS_DIR" -name "metrics.csv" 2>/dev/null | wc -l)
        echo "   Carpetas con metrics.csv: $COUNT"
        
        # Mostrar F1 de completados
        if [ "$COUNT" -gt 0 ]; then
            echo -e "${YELLOW}   F1 Scores:${NC}"
            for dir in "$RESULTS_DIR"/ABL_*/; do
                if [ -f "${dir}metrics.csv" ]; then
                    EXP_NAME=$(basename "$dir")
                    F1=$(tail -1 "${dir}metrics.csv" 2>/dev/null | cut -d',' -f4 2>/dev/null)
                    if [ -n "$F1" ]; then
                        echo "     $EXP_NAME: F1=$F1"
                    fi
                fi
            done
        fi
    else
        echo "   Directorio de resultados no existe aún"
    fi
}

check_last_logs() {
    echo -e "${YELLOW}📜 ÚLTIMAS LÍNEAS DEL LOG:${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" | sed 's/^/   /'
    fi
}

# Main
print_header
echo ""
check_disk
echo ""
check_progress
echo ""
check_current_experiment
echo ""
check_gpu
echo ""
check_process
echo ""
check_results_summary
echo ""
check_last_logs
echo ""
echo -e "${BLUE}=========================================${NC}"

# Loop mode
if [ "$1" == "--loop" ]; then
    echo "Próxima verificación en 30 minutos..."
    echo "========================================="
    sleep 1800
    exec "$0" --loop
fi
