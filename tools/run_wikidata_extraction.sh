#!/bin/bash
# 
# Ejecutor Completo: Extrae Wikidata y Valida Formato
# ====================================================
#
# Script que automatiza:
# 1. Extracción de entidades de Wikidata en español
# 2. Conversión a formato SPO
# 3. Validación del formato
# 4. Comparación con GeoSpanish.spo
#
# Requisitos previos:
#   pip install SPARQLWrapper requests
#
# Uso:
#   bash run_wikidata_extraction.sh
#   o
#   bash run_wikidata_extraction.sh --limit 5000 --output ./custom_kg.spo

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parámetros por defecto
LIMIT=1000
OUTPUT="./brain/kgs/WikidataSpanish-1000.spo"
REFERENCE="./brain/kgs/GeoSpanish.spo"
LANGUAGE="es"

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --reference)
            REFERENCE="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        *)
            echo "Opción desconocida: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Extractor + Validador de Wikidata para K-BERT              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verificar dependencias
echo -e "${YELLOW}[1/3] Verificando dependencias...${NC}"
python3 -c "import SPARQLWrapper" 2>/dev/null || {
    echo -e "${RED}✗ SPARQLWrapper no instalado${NC}"
    echo "   Instala con: pip install SPARQLWrapper"
    exit 1
}
echo -e "${GREEN}✓ Dependencias OK${NC}"
echo ""

# Ejecutar extractor
echo -e "${YELLOW}[2/3] Extrayendo entidades de Wikidata...${NC}"
echo "     Parámetros:"
echo "       - Límite: $LIMIT entidades"
echo "       - Idioma: $LANGUAGE"
echo "       - Salida: $OUTPUT"
echo ""

python3 wikidata_to_spo.py \
    --output "$OUTPUT" \
    --limit "$LIMIT" \
    --language "$LANGUAGE"

if [ ! -f "$OUTPUT" ]; then
    echo -e "${RED}✗ Error: Archivo no fue creado${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Archivo generado: $OUTPUT${NC}"
echo ""

# Validar formato
echo -e "${YELLOW}[3/3] Validando formato SPO...${NC}"

if [ -f "$REFERENCE" ]; then
    echo "     Comparando con referencia: $REFERENCE"
    python3 validate_spo_format.py \
        --spo_file "$OUTPUT" \
        --reference_file "$REFERENCE" \
        --sample_size 30
else
    echo "     Validando formato (sin referencia)"
    python3 validate_spo_format.py \
        --spo_file "$OUTPUT" \
        --sample_size 30
fi

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}✓ PROCESO COMPLETADO EXITOSAMENTE${NC}"
echo -e "${BLUE}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║ Próximo paso: Entrenar K-BERT con WikidataSpanish.spo        ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║ python3 run_kbert_ner_spanish.py \\                           ║${NC}"
echo -e "${BLUE}║   --kg_name $OUTPUT \\           ║${NC}"
echo -e "${BLUE}║   --train_path ./outputs/conll2002_tsv/train.tsv \\           ║${NC}"
echo -e "${BLUE}║   --dev_path ./outputs/conll2002_tsv/validation.tsv \\        ║${NC}"
echo -e "${BLUE}║   --test_path ./outputs/conll2002_tsv/test.tsv \\             ║${NC}"
echo -e "${BLUE}║   --epochs_num 5 \\                                            ║${NC}"
echo -e "${BLUE}║   --batch_size 16                                            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
