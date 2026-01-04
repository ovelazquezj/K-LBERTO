#!/bin/bash
# Traducir relaciones del KG de inglés a español
# Fecha: 2026-01-03

INPUT="brain/kgs/wikidata_ner_spanish.spo"
OUTPUT="brain/kgs/wikidata_ner_spanish_es.spo"
BACKUP="brain/kgs/wikidata_ner_spanish.spo.backup"

echo "Traduciendo relaciones del KG..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Backup
cp "$INPUT" "$BACKUP"
echo "✓ Backup creado: $BACKUP"

# Traducción con sed (en orden de frecuencia para optimizar)
sed 's/\toccupation\t/\tocupación\t/g; 
     s/\tdate_of_birth\t/\tfecha_de_nacimiento\t/g;
     s/\tlocated_in\t/\tubicado_en\t/g;
     s/\tcountry\t/\tpaís\t/g;
     s/\tcoordinates\t/\tcoordenadas\t/g;
     s/\tcitizenship\t/\tciudadanía\t/g;
     s/\tpopulation\t/\tpoblación\t/g;
     s/\tgiven_name\t/\tnombre\t/g;
     s/\tsport\t/\tdeporte\t/g;
     s/\tplace_of_birth\t/\tlugar_de_nacimiento\t/g;
     s/\tfamily_name\t/\tapellido\t/g;
     s/\tinstance_of\t/\ttipo_de\t/g;
     s/\teducated_at\t/\teducado_en\t/g;
     s/\tfield_of_work\t/\tcampo_de_trabajo\t/g;
     s/\tmember_of_sports_team\t/\tmiembro_de_equipo\t/g;
     s/\theadquarters\t/\tsede\t/g;
     s/\tindustry\t/\tindustria\t/g' "$INPUT" > "$OUTPUT"

echo "✓ Traducción completada"

# Estadísticas
echo ""
echo "=== ESTADÍSTICAS ==="
echo "Triplets originales: $(wc -l < $INPUT)"
echo "Triplets traducidos: $(wc -l < $OUTPUT)"

echo ""
echo "=== RELACIONES EN INGLÉS RESTANTES ==="
cut -f2 "$OUTPUT" | grep -E '^[a-z_]+$' | grep -v '^[a-záéíóúñ_]+$' | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== TOP 10 RELACIONES FINALES ==="
cut -f2 "$OUTPUT" | sort | uniq -c | sort -rn | head -10

echo ""
echo "✓ Archivo final: $OUTPUT"
