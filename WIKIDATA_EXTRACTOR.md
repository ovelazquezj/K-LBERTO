# Wikidata Spanish KG Extractor para K-BERT

Script Python para extraer 10K entidades de Wikidata en español y convertir al formato SPO (Subject-Predicate-Object) compatible con K-BERT.

## Instalación de Dependencias

```bash
pip install SPARQLWrapper requests
```

## Uso Básico

### Opción 1: Línea de comando (recomendado)

```bash
python3 wikidata_to_spo.py \
    --output ./brain/kgs/WikidataSpanish.spo \
    --limit 10000 \
    --language es
```

### Opción 2: Con parámetros por defecto

```bash
python3 wikidata_to_spo.py
```

Esto generará `./WikidataSpanish.spo` con 10K entidades en español.

## Parámetros

- `--output`: Ruta del archivo de salida (default: `./WikidataSpanish.spo`)
- `--limit`: Número máximo de entidades a extraer (default: 10000)
- `--language`: Código de idioma ISO (default: `es`)

## Propiedades Extraídas

El script extrae automáticamente estas propiedades básicas de cada entidad:

- **tipo_entidad**: Clasificación de la entidad (persona, organización, lugar, etc.)
- **ocupacion**: Profesión o rol
- **ubicacion**: Coordenadas geográficas (lat,lon)
- **lugar_nacimiento**: Lugar donde nació
- **pais**: País asociado
- **fecha_nacimiento**: Fecha de nacimiento
- **industria**: Sector industrial
- **fundador**: Quién fundó la entidad
- **ubicacion_administrativa**: Territorio administrativo
- **pais_ciudadania**: País de ciudadanía

## Formato de Salida (SPO)

El archivo generado tiene este formato (igual a `GeoSpanish.spo`):

```
Entidad[TAB]Predicado[TAB]Valor
```

Ejemplo:

```
Albert Einstein	tipo_entidad	físico
Albert Einstein	ocupacion	físico teórico
Albert Einstein	lugar_nacimiento	Ulm
Albert Einstein	pais	Alemania
Albert Einstein	fecha_nacimiento	1879
Marie Curie	tipo_entidad	científico
Marie Curie	ocupacion	química
Marie Curie	pais	Polonia
```

## Tiempo de Ejecución

- 10K entidades: **30-60 minutos** (depende de conexión)
- Rate limiting automático: 0.1-1 segundo entre queries
- No hay límite de Wikidata para uso académico

## Próximos Pasos

Una vez generado `WikidataSpanish.spo`, usar con tu script de entrenamiento:

```bash
python3 run_kbert_ner_spanish.py \
    --kg_name ./brain/kgs/WikidataSpanish.spo \
    --train_path ./outputs/conll2002_tsv/train.tsv \
    --dev_path ./outputs/conll2002_tsv/validation.tsv \
    --test_path ./outputs/conll2002_tsv/test.tsv \
    --epochs_num 5 \
    --batch_size 16 \
    --learning_rate 2e-05 \
    --output_model_path ./outputs/kbert_ner/kbert_wikidata_spanish.bin
```

## Solución de Problemas

### Error: "SPARQLWrapper not found"
```bash
pip install --upgrade SPARQLWrapper
```

### Error: "Connection timeout"
El script reintenta automáticamente. Si persiste:
- Verifica conexión a internet
- Intenta de nuevo más tarde
- Reduce `--limit` a 5000

### Archivo vacío o sin datos
- Verifica el idioma (`--language es`)
- Revisa logs para más detalles
- Intenta con `--limit 100` primero para testing

## Estructura del KG

El Knowledge Graph resultante tendrá:

```
WikidataSpanish.spo
├─ ~10,000 entidades
├─ ~50,000-100,000 triples
├─ 9 predicados básicos
└─ Todas con etiquetas en español
```

Compatible 100% con:
- `GeoSpanish.spo` (mismo formato)
- `run_kbert_ner_spanish.py` (sin cambios)
- K-BERT arquitectura

## Nota

El script usa la **API pública de Wikidata** sin autenticación. No hay límites de rate para uso no comercial.
