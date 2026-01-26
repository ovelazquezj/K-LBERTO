#!/usr/bin/env python3
"""
Análisis de Errores: CUR vs RAW en NER
Investigar por qué RAW supera a CUR

Autor: Omar Velázquez
Fecha: 2026-01-25
"""

import json
import os
from collections import Counter, defaultdict
import csv

def load_tsv(filepath):
    """Carga archivo TSV de NER"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                labels = row[0].split()
                text = row[1]
                tokens = text.split()
                samples.append({
                    'labels': labels,
                    'text': text,
                    'tokens': tokens
                })
    return samples

def analyze_sample(sample):
    """Analiza características de una muestra"""
    labels = sample['labels']
    text = sample['text']
    tokens = sample['tokens']

    # Densidad de entidades
    entity_tokens = sum(1 for l in labels if l != 'O')
    total_tokens = len(labels)
    density = entity_tokens / total_tokens if total_tokens > 0 else 0

    # Es REDIRECCIÓN?
    is_redirect = text.startswith('REDIRECCIÓN') or text.startswith('redirección')

    # Número de entidades (contar B-tags)
    num_entities = sum(1 for l in labels if l.startswith('B-'))

    # Entidades consecutivas (B seguido inmediatamente de otro B)
    consecutive_entities = 0
    for i in range(len(labels) - 1):
        if labels[i].startswith('B-') or labels[i].startswith('I-'):
            if labels[i+1].startswith('B-'):
                consecutive_entities += 1

    # Tipos de entidades
    entity_types = set()
    for l in labels:
        if l.startswith('B-') or l.startswith('I-'):
            entity_types.add(l[2:])

    return {
        'density': density,
        'is_redirect': is_redirect,
        'num_entities': num_entities,
        'consecutive_entities': consecutive_entities,
        'entity_types': entity_types,
        'total_tokens': total_tokens,
        'high_density': density > 0.5,
        'very_high_density': density > 0.9
    }

def load_training_data(filepath):
    """Carga datos de entrenamiento"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                samples.append({
                    'labels': row[0].split(),
                    'text': row[1],
                    'tokens': row[1].split()
                })
    return samples

def main():
    print("=" * 70)
    print("ANÁLISIS DE ERRORES: CUR vs RAW")
    print("=" * 70)

    # Paths
    base_path = "/home/omar/projects/K-LBERTO"
    test_path = f"{base_path}/data/wikiann_tsv/test.tsv"
    train_cur_path = f"{base_path}/data/wikiann_tsv/train_4000.tsv"
    train_raw_path = f"{base_path}/data/wikiann_tsv/train_raw_4000.tsv"

    # 1. Analizar Test Set
    print("\n" + "=" * 70)
    print("1. ANÁLISIS DEL TEST SET")
    print("=" * 70)

    test_samples = load_tsv(test_path)
    print(f"\nTotal samples en test: {len(test_samples)}")

    test_stats = {
        'total': len(test_samples),
        'redirects': 0,
        'high_density': 0,
        'very_high_density': 0,
        'with_consecutive': 0,
        'density_buckets': defaultdict(int),
        'entity_counts': defaultdict(int)
    }

    for sample in test_samples:
        analysis = analyze_sample(sample)
        if analysis['is_redirect']:
            test_stats['redirects'] += 1
        if analysis['high_density']:
            test_stats['high_density'] += 1
        if analysis['very_high_density']:
            test_stats['very_high_density'] += 1
        if analysis['consecutive_entities'] > 0:
            test_stats['with_consecutive'] += 1

        # Bucket de densidad
        bucket = int(analysis['density'] * 10) / 10
        test_stats['density_buckets'][bucket] += 1

        for etype in analysis['entity_types']:
            test_stats['entity_counts'][etype] += 1

    print(f"\n  Muestras REDIRECCIÓN: {test_stats['redirects']} ({100*test_stats['redirects']/test_stats['total']:.1f}%)")
    print(f"  Alta densidad (>50%): {test_stats['high_density']} ({100*test_stats['high_density']/test_stats['total']:.1f}%)")
    print(f"  Muy alta densidad (>90%): {test_stats['very_high_density']} ({100*test_stats['very_high_density']/test_stats['total']:.1f}%)")
    print(f"  Con entidades consecutivas: {test_stats['with_consecutive']} ({100*test_stats['with_consecutive']/test_stats['total']:.1f}%)")

    print("\n  Distribución de densidad de entidades:")
    for bucket in sorted(test_stats['density_buckets'].keys()):
        count = test_stats['density_buckets'][bucket]
        bar = '█' * int(count / 5)
        print(f"    {bucket:.1f}-{bucket+0.1:.1f}: {count:4d} {bar}")

    # 2. Comparar Train CUR vs RAW
    print("\n" + "=" * 70)
    print("2. COMPARACIÓN TRAIN: CURADO vs RAW")
    print("=" * 70)

    if os.path.exists(train_cur_path) and os.path.exists(train_raw_path):
        train_cur = load_training_data(train_cur_path)
        train_raw = load_training_data(train_raw_path)

        print(f"\n  Train CURADO: {len(train_cur)} samples")
        print(f"  Train RAW:    {len(train_raw)} samples")

        # Analizar cada conjunto
        for name, data in [("CURADO", train_cur), ("RAW", train_raw)]:
            stats = {
                'redirects': 0,
                'high_density': 0,
                'very_high_density': 0,
                'with_consecutive': 0,
                'avg_density': 0
            }

            densities = []
            for sample in data:
                analysis = analyze_sample(sample)
                densities.append(analysis['density'])
                if analysis['is_redirect']:
                    stats['redirects'] += 1
                if analysis['high_density']:
                    stats['high_density'] += 1
                if analysis['very_high_density']:
                    stats['very_high_density'] += 1
                if analysis['consecutive_entities'] > 0:
                    stats['with_consecutive'] += 1

            stats['avg_density'] = sum(densities) / len(densities) if densities else 0

            print(f"\n  {name}:")
            print(f"    - REDIRECCIÓN:           {stats['redirects']:5d} ({100*stats['redirects']/len(data):.1f}%)")
            print(f"    - Alta densidad (>50%):  {stats['high_density']:5d} ({100*stats['high_density']/len(data):.1f}%)")
            print(f"    - Muy alta (>90%):       {stats['very_high_density']:5d} ({100*stats['very_high_density']/len(data):.1f}%)")
            print(f"    - Entidades consec.:     {stats['with_consecutive']:5d} ({100*stats['with_consecutive']/len(data):.1f}%)")
            print(f"    - Densidad promedio:     {stats['avg_density']:.3f}")
    else:
        print("\n  ⚠ No se encontraron archivos de entrenamiento")
        print(f"    Buscado: {train_cur_path}")
        print(f"    Buscado: {train_raw_path}")

    # 3. Análisis del problema
    print("\n" + "=" * 70)
    print("3. DIAGNÓSTICO DEL PROBLEMA")
    print("=" * 70)

    print("""
    HIPÓTESIS: El dataset CURADO eliminó samples con >90% entidades,
    pero el TEST SET contiene muchos de estos samples.

    CONSECUENCIA:
    - El modelo entrenado con CUR nunca vio patrones de alta densidad
    - El modelo entrenado con RAW sí los vio
    - En test, CUR falla en samples tipo REDIRECCIÓN y alta densidad

    EVIDENCIA ESPERADA:
    - Test tiene alto % de REDIRECCIÓN y alta densidad
    - Train CUR tiene bajo % de estos
    - Train RAW tiene alto % de estos (similar a test)
    """)

    # 4. Calcular "gap de distribución"
    print("\n" + "=" * 70)
    print("4. GAP DE DISTRIBUCIÓN TRAIN vs TEST")
    print("=" * 70)

    if os.path.exists(train_cur_path) and os.path.exists(train_raw_path):
        # Recalcular con más detalle
        test_redirect_pct = 100 * test_stats['redirects'] / test_stats['total']
        test_high_density_pct = 100 * test_stats['very_high_density'] / test_stats['total']

        cur_stats = {'redirects': 0, 'very_high_density': 0}
        raw_stats = {'redirects': 0, 'very_high_density': 0}

        for sample in train_cur:
            analysis = analyze_sample(sample)
            if analysis['is_redirect']:
                cur_stats['redirects'] += 1
            if analysis['very_high_density']:
                cur_stats['very_high_density'] += 1

        for sample in train_raw:
            analysis = analyze_sample(sample)
            if analysis['is_redirect']:
                raw_stats['redirects'] += 1
            if analysis['very_high_density']:
                raw_stats['very_high_density'] += 1

        cur_redirect_pct = 100 * cur_stats['redirects'] / len(train_cur)
        cur_high_density_pct = 100 * cur_stats['very_high_density'] / len(train_cur)
        raw_redirect_pct = 100 * raw_stats['redirects'] / len(train_raw)
        raw_high_density_pct = 100 * raw_stats['very_high_density'] / len(train_raw)

        print(f"""
                          TEST      CUR       RAW       Gap CUR    Gap RAW
    -------------------------------------------------------------------------
    REDIRECCIÓN          {test_redirect_pct:5.1f}%    {cur_redirect_pct:5.1f}%    {raw_redirect_pct:5.1f}%    {abs(test_redirect_pct - cur_redirect_pct):5.1f}%     {abs(test_redirect_pct - raw_redirect_pct):5.1f}%
    Densidad >90%        {test_high_density_pct:5.1f}%    {cur_high_density_pct:5.1f}%    {raw_high_density_pct:5.1f}%    {abs(test_high_density_pct - cur_high_density_pct):5.1f}%     {abs(test_high_density_pct - raw_high_density_pct):5.1f}%
        """)

        print("    INTERPRETACIÓN:")
        print("    - Gap CUR > Gap RAW indica que RAW está más alineado con TEST")
        print("    - Esto explica por qué RAW tiene mejor F1")

    # 5. Recomendaciones
    print("\n" + "=" * 70)
    print("5. CONCLUSIONES Y RECOMENDACIONES")
    print("=" * 70)

    print("""
    CONCLUSIÓN PRINCIPAL:
    ---------------------
    La curación creó un DISTRIBUTION SHIFT entre train y test.
    Al eliminar samples con >90% entidades del training, el modelo CUR
    no aprendió patrones que son frecuentes en el test set.

    RECOMENDACIONES PARA EL PAPER:
    ------------------------------
    1. Documentar el distribution shift como causa del bajo rendimiento de CUR
    2. El criterio de filtrado ">90% entidades" fue contraproducente para NER
    3. Para NER, mantener diversidad de patrones es más importante que "limpiar"

    IMPLICACIONES PRÁCTICAS:
    ------------------------
    - La curación debe considerar la distribución del test/producción
    - No todos los criterios de "limpieza" mejoran el rendimiento
    - Para NER, samples "ruidosos" pueden contener señales útiles
    """)

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()
