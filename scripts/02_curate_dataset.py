#!/usr/bin/env python3
"""
Script 2: Curar WikiANN Spanish Dataset
Aplicar 5 criterios de filtrado para obtener ~3000 samples de alta calidad
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

from datasets import load_from_disk
import json
from collections import Counter

def validate_bio_tags(tokens, ner_tags, tag_names):
    """Valida que las tags IOB2 sean consistentes"""
    for i, tag_id in enumerate(ner_tags):
        tag = tag_names[tag_id]
        
        # I-TAG debe seguir a B-TAG o I-TAG del mismo tipo
        if tag.startswith('I-'):
            if i == 0:
                return False  # I-TAG no puede ser primera
            prev_tag = tag_names[ner_tags[i-1]]
            entity_type = tag[2:]
            if not (prev_tag == f'B-{entity_type}' or prev_tag == f'I-{entity_type}'):
                return False
    
    return True

def curate_dataset():
    """Aplica 5 criterios de curación"""
    
    print("="*60)
    print("CURANDO WikiANN SPANISH")
    print("="*60)
    
    # Cargar dataset
    print("\n[1/6] Cargando dataset...")
    dataset = load_from_disk("data/wikiann_spanish/raw")
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    # Usar solo train split
    data = dataset['train']
    print(f"  - Samples iniciales (train): {len(data)}")
    
    # Contadores
    removed = {
        "criterio_1_menos_3_tokens": 0,
        "criterio_2_sin_entities": 0,
        "criterio_3_bio_inconsistente": 0,
        "criterio_4_duplicados": 0,
        "criterio_5_mas_90pct_entities": 0
    }
    
    curated_samples = []
    seen_samples = set()
    
    print("\n[2/6] Aplicando criterios de filtrado...")
    print("  Procesando samples...")
    
    for idx, sample in enumerate(data):
        if (idx + 1) % 5000 == 0:
            print(f"    {idx + 1}/{len(data)} procesados...")
        
        tokens = sample['tokens']
        ner_tags = sample['ner_tags']
        
        # Criterio 1: <3 tokens
        if len(tokens) < 3:
            removed["criterio_1_menos_3_tokens"] += 1
            continue
        
        # Criterio 2: sin entities
        has_entity = any(tag != 0 for tag in ner_tags)
        if not has_entity:
            removed["criterio_2_sin_entities"] += 1
            continue
        
        # Criterio 3: BIO inconsistente
        if not validate_bio_tags(tokens, ner_tags, tag_names):
            removed["criterio_3_bio_inconsistente"] += 1
            continue
        
        # Criterio 4: duplicados
        sample_str = " ".join(tokens)
        if sample_str in seen_samples:
            removed["criterio_4_duplicados"] += 1
            continue
        seen_samples.add(sample_str)
        
        # Criterio 5: >90% entities
        entity_tokens = sum(1 for tag in ner_tags if tag != 0)
        entity_pct = entity_tokens / len(tokens)
        if entity_pct > 0.9:
            removed["criterio_5_mas_90pct_entities"] += 1
            continue
        
        # Sample pasó todos los criterios
        curated_samples.append(sample)
    
    print(f"\n  ✓ Samples curados: {len(curated_samples)}")
    print(f"\n  Removidos por criterio:")
    for criterio, count in removed.items():
        print(f"    - {criterio}: {count}")
    
    # Verificar distribución de entity types
    print("\n[3/6] Verificando distribución de entities...")
    entity_counts = Counter()
    total_tokens = 0
    total_entity_tokens = 0
    
    for sample in curated_samples:
        total_tokens += len(sample['tokens'])
        for tag_id in sample['ner_tags']:
            if tag_id != 0:
                total_entity_tokens += 1
            tag_name = tag_names[tag_id]
            if tag_name.startswith('B-'):
                entity_type = tag_name[2:]
                entity_counts[entity_type] += 1
    
    total_entities = sum(entity_counts.values())
    entity_density = total_entity_tokens / total_tokens
    
    print(f"  - Total entities: {total_entities}")
    print(f"  - Entity density: {entity_density:.2%}")
    for entity_type, count in entity_counts.most_common():
        pct = (count / total_entities) * 100
        print(f"  - {entity_type}: {count} ({pct:.1f}%)")
    
    # Guardar dataset curado
    print("\n[4/6] Guardando dataset curado...")
    
    curated_data = {
        'tokens': [s['tokens'] for s in curated_samples],
        'ner_tags': [s['ner_tags'] for s in curated_samples],
        'langs': [s['langs'] for s in curated_samples]
    }
    
    with open("data/curated_datasets/wikiann_spanish_curated.json", "w") as f:
        json.dump(curated_data, f, ensure_ascii=False, indent=2)
    
    # Guardar estadísticas
    print("\n[5/6] Guardando estadísticas...")
    stats = {
        "samples_iniciales": len(data),
        "samples_curados": len(curated_samples),
        "samples_removidos_total": len(data) - len(curated_samples),
        "tasa_retencion": len(curated_samples) / len(data),
        "removidos_por_criterio": removed,
        "entity_distribution": dict(entity_counts),
        "entity_density": entity_density
    }
    
    with open("data/curated_datasets/curation_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n[6/6] Verificando target...")
    if 2500 <= len(curated_samples) <= 3500:
        print(f"  ✓ Target alcanzado: {len(curated_samples)} samples (target: ~3000)")
    elif len(curated_samples) > 3500:
        print(f"  ⚠ Por encima del target: {len(curated_samples)} samples")
        print(f"    Se puede hacer subset aleatorio para llegar a 3000")
    else:
        print(f"  ⚠ Por debajo del target: {len(curated_samples)} samples")
        print(f"    Considerar relajar criterios de filtrado")
    
    print("\n" + "="*60)
    print("CURACIÓN COMPLETADA")
    print("="*60)
    print(f"\nDataset guardado en: data/curated_datasets/wikiann_spanish_curated.json")
    print(f"Estadísticas en: data/curated_datasets/curation_stats.json")

if __name__ == "__main__":
    curate_dataset()
