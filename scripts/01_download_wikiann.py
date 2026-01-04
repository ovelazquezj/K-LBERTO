#!/usr/bin/env python3
"""
Script 1: Descargar WikiANN Spanish Dataset
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

from datasets import load_dataset
import json
from collections import Counter

def download_wikiann_spanish():
    """Descarga WikiANN Spanish y genera estadísticas iniciales"""
    
    print("="*60)
    print("DESCARGANDO WikiANN SPANISH")
    print("="*60)
    
    # Descargar dataset
    print("\n[1/4] Descargando dataset...")
    dataset = load_dataset("wikiann", "es")
    
    print(f"✓ Dataset descargado")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Validation: {len(dataset['validation'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")
    
    # Analizar estructura
    print("\n[2/4] Analizando estructura...")
    sample = dataset['train'][0]
    print(f"  - Keys: {sample.keys()}")
    print(f"  - Ejemplo tokens: {sample['tokens'][:10]}")
    print(f"  - Ejemplo tags: {sample['ner_tags'][:10]}")
    
    # Mapeo de tags IOB2
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    # Estadísticas de entity types
    print("\n[3/4] Contando entity types...")
    entity_counts = Counter()
    
    for split in ['train', 'validation', 'test']:
        for sample in dataset[split]:
            for tag_id in sample['ner_tags']:
                tag_name = tag_names[tag_id]
                if tag_name.startswith('B-'):
                    entity_type = tag_name[2:]
                    entity_counts[entity_type] += 1
    
    total_entities = sum(entity_counts.values())
    print(f"  - Total entities: {total_entities}")
    for entity_type, count in entity_counts.most_common():
        pct = (count / total_entities) * 100
        print(f"  - {entity_type}: {count} ({pct:.1f}%)")
    
    # Guardar dataset
    print("\n[4/4] Guardando dataset localmente...")
    dataset.save_to_disk("data/wikiann_spanish/raw")
    
    # Guardar estadísticas
    stats = {
        "splits": {
            "train": len(dataset['train']),
            "validation": len(dataset['validation']),
            "test": len(dataset['test'])
        },
        "entity_distribution": dict(entity_counts),
        "tag_names": tag_names
    }
    
    with open("data/wikiann_spanish/initial_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Dataset guardado en: data/wikiann_spanish/raw")
    print("✓ Estadísticas en: data/wikiann_spanish/initial_stats.json")
    print("\n" + "="*60)
    print("DESCARGA COMPLETADA")
    print("="*60)

if __name__ == "__main__":
    download_wikiann_spanish()
