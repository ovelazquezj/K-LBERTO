#!/usr/bin/env python3
"""
Script 3: Generar Subsets Estratificados
Crear 7 datasets: 500, 750, 1000, 1500, 2000, 2500, 3000 samples
Mantener distribución de entity types balanceada
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
import random
from collections import Counter
import numpy as np

def calculate_entity_stats(samples, ner_tags_list):
    """Calcula estadísticas de entities para un conjunto de samples"""
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    entity_counts = Counter()
    total_tokens = 0
    total_entity_tokens = 0
    sentence_lengths = []
    entities_per_sentence = []
    
    for tokens, ner_tags in zip(samples, ner_tags_list):
        total_tokens += len(tokens)
        entity_count = 0
        sentence_lengths.append(len(tokens))
        
        for tag_id in ner_tags:
            if tag_id != 0:
                total_entity_tokens += 1
            tag_name = tag_names[tag_id]
            if tag_name.startswith('B-'):
                entity_type = tag_name[2:]
                entity_counts[entity_type] += 1
                entity_count += 1
        
        entities_per_sentence.append(entity_count)
    
    total_entities = sum(entity_counts.values())
    
    return {
        'total_entities': total_entities,
        'entity_distribution': dict(entity_counts),
        'entity_density': total_entity_tokens / total_tokens if total_tokens > 0 else 0,
        'avg_sentence_length': np.mean(sentence_lengths),
        'std_sentence_length': np.std(sentence_lengths),
        'avg_entities_per_sentence': np.mean(entities_per_sentence),
        'PER_pct': (entity_counts['PER'] / total_entities * 100) if total_entities > 0 else 0,
        'LOC_pct': (entity_counts['LOC'] / total_entities * 100) if total_entities > 0 else 0,
        'ORG_pct': (entity_counts['ORG'] / total_entities * 100) if total_entities > 0 else 0
    }

def stratified_sample(tokens_list, ner_tags_list, langs_list, target_size, seed=42):
    """
    Realiza muestreo estratificado basado en entity types dominantes
    """
    random.seed(seed)
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    # Clasificar cada sample por su entity type dominante
    samples_by_type = {'PER': [], 'LOC': [], 'ORG': [], 'MIXED': []}
    
    for idx, (tokens, ner_tags, langs) in enumerate(zip(tokens_list, ner_tags_list, langs_list)):
        entity_counts = Counter()
        for tag_id in ner_tags:
            tag_name = tag_names[tag_id]
            if tag_name.startswith('B-'):
                entity_type = tag_name[2:]
                entity_counts[entity_type] += 1
        
        if not entity_counts:
            continue
        
        # Determinar tipo dominante
        most_common = entity_counts.most_common(2)
        if len(most_common) == 1:
            dominant_type = most_common[0][0]
        elif most_common[0][1] > most_common[1][1] * 1.5:  # 50% más que el segundo
            dominant_type = most_common[0][0]
        else:
            dominant_type = 'MIXED'
        
        samples_by_type[dominant_type].append(idx)
    
    # Calcular proporciones en dataset completo
    total_samples = sum(len(indices) for indices in samples_by_type.values())
    proportions = {k: len(v) / total_samples for k, v in samples_by_type.items()}
    
    # Muestrear proporcionalmente
    selected_indices = []
    for entity_type, indices in samples_by_type.items():
        n_samples = int(target_size * proportions[entity_type])
        if n_samples > len(indices):
            n_samples = len(indices)
        selected = random.sample(indices, n_samples)
        selected_indices.extend(selected)
    
    # Si no llegamos al target, agregar samples aleatorios
    if len(selected_indices) < target_size:
        remaining_indices = set(range(len(tokens_list))) - set(selected_indices)
        additional = random.sample(list(remaining_indices), 
                                  min(target_size - len(selected_indices), len(remaining_indices)))
        selected_indices.extend(additional)
    
    # Si nos pasamos, recortar aleatoriamente
    if len(selected_indices) > target_size:
        selected_indices = random.sample(selected_indices, target_size)
    
    # Extraer samples seleccionados
    selected_tokens = [tokens_list[i] for i in selected_indices]
    selected_ner_tags = [ner_tags_list[i] for i in selected_indices]
    selected_langs = [langs_list[i] for i in selected_indices]
    
    return selected_tokens, selected_ner_tags, selected_langs

def generate_subsets():
    """Genera 7 subsets estratificados"""
    
    print("="*60)
    print("GENERANDO SUBSETS ESTRATIFICADOS")
    print("="*60)
    
    # Cargar dataset curado
    print("\n[1/4] Cargando dataset curado...")
    with open("data/curated_datasets/wikiann_spanish_curated.json", "r") as f:
        data = json.load(f)
    
    tokens_list = data['tokens']
    ner_tags_list = data['ner_tags']
    langs_list = data['langs']
    
    print(f"  - Total samples disponibles: {len(tokens_list)}")
    
    # Calcular estadísticas del dataset completo
    full_stats = calculate_entity_stats(tokens_list, ner_tags_list)
    print(f"\n  Distribución dataset completo:")
    print(f"    - PER: {full_stats['PER_pct']:.1f}%")
    print(f"    - LOC: {full_stats['LOC_pct']:.1f}%")
    print(f"    - ORG: {full_stats['ORG_pct']:.1f}%")
    print(f"    - Avg sentence length: {full_stats['avg_sentence_length']:.1f}")
    print(f"    - Entity density: {full_stats['entity_density']:.2%}")
    
    # Tamaños de subsets a generar
    subset_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000]
    
    print(f"\n[2/4] Generando {len(subset_sizes)} subsets estratificados...")
    
    all_stats = {}
    
    for size in subset_sizes:
        print(f"\n  Generando subset de {size} samples...")
        
        # Generar subset estratificado
        subset_tokens, subset_ner_tags, subset_langs = stratified_sample(
            tokens_list, ner_tags_list, langs_list, size
        )
        
        # Calcular estadísticas
        stats = calculate_entity_stats(subset_tokens, subset_ner_tags)
        all_stats[size] = stats
        
        print(f"    - PER: {stats['PER_pct']:.1f}% (diff: {abs(stats['PER_pct'] - full_stats['PER_pct']):.1f}%)")
        print(f"    - LOC: {stats['LOC_pct']:.1f}% (diff: {abs(stats['LOC_pct'] - full_stats['LOC_pct']):.1f}%)")
        print(f"    - ORG: {stats['ORG_pct']:.1f}% (diff: {abs(stats['ORG_pct'] - full_stats['ORG_pct']):.1f}%)")
        print(f"    - Avg length: {stats['avg_sentence_length']:.1f} (diff: {abs(stats['avg_sentence_length'] - full_stats['avg_sentence_length']):.1f})")
        
        # Guardar subset
        subset_data = {
            'tokens': subset_tokens,
            'ner_tags': subset_ner_tags,
            'langs': subset_langs,
            'size': size
        }
        
        filename = f"data/curated_datasets/subset_{size}.json"
        with open(filename, "w") as f:
            json.dump(subset_data, f, ensure_ascii=False, indent=2)
        
        print(f"    ✓ Guardado: {filename}")
    
    # Guardar estadísticas comparativas
    print(f"\n[3/4] Guardando estadísticas comparativas...")
    
    comparison = {
        'full_dataset': full_stats,
        'subsets': all_stats,
        'subset_sizes': subset_sizes
    }
    
    with open("data/curated_datasets/subset_statistics.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # Crear tabla de validación
    print(f"\n[4/4] Validando distribuciones...")
    print("\n" + "="*80)
    print(f"{'Size':<8} {'PER%':<8} {'LOC%':<8} {'ORG%':<8} {'AvgLen':<10} {'Density':<10}")
    print("="*80)
    print(f"{'FULL':<8} {full_stats['PER_pct']:<8.1f} {full_stats['LOC_pct']:<8.1f} "
          f"{full_stats['ORG_pct']:<8.1f} {full_stats['avg_sentence_length']:<10.1f} "
          f"{full_stats['entity_density']:<10.2%}")
    print("-"*80)
    
    max_diff = 0
    for size in subset_sizes:
        stats = all_stats[size]
        per_diff = abs(stats['PER_pct'] - full_stats['PER_pct'])
        loc_diff = abs(stats['LOC_pct'] - full_stats['LOC_pct'])
        org_diff = abs(stats['ORG_pct'] - full_stats['ORG_pct'])
        max_diff = max(max_diff, per_diff, loc_diff, org_diff)
        
        print(f"{size:<8} {stats['PER_pct']:<8.1f} {stats['LOC_pct']:<8.1f} "
              f"{stats['ORG_pct']:<8.1f} {stats['avg_sentence_length']:<10.1f} "
              f"{stats['entity_density']:<10.2%}")
    
    print("="*80)
    
    # Validación
    print(f"\n  Diferencia máxima en distribución entity types: {max_diff:.1f}%")
    if max_diff < 5.0:
        print("  ✓ VALIDACIÓN EXITOSA: Diferencia < 5%")
    else:
        print("  ⚠ ADVERTENCIA: Diferencia >= 5%")
    
    print("\n" + "="*60)
    print("GENERACIÓN DE SUBSETS COMPLETADA")
    print("="*60)
    print(f"\n{len(subset_sizes)} subsets guardados en: data/curated_datasets/")
    print("Estadísticas en: data/curated_datasets/subset_statistics.json")

if __name__ == "__main__":
    generate_subsets()
