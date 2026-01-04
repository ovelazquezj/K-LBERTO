#!/usr/bin/env python3
"""
Script 7: Convertir datasets JSON WikiANN a formato TSV K-BERT NER (español)
Formato: palabras completas separadas por espacios
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
import os
from datasets import load_from_disk

def convert_json_to_tsv(json_path, output_path):
    """
    Convierte dataset JSON WikiANN a formato TSV K-BERT
    
    Formato TSV para NER español:
    label	text
    B-PER I-PER O B-LOC	José García vive en Madrid
    """
    
    print(f"  Convirtiendo {os.path.basename(json_path)}...")
    
    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tokens_list = data['tokens']
    ner_tags_list = data['ner_tags']
    
    # Mapeo de IDs a nombres de tags IOB2
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    # Escribir TSV
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header (columnas: label, text)
        f.write("label\ttext\n")
        
        # Cada sample
        for tokens, ner_tags in zip(tokens_list, ner_tags_list):
            # Tags IOB2 separados por espacios
            tag_str = ' '.join([tag_names[tag_id] for tag_id in ner_tags])
            
            # Palabras separadas por espacios
            token_str = ' '.join(tokens)
            
            # Escribir línea: tags\tpalabras
            f.write(f"{tag_str}\t{token_str}\n")
    
    print(f"    ✓ {len(tokens_list)} samples")
    return len(tokens_list)


def create_validation_set():
    """Crear validation/test sets desde WikiANN HuggingFace"""
    print("\n  Creando validation y test sets desde WikiANN (HuggingFace)...")
    
    from datasets import load_dataset
    
    # Cargar directamente desde HuggingFace
    print("    Descargando WikiANN (puede tardar un momento)...")
    wikiann = load_dataset("wikiann", "es")
    
    # Validation: primeros 1000 de validation split
    val_tokens = wikiann['validation']['tokens'][:1000]
    val_tags = wikiann['validation']['ner_tags'][:1000]
    
    # Test: primeros 1000 de test split
    test_tokens = wikiann['test']['tokens'][:1000]
    test_tags = wikiann['test']['ner_tags'][:1000]
    
    # Guardar como JSON temporal
    val_data = {
        'tokens': val_tokens,
        'ner_tags': val_tags,
        'langs': [['es'] * len(t) for t in val_tokens]
    }
    
    test_data = {
        'tokens': test_tokens,
        'ner_tags': test_tags,
        'langs': [['es'] * len(t) for t in test_tokens]
    }
    
    os.makedirs("data/curated_datasets", exist_ok=True)
    
    with open("data/curated_datasets/validation_temp.json", 'w') as f:
        json.dump(val_data, f)
    
    with open("data/curated_datasets/test_temp.json", 'w') as f:
        json.dump(test_data, f)
    
    print("    ✓ Validation: 1000 samples")
    print("    ✓ Test: 1000 samples")


def convert_all_subsets():
    """Convierte todos los subsets + validation/test"""
    
    print("="*60)
    print("CONVIRTIENDO DATASETS JSON → TSV (NER Español)")
    print("="*60)
    
    # Crear directorio de salida
    output_dir = "data/wikiann_tsv"
    os.makedirs(output_dir, exist_ok=True)
    
    # Subset sizes
    subset_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000]
    
    print("\n[1/3] Convirtiendo subsets de training...")
    total_samples = {}
    
    for size in subset_sizes:
        json_path = f"data/curated_datasets/subset_{size}.json"
        tsv_path = f"{output_dir}/train_{size}.tsv"
        
        if not os.path.exists(json_path):
            print(f"  ⚠ {json_path} no existe - saltando")
            continue
        
        count = convert_json_to_tsv(json_path, tsv_path)
        total_samples[size] = count
    
    # Crear validation/test si no existen
    print("\n[2/3] Preparando validation y test sets...")
    if not os.path.exists("data/curated_datasets/validation_temp.json"):
        create_validation_set()
    
    # Convertir validation
    print("\n[3/3] Convirtiendo validation y test...")
    
    val_json = "data/curated_datasets/validation_temp.json"
    if os.path.exists(val_json):
        val_count = convert_json_to_tsv(val_json, f"{output_dir}/dev.tsv")
        print(f"    ✓ dev.tsv: {val_count} samples")
    
    test_json = "data/curated_datasets/test_temp.json"
    if os.path.exists(test_json):
        test_count = convert_json_to_tsv(test_json, f"{output_dir}/test.tsv")
        print(f"    ✓ test.tsv: {test_count} samples")
    
    # Resumen
    print("\n" + "="*60)
    print("CONVERSIÓN COMPLETADA")
    print("="*60)
    print(f"\nArchivos TSV generados en: {output_dir}/")
    print("\nTraining sets:")
    for size in subset_sizes:
        if size in total_samples:
            print(f"  - train_{size}.tsv: {total_samples[size]} samples")
    
    if os.path.exists(f"{output_dir}/dev.tsv"):
        print(f"\nValidation/Test sets:")
        print(f"  - dev.tsv: {val_count} samples")
        print(f"  - test.tsv: {test_count} samples")
    
    # Mostrar ejemplo
    print("\n" + "="*60)
    print("EJEMPLO DE FORMATO (train_500.tsv):")
    print("="*60)
    example_file = f"{output_dir}/train_500.tsv"
    if os.path.exists(example_file):
        with open(example_file, 'r') as f:
            for i, line in enumerate(f):
                print(line.rstrip())
                if i >= 3:  # Header + 3 ejemplos
                    break


if __name__ == "__main__":
    try:
        convert_all_subsets()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
