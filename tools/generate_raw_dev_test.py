#!/usr/bin/env python3
"""Generate RAW (uncurated) WikiANN dev and test sets for ablation study"""
from datasets import load_dataset

ID_TO_TAG = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

def generate_raw_split(dataset, split: str, output_path: str):
    """Generate raw TSV from WikiANN split"""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in dataset[split]:
            tokens = sample['tokens']
            tags = [ID_TO_TAG[t] for t in sample['ner_tags']]
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
            count += 1
    return count

if __name__ == "__main__":
    print("Cargando WikiANN español...")
    dataset = load_dataset("unimelb-nlp/wikiann", "es", trust_remote_code=True)
    
    print("Generando dev_raw.tsv...")
    n_dev = generate_raw_split(dataset, 'validation', "data/wikiann_tsv/dev_raw.tsv")
    print(f"  → {n_dev} samples")
    
    print("Generando test_raw.tsv...")
    n_test = generate_raw_split(dataset, 'test', "data/wikiann_tsv/test_raw.tsv")
    print(f"  → {n_test} samples")
    
    print("Completado.")
