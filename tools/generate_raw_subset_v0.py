#!/usr/bin/env python3

"""
Generate RAW (uncurated) WikiANN subset for ablation study.

Generates data in K-LBERTO format:
    label<TAB>text
    O B-PER I-PER<TAB>REDIRECCIÓN Juan Pérez

This is the UNCURATED version - no filtering applied.
Used for ablation study comparing curated vs raw data.
"""
from datasets import load_dataset
from pathlib import Path

ID_TO_TAG = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}


def generate_raw_subset(n_samples: int, output_path: str, split: str = 'train'):
    """
    Extract n samples WITHOUT any curation filters.
    
    Args:
        n_samples: Number of samples to extract
        output_path: Output TSV file path
        split: Dataset split ('train', 'validation', 'test')
    """
    print(f"Loading WikiANN Spanish ({split})...")
    dataset = load_dataset("unimelb-nlp/wikiann", "es", trust_remote_code=True)
    
    print(f"Extracting {n_samples} raw samples (no filtering)...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("label\ttext\n")  # Header required by K-LBERTO

        
        for i, sample in enumerate(dataset[split]):
            if i >= n_samples:
                break
            
            tokens = sample['tokens']
            tags = [ID_TO_TAG[t] for t in sample['ner_tags']]
            
            # K-LBERTO format: "O B-PER I-PER\tREDIRECCIÓN Juan Pérez"

            labels_str = ' '.join(tags)
            text_str = ' '.join(tokens)
            f.write(f"{labels_str}\t{text_str}\n")
    
    print(f"✅ Saved: {output_path} ({n_samples} samples)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate raw WikiANN subset for K-LBERTO")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/wikiann_tsv/train_raw_2000.tsv", help="Output path")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'validation', 'test'], help="Dataset split")
    
    args = parser.parse_args()
    generate_raw_subset(args.n_samples, args.output, args.split)
