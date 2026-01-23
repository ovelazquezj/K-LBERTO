#!/usr/bin/env python3
"""
Generate WikiANN subsets for K-LBERTO ablation studies.
Creates datasets in K-LBERTO format with flexible size and curation options.

Format:
    label<TAB>text
    O B-PER I-PER<TAB>REDIRECCIÓN Juan Pérez

Usage:
    # Raw (uncurated) dataset D=4000
    python tools/generate_raw_subset.py --size 4000 --output data/wikiann_tsv/train_raw_4000.tsv
    
    # Curated dataset D=4000
    python tools/generate_raw_subset.py --size 4000 --curated --output data/wikiann_tsv/train_4000.tsv
"""
import sys
from pathlib import Path
from datasets import load_dataset

ID_TO_TAG = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

def should_include_sample(sample, curated: bool = False):
    """
    Apply curation filters if requested.
    
    Curation criteria (based on original K-LBERTO preprocessing):
    - Has at least one named entity
    - Sentence length >= 3 tokens
    - Sentence length <= 128 tokens (K-BERT limit)
    - No malformed sequences
    """
    if not curated:
        return True  # Include all samples for raw dataset
    
    tokens = sample['tokens']
    tags = sample['ner_tags']
    
    # Filter 1: Length constraints
    if len(tokens) < 3 or len(tokens) > 128:
        return False
    
    # Filter 2: Must have at least one named entity (not all O tags)
    has_entity = any(tag != 0 for tag in tags)
    if not has_entity:
        return False
    
    # Filter 3: Token/tag alignment
    if len(tokens) != len(tags):
        return False
    
    return True

def validate_format(filepath: str) -> bool:
    """Validate generated file has correct K-LBERTO format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Check header
        if not lines or lines[0].strip() != "label\ttext":
            print("❌ ERROR: Missing or incorrect header")
            return False
        
        # Check sample lines
        for i, line in enumerate(lines[1:6], start=2):  # Check first 5 samples
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"❌ ERROR: Line {i} doesn't have exactly 2 TAB-separated fields")
                return False
            
            labels, text = parts
            label_tokens = labels.split()
            text_tokens = text.split()
            
            if len(label_tokens) != len(text_tokens):
                print(f"❌ ERROR: Line {i} label/text token count mismatch")
                print(f"   Labels: {len(label_tokens)} tokens")
                print(f"   Text:   {len(text_tokens)} tokens")
                return False
    
    return True

def compute_statistics(filepath: str) -> dict:
    """Compute dataset statistics."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
    
    total_samples = len(lines)
    total_tokens = 0
    entity_counts = {'PER': 0, 'ORG': 0, 'LOC': 0}
    
    for line in lines:
        labels, text = line.strip().split('\t')
        tokens = text.split()
        total_tokens += len(tokens)
        
        # Count entities
        for label in labels.split():
            if label.startswith('B-'):
                entity_type = label.split('-')[1]
                entity_counts[entity_type] += 1
    
    return {
        'samples': total_samples,
        'tokens': total_tokens,
        'avg_length': total_tokens / total_samples if total_samples > 0 else 0,
        'entities': entity_counts,
        'total_entities': sum(entity_counts.values())
    }

def generate_subset(size: int, output_path: str, curated: bool = False, split: str = 'train'):
    """
    Generate WikiANN subset with optional curation.
    
    Args:
        size: Target number of samples
        output_path: Output TSV file path
        curated: Apply curation filters if True
        split: Dataset split ('train', 'validation', 'test')
    """
    print(f"\n{'='*60}")
    print(f"Generating {'CURATED' if curated else 'RAW'} WikiANN subset")
    print(f"{'='*60}")
    print(f"Target size: {size:,} samples")
    print(f"Output: {output_path}")
    print(f"Split: {split}")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading WikiANN Spanish ({split})...")
    dataset = load_dataset("unimelb-nlp/wikiann", "es", trust_remote_code=True)
    total_available = len(dataset[split])
    print(f"Available samples: {total_available:,}")
    
    if size > total_available:
        print(f"⚠️  WARNING: Requested {size} samples but only {total_available} available")
        size = total_available
    
    # Generate subset
    print(f"\n{'Filtering and extracting' if curated else 'Extracting'} samples...")
    
    accepted = 0
    processed = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("label\ttext\n")  # K-LBERTO header
        
        for sample in dataset[split]:
            processed += 1
            
            if not should_include_sample(sample, curated):
                continue
            
            tokens = sample['tokens']
            tags = [ID_TO_TAG[t] for t in sample['ner_tags']]
            
            # K-LBERTO format: "O B-PER I-PER\tREDIRECCIÓN Juan Pérez"
            labels_str = ' '.join(tags)
            text_str = ' '.join(tokens)
            f.write(f"{labels_str}\t{text_str}\n")
            
            accepted += 1
            
            if accepted >= size:
                break
            
            # Progress indicator
            if accepted % 500 == 0:
                print(f"  Progress: {accepted:,}/{size:,} samples", end='\r')
    
    print(f"\n✅ Extraction complete!")
    print(f"   Processed: {processed:,} samples")
    print(f"   Accepted:  {accepted:,} samples")
    if curated:
        print(f"   Filtered:  {processed - accepted:,} samples ({100*(processed-accepted)/processed:.1f}%)")
    
    # Validate format
    print(f"\nValidating format...")
    if validate_format(output_path):
        print(f"✅ Format validation passed")
    else:
        print(f"❌ Format validation FAILED")
        sys.exit(1)
    
    # Compute statistics
    print(f"\nDataset statistics:")
    stats = compute_statistics(output_path)
    print(f"  Samples:       {stats['samples']:,}")
    print(f"  Tokens:        {stats['tokens']:,}")
    print(f"  Avg length:    {stats['avg_length']:.1f} tokens/sample")
    print(f"  Total entities: {stats['total_entities']:,}")
    print(f"    PER:         {stats['entities']['PER']:,}")
    print(f"    ORG:         {stats['entities']['ORG']:,}")
    print(f"    LOC:         {stats['entities']['LOC']:,}")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate WikiANN subsets for K-LBERTO ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raw (uncurated) D=4000
  %(prog)s --size 4000 --output data/wikiann_tsv/train_raw_4000.tsv
  
  # Curated D=4000  
  %(prog)s --size 4000 --curated --output data/wikiann_tsv/train_4000.tsv
  
  # Development/test sets
  %(prog)s --size 1000 --curated --split validation --output data/wikiann_tsv/dev_1000.tsv
        """
    )
    
    parser.add_argument(
        "--size", 
        type=int, 
        required=True,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output TSV file path (e.g., data/wikiann_tsv/train_4000.tsv)"
    )
    
    parser.add_argument(
        "--curated",
        action="store_true",
        help="Apply curation filters (length, entity presence, quality checks)"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        choices=['train', 'validation', 'test'],
        help="Dataset split to extract from (default: train)"
    )
    
    args = parser.parse_args()
    
    generate_subset(
        size=args.size,
        output_path=args.output,
        curated=args.curated,
        split=args.split
    )
