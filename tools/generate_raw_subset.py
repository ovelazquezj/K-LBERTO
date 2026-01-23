#!/usr/bin/env python3
"""
Generate WikiANN subsets for K-LBERTO ablation studies.
Creates datasets in K-LBERTO format with flexible size and curation options.

FIXED VERSION: Implements real curation filters
- Removes "REDIRECCIÓN" samples
- Filters high entity density (>90% entities)
- Removes duplicates
- Filters suspicious patterns

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

def should_include_sample(sample, curated: bool = False, seen_samples: set = None):
    """
    Apply curation filters if requested.

    Curation criteria:
    1. Length constraints (3-128 tokens)
    2. Has at least one named entity
    3. No "REDIRECCIÓN" token (Wikipedia redirect artifacts)
    4. Entity density <= 90% (avoid low-quality samples)
    5. No duplicates (by text hash)
    6. No suspicious patterns (parentheses as entities, etc.)
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

    # Filter 4: Remove Wikipedia redirects
    if "REDIRECCIÓN" in tokens or "REDIRECT" in tokens:
        return False

    # Filter 5: Entity density check (avoid >90% entities)
    entity_count = sum(1 for tag in tags if tag != 0)
    entity_density = entity_count / len(tokens)
    if entity_density > 0.90:
        return False

    # Filter 6: Duplicate detection (by text)
    if seen_samples is not None:
        text_hash = ' '.join(tokens)
        if text_hash in seen_samples:
            return False
        seen_samples.add(text_hash)

    # Filter 7: Suspicious patterns
    # - Single character tokens tagged as entities
    # - Parentheses/brackets tagged as part of entities
    suspicious_tokens = {'(', ')', '[', ']', '{', '}', '*', '**', '***'}
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # Single char as entity (except common cases like country codes)
        if len(token) == 1 and tag != 0 and token not in {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}:
            return False
        
        # Punctuation as entities
        if token in suspicious_tokens and tag != 0:
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
    has_redirection = 0
    high_entity_density = 0

    for line in lines:
        labels, text = line.strip().split('\t')
        tokens = text.split()
        total_tokens += len(tokens)

        # Count REDIRECCIÓN (should be 0 in curated)
        if "REDIRECCIÓN" in text or "REDIRECT" in text:
            has_redirection += 1

        # Count high entity density
        label_list = labels.split()
        entity_count = sum(1 for l in label_list if l != 'O')
        if entity_count / len(label_list) > 0.90:
            high_entity_density += 1

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
        'total_entities': sum(entity_counts.values()),
        'redirection_samples': has_redirection,
        'high_density_samples': high_entity_density
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

    # Generate subset
    print(f"\n{'Filtering and extracting' if curated else 'Extracting'} samples...")

    accepted = 0
    processed = 0
    seen_samples = set() if curated else None

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("label\ttext\n")  # K-LBERTO header

        for sample in dataset[split]:
            if processed >= total_available:
                break

            processed += 1

            if not should_include_sample(sample, curated, seen_samples):
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
        filtered = processed - accepted
        print(f"   Filtered:  {filtered:,} samples ({100*filtered/processed:.1f}%)")

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
    print(f"  Samples:           {stats['samples']:,}")
    print(f"  Tokens:            {stats['tokens']:,}")
    print(f"  Avg length:        {stats['avg_length']:.1f} tokens/sample")
    print(f"  Total entities:    {stats['total_entities']:,}")
    print(f"    PER:             {stats['entities']['PER']:,}")
    print(f"    ORG:             {stats['entities']['ORG']:,}")
    print(f"    LOC:             {stats['entities']['LOC']:,}")
    print(f"  REDIRECCIÓN count: {stats['redirection_samples']:,}")
    print(f"  High density (>90%): {stats['high_density_samples']:,}")

    if curated:
        if stats['redirection_samples'] > 0:
            print(f"  ⚠️  WARNING: {stats['redirection_samples']} REDIRECCIÓN samples in curated dataset!")
        if stats['high_density_samples'] > 0:
            print(f"  ⚠️  WARNING: {stats['high_density_samples']} high-density samples in curated dataset!")

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
    
