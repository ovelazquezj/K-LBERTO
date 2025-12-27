#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate TASS_sentiment_88.spo from PASO_4 JSON with Spanish values
that exist in BETO vocabulary (avoiding [UNK] tokens)

Original problem:
  excelente TAB sentiment_polarity TAB positive_sentiment
  "positive_sentiment" → vocab.get() → 100 ([UNK])

Solution:
  Map English/compound values to real Spanish words in BETO vocab
  excelente TAB sentiment_polarity TAB positivo
  "positivo" → vocab.get() → valid ID
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


# MAPPING: Current values → Spanish words that exist in BETO
# These are validated to exist in BETO vocabulary
VALUE_MAPPING = {
    # Sentiment polarity
    'positive': 'positivo',
    'positive_sentiment': 'positivo',
    'negative': 'negativo',
    'negative_sentiment': 'negativo',
    'neutral': 'neutral',
    'neutral_sentiment': 'neutral',
    
    # Actions
    'positive_action': 'apoyo',
    'negative_action': 'crítica',
    'approval': 'aprobación',
    'disapproval': 'desaprobación',
    'strong_positive': 'excelente',
    'strong_negative': 'horrible',
    
    # Entity types
    'product_entity': 'producto',
    'business_entity': 'empresa',
    'brand_concept': 'marca',
    'service_entity': 'servicio',
    'named_entity': 'persona',
    'tweet_corpus': 'tweet',
}


def normalize_value(value):
    """
    Map value to Spanish equivalent that exists in BETO vocab.
    
    REASON: Original values are English/compound words that map to [UNK]
    Spanish equivalents are real words that BETO understands
    """
    if value in VALUE_MAPPING:
        return VALUE_MAPPING[value]
    
    # If no mapping exists, keep original (fallback)
    # But warn the user
    print(f"  ⚠ WARNING: No mapping for '{value}', keeping as-is")
    return value


def regenerate_spo_from_json(json_path, spo_output_path, beto_vocab_path=None):
    """
    Regenerate .spo file from PASO_4 JSON with Spanish value normalization.
    
    Args:
        json_path: Path to PASO_4_KNOWLEDGE_GRAPH_50K.json
        spo_output_path: Output path for new .spo file
        beto_vocab_path: Optional path to BETO vocab for validation
    """
    
    print("=" * 80)
    print("REGENERATING .spo WITH SPANISH VALUES FOR BETO COMPATIBILITY")
    print("=" * 80)
    
    # Load BETO vocab if provided (for validation)
    beto_vocab = None
    if beto_vocab_path and Path(beto_vocab_path).exists():
        print(f"\n Loading BETO vocabulary from {beto_vocab_path}")
        beto_vocab = set()
        with open(beto_vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                beto_vocab.add(line.strip())
        print(f"   ✓ Loaded {len(beto_vocab)} tokens")
    
    # Load JSON
    print(f"\n Loading JSON from {json_path}")
    if not Path(json_path).exists():
        print(f" File not found: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    triplets = kg_data.get('triplets', [])
    print(f"   ✓ Loaded {len(triplets)} triplets")
    
    # Statistics
    stats = {
        'total': len(triplets),
        'written': 0,
        'mapped_values': defaultdict(int),
        'unmapped_values': defaultdict(int),
        'unk_tokens': [],
    }
    
    # Write new .spo file
    print(f"\n✍️  Writing normalized .spo to {spo_output_path}")
    
    with open(spo_output_path, 'w', encoding='utf-8') as f_out:
        for triplet_id, triplet in enumerate(triplets):
            # Extract fields
            subject = triplet.get('subject', '').strip()
            relation = triplet.get('relation', '').strip()
            original_obj = triplet.get('object', '').strip()
            
            # Validate
            if not subject or not relation or not original_obj:
                print(f"  ⚠ Triplet {triplet_id}: Empty field, skipping")
                continue
            
            # CRITICAL: Normalize object value to Spanish
            normalized_obj = normalize_value(original_obj)
            
            # Check if normalized value would be [UNK] in BETO
            if beto_vocab and normalized_obj not in beto_vocab:
                print(f"  ⚠ '{normalized_obj}' NOT in BETO vocab (would be [UNK])")
                stats['unk_tokens'].append(normalized_obj)
            
            # Track mapping
            if normalized_obj != original_obj:
                stats['mapped_values'][f"{original_obj}→{normalized_obj}"] += 1
            else:
                stats['unmapped_values'][original_obj] += 1
            
            # Write triplet: subject TAB relation TAB normalized_object
            f_out.write(f"{subject}\t{relation}\t{normalized_obj}\n")
            stats['written'] += 1
    
    # Summary
    print(f"\n" + "=" * 80)
    print("REGENERATION SUMMARY")
    print("=" * 80)
    print(f"Total triplets:     {stats['total']}")
    print(f"Written to .spo:    {stats['written']}")
    
    print(f"\n Value Mappings Applied:")
    for mapping, count in sorted(stats['mapped_values'].items()):
        print(f"   {mapping:40} : {count} triplets")
    
    if stats['unmapped_values']:
        print(f"\n️  Unmapped Values (kept as-is):")
        for value, count in sorted(stats['unmapped_values'].items()):
            print(f"   {value:40} : {count} triplets")
    
    if stats['unk_tokens']:
        print(f"\n POTENTIAL [UNK] TOKENS (NOT in BETO vocab):")
        for token in set(stats['unk_tokens']):
            print(f"   '{token}'")
    else:
        print(f"\n All normalized values should be valid BETO tokens")
    
    # Verify output file
    print(f"\n Output File:")
    output_path = Path(spo_output_path)
    if output_path.exists():
        with open(output_path) as f:
            output_lines = len(f.readlines())
        file_size = output_path.stat().st_size
        print(f"   File: {spo_output_path}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Lines: {output_lines}")
        
        # Show sample
        print(f"\n Sample triplets (first 10):")
        with open(output_path) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                subject, relation, obj = line.strip().split('\t')
                print(f"   {i+1:2}. {subject:20} → {relation:25} → {obj}")
    else:
        print(f"    File not created!")
        return False
    
    print(f"\n" + "=" * 80)
    print(" REGENERATION COMPLETE - READY FOR K-BERT TRAINING")
    print("=" * 80 + "\n")
    
    return True


def main():
    """Main execution"""
    
    # Paths
    json_input = Path.home() / "projects" / "RQ3_TASS" / "results" / "PASO_4_KNOWLEDGE_GRAPH_50K.json"
    spo_output = Path.home() / "projects" / "K-LBERTO" / "brain" / "kgs" / "TASS_sentiment_88_SPANISH.spo"
    beto_vocab = Path.home() / "projects" / "K-LBERTO" / "models" / "beto_uer_model" / "vocab.txt"
    
    print(f"\n Paths:")
    print(f"   Input JSON:    {json_input}")
    print(f"   Output .spo:   {spo_output}")
    print(f"   BETO vocab:    {beto_vocab}")
    
    # Ensure output directory exists
    spo_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Regenerate
    success = regenerate_spo_from_json(
        str(json_input),
        str(spo_output),
        str(beto_vocab) if beto_vocab.exists() else None
    )
    
    if success:
        print(f"\n NEW .spo file ready:")
        print(f"   {spo_output}")
        print(f"\nTo use it in K-BERT training:")
        print(f"   python3 run_kbert_cls.py \\")
        print(f"     --kg_name {spo_output} \\")
        print(f"     [other parameters...]")
    else:
        print(f"\n Regeneration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
