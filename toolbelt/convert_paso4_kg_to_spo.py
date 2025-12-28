#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PASO_4 Knowledge Graph from JSON to K-BERT .spo format

Input:  ~/projects/RQ3_TASS/results/PASO_4_KNOWLEDGE_GRAPH_50K.json
        Format: JSON with triplets array
        {
          "triplets": [
            {"subject": "excelente", "relation": "sentiment_polarity", "object": "positive"},
            ...
          ]
        }

Output: ~/projects/K-LBERTO/brain/kgs/TASS_sentiment_88.spo
        Format: subject TAB relation TAB object (one per line)
        excelente	sentiment_polarity	positive
        horrible	sentiment_polarity	negative
        ...

K-BERT requirements:
  - Tab-separated values (no spaces)
  - One triplet per line
  - No header
  - Valid UTF-8 encoding
  - No control characters
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def validate_triplet(subject, relation, obj):
    """
    Validate triplet for K-BERT compatibility
    
    Returns: (is_valid, error_message)
    """
    
    # Check for empty values
    if not subject or not relation or not obj:
        return False, "Empty field in triplet"
    
    # Check for newlines (K-BERT can't handle them in values)
    if '\n' in subject or '\n' in relation or '\n' in obj:
        return False, "Newline character in triplet"
    
    # Check for tabs (would break TSV format)
    if '\t' in subject or '\t' in relation or '\t' in obj:
        return False, "Tab character in triplet"
    
    # Check for control characters
    for char in [subject, relation, obj]:
        for c in char:
            if ord(c) < 32 and c not in '\t\n':
                return False, f"Control character U+{ord(c):04X} in triplet"
    
    return True, None


def convert_kg_json_to_spo(input_file, output_file):
    """
    Convert Knowledge Graph from JSON to .spo format
    
    Args:
        input_file: Path to input JSON
        output_file: Path to output .spo file
    
    Returns:
        (total_triplets, valid_triplets, skipped_triplets, statistics)
    """
    
    print(f"\n📖 Reading JSON: {input_file}")
    
    # Load JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    triplets = kg_data.get('triplets', [])
    print(f"  ✓ Loaded {len(triplets)} triplets from JSON")
    
    # Statistics
    stats = {
        'total': len(triplets),
        'valid': 0,
        'skipped': 0,
        'relations': defaultdict(int),
        'subjects': set(),
        'objects': set(),
        'errors': defaultdict(int)
    }
    
    # Write .spo file
    print(f"\n✍️  Writing .spo: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        
        for triplet_id, triplet in enumerate(triplets):
            
            # Extract fields
            subject = triplet.get('subject', '').strip()
            relation = triplet.get('relation', '').strip()
            obj = triplet.get('object', '').strip()
            
            # Validate
            is_valid, error_msg = validate_triplet(subject, relation, obj)
            
            if not is_valid:
                stats['skipped'] += 1
                stats['errors'][error_msg] += 1
                if stats['skipped'] <= 5:  # Show first 5 errors
                    print(f"  ⚠ Triplet {triplet_id}: {error_msg}")
                continue
            
            # Write triplet (subject TAB relation TAB object)
            f_out.write(f"{subject}\t{relation}\t{obj}\n")
            
            # Update statistics
            stats['valid'] += 1
            stats['relations'][relation] += 1
            stats['subjects'].add(subject)
            stats['objects'].add(obj)
    
    return stats['total'], stats['valid'], stats['skipped'], stats


def main():
    """Main conversion pipeline"""
    
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH CONVERSION: JSON → .spo Format (K-BERT)")
    print("="*70)
    
    # Source and destination
    source_file = Path.home() / "projects" / "RQ3_TASS" / "results" / "PASO_4_KNOWLEDGE_GRAPH_50K.json"
    dest_dir = Path.home() / "projects" / "K-LBERTO" / "brain" / "kgs"
    dest_file = dest_dir / "TASS_sentiment_88.spo"
    
    # Validate source
    if not source_file.exists():
        print(f"\n❌ Source file not found: {source_file}")
        sys.exit(1)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Source: {source_file}")
    print(f"📁 Destination: {dest_file}")
    
    # Convert
    total, valid, skipped, stats = convert_kg_json_to_spo(str(source_file), str(dest_file))
    
    # Summary
    print(f"\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Total triplets:     {total}")
    print(f"Valid triplets:     {valid}")
    print(f"Skipped triplets:   {skipped}")
    print(f"Success rate:       {valid/total*100:.1f}%")
    
    if stats['errors']:
        print(f"\nError distribution:")
        for error_msg, count in stats['errors'].items():
            print(f"  • {error_msg}: {count}")
    
    print(f"\n📊 Relation Types ({len(stats['relations'])}):")
    for relation, count in sorted(stats['relations'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {relation:30} : {count:3} triplets")
    
    print(f"\n📈 Statistics:")
    print(f"    Unique subjects: {len(stats['subjects'])}")
    print(f"    Unique objects:  {len(stats['objects'])}")
    print(f"    Unique relations: {len(stats['relations'])}")
    
    # Verify output
    print(f"\n✅ Output File:")
    if dest_file.exists():
        with open(dest_file) as f:
            output_lines = len(f.readlines())
        
        file_size = dest_file.stat().st_size
        print(f"    File: {dest_file}")
        print(f"    Size: {file_size:,} bytes")
        print(f"    Lines: {output_lines}")
        
        # Show sample
        print(f"\n📄 Sample triplets (first 5):")
        with open(dest_file) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                subject, relation, obj = line.strip().split('\t')
                print(f"    {i+1}. {subject:20} → {relation:25} → {obj}")
    else:
        print(f"    ❌ File not created!")
        sys.exit(1)
    
    print(f"\n✅ CONVERSION COMPLETE")
    print(f"Ready for: python3 run_kbert_cls.py --kg_name {dest_file} ...")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
