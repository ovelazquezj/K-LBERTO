#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert TASS dataset from RQ3_TASS format to K-BERT classification format

Input:  ~/projects/RQ3_TASS/data/raw/{train,test}.tsv
        Format: label TAB text
        Labels: N, P, NEU, NONE (strings)

Output: ~/projects/K-LBERTO/datasets/tass_spanish/{train,test}.tsv
        Format: label TAB text_a
        Labels: 0, 1, 2, 3 (numeric)

Mapping:
  N    → 0 (Negative)
  P    → 1 (Positive)
  NEU  → 2 (Neutral)
  NONE → 3 (No Opinion)
"""

import os
import sys
from pathlib import Path

# Label mapping: string → numeric
LABEL_MAP = {
    'N': 0,      # Negative
    'P': 1,      # Positive
    'NEU': 2,    # Neutral
    'NONE': 3    # No Opinion
}

def convert_tass_file(input_file, output_file):
    """
    Convert single TASS TSV file to K-BERT format
    
    Args:
        input_file: Path to input TSV (label TAB text)
        output_file: Path to output TSV (label TAB text_a)
    
    Returns:
        (total_lines, valid_lines, skipped_lines)
    """
    
    total = 0
    valid = 0
    skipped = 0
    
    print(f"\n📖 Reading: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            
            # Write header
            f_out.write("label\ttext_a\n")
            
            for line_id, line in enumerate(f_in):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip header
                if line_id == 0 and line.startswith("label"):
                    print("  ✓ Header detected, skipping")
                    continue
                
                total += 1
                
                # Parse TSV
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"  ⚠ Line {total}: Invalid format (expected 2 columns)")
                    skipped += 1
                    continue
                
                label_str, text = parts
                label_str = label_str.strip()
                text = text.strip()
                
                # Validate label
                if label_str not in LABEL_MAP:
                    print(f"  ⚠ Line {total}: Unknown label '{label_str}'")
                    skipped += 1
                    continue
                
                # Validate text
                if not text:
                    print(f"  ⚠ Line {total}: Empty text")
                    skipped += 1
                    continue
                
                # Convert label to numeric
                label_num = LABEL_MAP[label_str]
                
                # Write converted line
                f_out.write(f"{label_num}\t{text}\n")
                valid += 1
    
    return total, valid, skipped


def main():
    """Main conversion pipeline"""
    
    print("\n" + "="*70)
    print("TASS DATASET CONVERSION: RQ3_TASS → K-LBERTO Format")
    print("="*70)
    
    # Source and destination
    source_dir = Path.home() / "projects" / "RQ3_TASS" / "data" / "raw"
    dest_dir = Path.home() / "projects" / "K-LBERTO" / "datasets" / "tass_spanish"
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Source: {source_dir}")
    print(f"📁 Destination: {dest_dir}")
    
    # Label mapping reference
    print(f"\n🏷️  Label Mapping:")
    for label_str, label_num in LABEL_MAP.items():
        print(f"    {label_str:6} → {label_num}")
    
    # Process train and test
    datasets = {
        'train': source_dir / 'train.tsv',
        'test': source_dir / 'test.tsv'
    }
    
    total_stats = {'total': 0, 'valid': 0, 'skipped': 0}
    
    for split_name, input_file in datasets.items():
        
        if not input_file.exists():
            print(f"\n❌ File not found: {input_file}")
            continue
        
        output_file = dest_dir / f"{split_name}.tsv"
        
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()}")
        print(f"{'='*70}")
        
        total, valid, skipped = convert_tass_file(str(input_file), str(output_file))
        
        total_stats['total'] += total
        total_stats['valid'] += valid
        total_stats['skipped'] += skipped
        
        print(f"\n✅ {split_name.upper()} COMPLETE")
        print(f"  Total lines:  {total}")
        print(f"  Valid lines:  {valid}")
        print(f"  Skipped:      {skipped}")
        print(f"  Output:       {output_file}")
    
    # Summary
    print(f"\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Total lines processed:  {total_stats['total']}")
    print(f"Total valid lines:      {total_stats['valid']}")
    print(f"Total skipped:          {total_stats['skipped']}")
    print(f"Success rate:           {total_stats['valid']/total_stats['total']*100:.1f}%")
    
    # Verify outputs
    print(f"\n✅ FILES CREATED:")
    for split_name in datasets.keys():
        output_file = dest_dir / f"{split_name}.tsv"
        if output_file.exists():
            with open(output_file) as f:
                lines = len(f.readlines())
            print(f"  ✓ {output_file} ({lines} lines)")
        else:
            print(f"  ✗ {output_file} NOT FOUND")
    
    print(f"\n✅ CONVERSION COMPLETE")
    print(f"Ready for: python3 run_kbert_cls.py --train_path {dest_dir}/train.tsv ...")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
