#!/usr/bin/env python3
"""
Convert CoNLL 2002 JSONL format to TSV format for K-BERT NER
Input: JSONL files with tokens and ner_tags
Output: TSV files with space-separated tokens and labels
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('convert_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_jsonl_to_tsv(input_path, output_path):
    """Convert JSONL to TSV format for K-BERT NER"""
    
    logger.info(f"Converting {input_path} to {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Write header
            f_out.write("tokens\tlabels\n")
            
            line_count = 0
            for line in f_in:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                tokens = data.get('tokens', [])
                ner_tags = data.get('ner_tags', [])
                
                # Format: space-separated tokens and labels
                tokens_str = ' '.join(tokens)
                labels_str = ' '.join(ner_tags)
                
                f_out.write(f"{tokens_str}\t{labels_str}\n")
                line_count += 1
            
            logger.info(f"  ✓ Converted {line_count} samples")
    
    return line_count

def main():
    logger.info("=" * 80)
    logger.info("Convert CoNLL 2002 JSONL to TSV for K-BERT NER")
    logger.info("=" * 80)
    
    # Define paths
    base_dir = Path('./outputs/conll2002_spanish')
    output_dir = Path('./outputs/conll2002_tsv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': base_dir / 'train' / 'train.jsonl',
        'validation': base_dir / 'validation' / 'validation.jsonl',
        'test': base_dir / 'test' / 'test.jsonl'
    }
    
    stats = {}
    
    for split_name, input_file in splits.items():
        if not input_file.exists():
            logger.warning(f"⚠ Input file not found: {input_file}")
            continue
        
        logger.info(f"\n[Processing {split_name}]")
        
        output_file = output_dir / f"{split_name}.tsv"
        count = convert_jsonl_to_tsv(input_file, output_file)
        stats[split_name] = count
        logger.info(f"  ✓ Output: {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nStatistics:")
    for split_name, count in stats.items():
        logger.info(f"  {split_name}: {count} samples")
    
    logger.info(f"\nOutput directory: {output_dir.absolute()}")
    logger.info(f"\nFiles ready for K-BERT NER training:")
    logger.info(f"  - {output_dir}/train.tsv")
    logger.info(f"  - {output_dir}/validation.tsv")
    logger.info(f"  - {output_dir}/test.tsv")
    logger.info("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
