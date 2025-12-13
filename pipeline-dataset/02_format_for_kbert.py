#!/usr/bin/env python3
"""
Format Dataset for K-BERT Training
CORRECTED: Uses ONLY Python standard library - NO numpy
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from statistics import mean

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_format_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KBERTFormat:
    TAG_TO_ID = {
        'O': 0, 'B-PER': 1, 'I-PER': 2,
        'B-ORG': 3, 'I-ORG': 4,
        'B-LOC': 5, 'I-LOC': 6,
        'B-MISC': 7, 'I-MISC': 8,
    }
    ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}
    NUM_LABELS = len(TAG_TO_ID)

def convert_conll_to_kbert_format(conll_sample: Dict) -> Dict:
    tokens = conll_sample.get('tokens', [])
    tag_ids = conll_sample.get('ner_tags_ids', [])
    tag_strings = conll_sample.get('ner_tags', [])
    text = ' '.join(tokens)
    
    return {
        'text': text,
        'tokens': tokens,
        'ner_tags': tag_strings,
        'ner_tags_ids': tag_ids,
        'sample_id': conll_sample.get('id', ''),
        'length': len(tokens)
    }

def validate_kbert_format(sample: Dict) -> Tuple[bool, str]:
    required_fields = ['tokens', 'ner_tags', 'ner_tags_ids']
    for field in required_fields:
        if field not in sample:
            return False, f"Missing: {field}"
    
    if len(sample['tokens']) != len(sample['ner_tags']) or len(sample['tokens']) != len(sample['ner_tags_ids']):
        return False, "Length mismatch"
    
    valid_tags = set(KBERTFormat.TAG_TO_ID.keys())
    if not set(sample['ner_tags']).issubset(valid_tags):
        return False, "Invalid tags"
    
    return True, ""

def compute_format_statistics(samples: list) -> dict:
    stats = {
        'total_samples': len(samples),
        'valid_samples': 0,
        'invalid_samples': 0,
        'length_distribution': {'min': float('inf'), 'max': 0, 'mean': 0},
        'entity_statistics': {'total_entities': 0, 'entities_per_sample': 0, 'entity_type_distribution': {}}
    }
    
    lengths = []
    entities_per_sample_list = []
    
    for sample in samples:
        is_valid, _ = validate_kbert_format(sample)
        
        if is_valid:
            stats['valid_samples'] += 1
            tokens = sample['tokens']
            tags = sample['ner_tags']
            length = len(tokens)
            lengths.append(length)
            
            stats['length_distribution']['min'] = min(stats['length_distribution']['min'], length)
            stats['length_distribution']['max'] = max(stats['length_distribution']['max'], length)
            
            entity_count = sum(1 for tag in tags if tag != 'O')
            entities_per_sample_list.append(entity_count)
            stats['entity_statistics']['total_entities'] += entity_count
            
            for tag in tags:
                if tag != 'O':
                    entity_type = tag.split('-')[1]
                    stats['entity_statistics']['entity_type_distribution'][entity_type] = \
                        stats['entity_statistics']['entity_type_distribution'].get(entity_type, 0) + 1
        else:
            stats['invalid_samples'] += 1
    
    if lengths:
        stats['length_distribution']['mean'] = mean(lengths)
    if entities_per_sample_list:
        stats['entity_statistics']['entities_per_sample'] = mean(entities_per_sample_list)
    
    return stats

def main():
    logger.info("=" * 80)
    logger.info("PHASE 1: Format Dataset for K-BERT")
    logger.info("=" * 80)
    
    input_dir = Path('./outputs/conll2002_spanish')
    output_dir = Path('./outputs/conll2002_kbert_format')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input: {input_dir.absolute()}")
    logger.info(f"Output: {output_dir.absolute()}")
    
    all_stats = {}
    
    for split_name in ['train', 'validation', 'test']:
        split_dir = input_dir / split_name
        jsonl_file = split_dir / f"{split_name}.jsonl"
        
        if not jsonl_file.exists():
            logger.warning(f"⚠ Not found: {split_name}")
            continue
        
        logger.info(f"\n  Processing {split_name}...")
        
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"    Loaded: {len(samples)}")
        logger.info(f"    Converting...")
        
        converted_samples = [convert_conll_to_kbert_format(s) for s in samples]
        logger.info(f"    ✓ Converted {len(converted_samples)}")
        
        logger.info(f"    Validating...")
        stats = compute_format_statistics(converted_samples)
        logger.info(f"    ✓ Valid: {stats['valid_samples']}/{stats['total_samples']}")
        logger.info(f"    Length: min={stats['length_distribution']['min']}, max={stats['length_distribution']['max']}, mean={stats['length_distribution']['mean']:.2f}")
        
        logger.info(f"    Saving...")
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(exist_ok=True)
        
        jsonl_output = split_output_dir / f"{split_name}_kbert.jsonl"
        with open(jsonl_output, 'w', encoding='utf-8') as f:
            for sample in converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"    ✓ {split_name}")
        
        stats_output = split_output_dir / f"{split_name}_stats.json"
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        all_stats[split_name] = stats
    
    logger.info("\n[STEP 6] K-BERT Configuration...")
    kbert_config = {
        'task': 'ner',
        'model': 'BETO',
        'num_labels': KBERTFormat.NUM_LABELS,
        'label_to_id': KBERTFormat.TAG_TO_ID,
        'id_to_label': KBERTFormat.ID_TO_TAG,
        'statistics': all_stats
    }
    
    config_output = output_dir / 'kbert_config.json'
    with open(config_output, 'w', encoding='utf-8') as f:
        json.dump(kbert_config, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Configuration saved")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ STEP 2 COMPLETE")
    logger.info("=" * 80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        exit(1)
