#!/usr/bin/env python3
"""
Prepare Train/Val/Test Splits & BETO Compatibility Check
CORRECTED: Uses ONLY Python standard library - NO numpy
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_splits_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BETOVocabularyChecker:
    COMMON_SPANISH_TOKENS = {
        'el', 'la', 'de', 'y', 'a', 'en', 'que', 'es', 'se',
        'por', 'con', 'para', 'una', 'un', 'no', 'su', 'al',
        'presidente', 'país', 'día', 'año', 'parte', 'tiempo', 'grupo',
    }
    
    @staticmethod
    def estimate_coverage(tokens: List[str]) -> Dict:
        coverage = {
            'total_tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'estimated_coverage_pct': 0,
        }
        
        estimated_known = 0
        for token in set(tokens):
            token_lower = token.lower()
            if token_lower in BETOVocabularyChecker.COMMON_SPANISH_TOKENS:
                estimated_known += 1
            elif len(token) >= 3 and token.isalpha():
                estimated_known += 1
            elif token.isdigit():
                estimated_known += 1
            elif not token.isascii():
                estimated_known += 1
        
        coverage['estimated_coverage_pct'] = (estimated_known / len(set(tokens)) * 100) if set(tokens) else 100
        return coverage

def prepare_training_splits(input_dir: Path, output_dir: Path) -> Dict:
    splits_metadata = {'splits': {}, 'total_samples': 0}
    
    formatted_dir = input_dir / 'conll2002_kbert_format'
    all_tokens = set()
    
    for split_name in ['train', 'validation', 'test']:
        split_dir = formatted_dir / split_name
        jsonl_file = split_dir / f"{split_name}_kbert.jsonl"
        
        if not jsonl_file.exists():
            logger.warning(f"⚠ Not found: {split_name}")
            continue
        
        logger.info(f"\n  Processing {split_name}...")
        
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        split_tokens = set()
        for sample in samples:
            split_tokens.update(sample.get('tokens', []))
            all_tokens.update(sample.get('tokens', []))
        
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = split_output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, sample in enumerate(samples):
                sample_with_id = {**sample, 'split_id': idx}
                f.write(json.dumps(sample_with_id, ensure_ascii=False) + '\n')
        
        vocab_check = BETOVocabularyChecker.estimate_coverage(list(split_tokens))
        
        splits_metadata['splits'][split_name] = {
            'num_samples': len(samples),
            'num_unique_tokens': len(split_tokens),
            'vocabulary_coverage': vocab_check
        }
        
        splits_metadata['total_samples'] += len(samples)
        
        logger.info(f"    ✓ {len(samples)} samples")
        logger.info(f"    ✓ {len(split_tokens)} unique tokens")
        logger.info(f"    ✓ BETO coverage: {vocab_check['estimated_coverage_pct']:.1f}%")
    
    global_vocab_check = BETOVocabularyChecker.estimate_coverage(list(all_tokens))
    splits_metadata['global_vocabulary_coverage'] = global_vocab_check
    
    logger.info(f"\n  Global:")
    logger.info(f"    ✓ Total tokens: {len(all_tokens)}")
    logger.info(f"    ✓ BETO coverage: {global_vocab_check['estimated_coverage_pct']:.1f}%")
    
    return splits_metadata

def generate_training_config(splits_metadata: Dict) -> Dict:
    total_samples = splits_metadata['total_samples']
    
    config = {
        'experiment': {
            'name': 'K-BERT-BETO-CoNLL2002-Spanish-Phase1',
            'description': 'Baseline K-BERT with BETO model on CoNLL 2002 Spanish NER',
        },
        'model': {
            'model_name': 'BETO',
            'model_path': '/path/to/beto_converted_to_uer',
            'model_type': 'bert',
            'use_knowledge_graph': False,
        },
        'dataset': {
            'dataset_name': 'CoNLL 2002 Spanish',
            'task': 'ner',
            'num_labels': 9,
            'total_samples': total_samples,
            'splits': splits_metadata['splits'],
        },
        'training_hyperparameters': {
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'max_seq_length': 512,
            'warmup_steps': int(total_samples / 16 * 3 * 0.1),
            'weight_decay': 0.01,
        },
        'hardware': {
            'device': 'cuda',
            'device_name': 'NVIDIA Jetson Orin NX',
        },
    }
    
    return config

def main():
    logger.info("=" * 80)
    logger.info("PHASE 1: Prepare Splits & BETO Compatibility")
    logger.info("=" * 80)
    
    input_dir = Path('./outputs')
    output_dir = Path('./outputs/conll2002_training_ready')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input: {input_dir.absolute()}")
    logger.info(f"Output: {output_dir.absolute()}")
    
    logger.info("\n[STEP 1] Preparing splits...")
    splits_metadata = prepare_training_splits(input_dir, output_dir)
    
    logger.info("\n[STEP 2] Generating config...")
    training_config = generate_training_config(splits_metadata)
    
    config_file = output_dir / 'training_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Config saved")
    
    logger.info("\n[STEP 3] Saving metadata...")
    metadata_file = output_dir / 'splits_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(splits_metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Metadata saved")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ STEP 3 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nTotal samples: {splits_metadata['total_samples']}")
    logger.info(f"BETO coverage: {splits_metadata['global_vocabulary_coverage']['estimated_coverage_pct']:.1f}%")
    logger.info(f"✓ Ready for K-BERT training\n")
    logger.info("=" * 80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        exit(1)
