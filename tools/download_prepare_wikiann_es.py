#!/usr/bin/env python3
"""
Official WikiANN Spanish Dataset Downloader and Converter for K-BERT

Sources:
- WikiANN Official: https://huggingface.co/datasets/unimelb-nlp/wikiann
- Format: IOB2 with tags [O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC]
- Paper: Rahimi et al. (2019) "Massively Multilingual Transfer for NER"

Description:
This script downloads the official WikiANN Spanish dataset from Hugging Face,
converts it from Parquet format to TSV format compatible with K-BERT,
and generates train/dev/test splits with proper sentence separation.

Format specification:
- Input: Hugging Face dataset with fields {tokens, tags} in IOB2 format
- Output: TSV files with format: token\ttag (with blank lines for sentence separation)
- Tag mapping: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikiANNProcessor:
    """Process WikiANN dataset from Hugging Face for K-BERT training"""
    
    # IOB2 tag definitions (official WikiANN format)
    TAG_MAPPING = {
        'O': 0,
        'B-PER': 1,
        'I-PER': 2,
        'B-ORG': 3,
        'I-ORG': 4,
        'B-LOC': 5,
        'I-LOC': 6
    }
    
    REVERSE_TAG_MAPPING = {v: k for k, v in TAG_MAPPING.items()}
    
    # WikiANN uses numeric tags, map them to IOB2 labels
    ID_TO_TAG = {
        0: 'O',
        1: 'B-PER',
        2: 'I-PER',
        3: 'B-ORG',
        4: 'I-ORG',
        5: 'B-LOC',
        6: 'I-LOC'
    }
    
    def __init__(self, output_dir: str = "./pipeline-dataset/outputs/wikiann", 
                 language: str = "es", cache_dir: Optional[str] = None):
        """
        Initialize WikiANN processor
        
        Args:
            output_dir: Directory where TSV files will be saved
            language: Language code (default: "es" for Spanish)
            cache_dir: Cache directory for downloaded dataset (optional)
        """
        self.output_dir = Path(output_dir)
        self.language = language
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface" / "datasets"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Language: {self.language}")
    
    def download_dataset(self) -> Optional[Dict]:
        """
        Download WikiANN dataset from Hugging Face official source
        
        Returns:
            Dictionary with train/dev/test splits or None if download fails
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("ERROR: 'datasets' library not installed")
            logger.error("Install with: pip install datasets --break-system-packages")
            return None
        
        try:
            logger.info(f"Downloading WikiANN ({self.language}) from Hugging Face...")
            logger.info("Source: https://huggingface.co/datasets/unimelb-nlp/wikiann")
            
            # Official dataset from unimelb-nlp
            dataset = load_dataset("unimelb-nlp/wikiann", self.language, trust_remote_code=True)
            
            logger.info(f"✓ Dataset downloaded successfully")
            logger.info(f"  - Train samples: {len(dataset['train'])}")
            logger.info(f"  - Validation samples: {len(dataset['validation'])}")
            logger.info(f"  - Test samples: {len(dataset['test'])}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("1. Check internet connection")
            logger.error("2. Verify language code is valid (es = Spanish)")
            logger.error("3. Update datasets library: pip install --upgrade datasets --break-system-packages")
            return None
    
    def validate_tags(self, tags: List) -> bool:
        """Validate that tags are in correct IOB2 format"""
        for tag in tags:
            if tag not in self.TAG_MAPPING:
                logger.warning(f"Unknown tag: {tag}")
                return False
        return True
    
    def convert_to_tsv(self, dataset: Dict, split_name: str = "train") -> Tuple[str, int]:
        """
        Convert dataset split to TSV format compatible with K-BERT
        
        Format:
        token1\ttag1
        token2\ttag2
        
        [blank line for sentence separation]
        
        token3\ttag3
        ...
        
        Args:
            dataset: Dataset dictionary with splits
            split_name: Name of split (train, validation, test)
            
        Returns:
            Tuple of (output_file_path, number_of_sentences)
        """
        if split_name not in dataset:
            logger.error(f"Split '{split_name}' not found in dataset")
            return None, 0
        
        split_data = dataset[split_name]
        output_file = self.output_dir / f"{split_name}.tsv"
        
        logger.info(f"Converting {split_name} split ({len(split_data)} samples)...")
        
        num_sentences = 0
        num_tokens = 0
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample_idx, sample in enumerate(split_data):
                    tokens = sample['tokens']
                    # WikiANN uses 'ner_tags', not 'tags'
                    tag_ids = sample['ner_tags']
                    
                    # Validate sample
                    if len(tokens) != len(tag_ids):
                        logger.warning(f"Sample {sample_idx}: token/tag mismatch ({len(tokens)} vs {len(tag_ids)})")
                        continue
                    
                    # Convert tag IDs to labels (WikiANN provides numeric IDs, convert to IOB2 labels)
                    tag_labels = []
                    for tag_id in tag_ids:
                        if isinstance(tag_id, int):
                            label = self.ID_TO_TAG.get(tag_id, 'O')
                            tag_labels.append(label)
                        else:
                            tag_labels.append(str(tag_id))
                    
                    # Validate tags
                    if not self.validate_tags(tag_labels):
                        logger.warning(f"Sample {sample_idx}: invalid tags, skipping")
                        continue
                    
                    # Write tokens and tags
                    for token, tag in zip(tokens, tag_labels):
                        f.write(f"{token}\t{tag}\n")
                        num_tokens += 1
                    
                    # Write blank line to separate sentences
                    f.write("\n")
                    num_sentences += 1
                    
                    # Progress logging
                    if (sample_idx + 1) % 1000 == 0:
                        logger.info(f"  Processed {sample_idx + 1}/{len(split_data)} samples")
            
            logger.info(f"✓ {split_name}.tsv created successfully")
            logger.info(f"  - Path: {output_file}")
            logger.info(f"  - Sentences: {num_sentences}")
            logger.info(f"  - Tokens: {num_tokens}")
            
            return str(output_file), num_sentences
            
        except Exception as e:
            logger.error(f"Error converting {split_name}: {str(e)}")
            return None, 0
    
    def generate_statistics(self, dataset: Dict, output_file: str):
        """Generate dataset statistics"""
        stats = {
            'language': self.language,
            'dataset_name': 'WikiANN',
            'source': 'https://huggingface.co/datasets/unimelb-nlp/wikiann',
            'splits': {}
        }
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                split_data = dataset[split_name]
                
                total_tokens = 0
                total_sentences = len(split_data)
                tag_counts = defaultdict(int)
                
                for sample in split_data:
                    tokens = sample['tokens']
                    tag_ids = sample['ner_tags']  # WikiANN uses 'ner_tags'
                    
                    total_tokens += len(tokens)
                    
                    # Count tags
                    for tag_id in tag_ids:
                        if isinstance(tag_id, int):
                            tag_label = self.ID_TO_TAG.get(tag_id, 'O')
                        else:
                            tag_label = str(tag_id)
                        tag_counts[tag_label] += 1
                
                stats['splits'][split_name] = {
                    'num_sentences': total_sentences,
                    'num_tokens': total_tokens,
                    'avg_tokens_per_sentence': round(total_tokens / total_sentences, 2),
                    'tag_distribution': dict(tag_counts)
                }
        
        # Write statistics
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("DATASET STATISTICS")
        logger.info(f"{'='*60}")
        for split_name, split_stats in stats['splits'].items():
            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"  Sentences: {split_stats['num_sentences']}")
            logger.info(f"  Tokens: {split_stats['num_tokens']}")
            logger.info(f"  Avg tokens/sentence: {split_stats['avg_tokens_per_sentence']}")
            logger.info(f"  Tag distribution:")
            for tag, count in sorted(split_stats['tag_distribution'].items()):
                pct = (count / split_stats['num_tokens']) * 100
                logger.info(f"    {tag:8s}: {count:7d} ({pct:5.2f}%)")
    
    def process(self) -> bool:
        """
        Main processing pipeline
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*60)
        logger.info("WikiANN Spanish Dataset Preparation for K-BERT")
        logger.info("="*60)
        
        # Step 1: Download dataset
        dataset = self.download_dataset()
        if dataset is None:
            return False
        
        # Step 2: Convert to TSV
        logger.info("\n" + "-"*60)
        logger.info("CONVERTING TO TSV FORMAT")
        logger.info("-"*60 + "\n")
        
        for split_name in ['train', 'validation', 'test']:
            tsv_file, num_sentences = self.convert_to_tsv(dataset, split_name)
            if tsv_file is None:
                logger.error(f"Failed to convert {split_name} split")
                return False
        
        # Step 3: Generate statistics
        logger.info("\n" + "-"*60)
        logger.info("GENERATING STATISTICS")
        logger.info("-"*60 + "\n")
        
        stats_file = self.output_dir / "dataset_statistics.json"
        self.generate_statistics(dataset, str(stats_file))
        logger.info(f"✓ Statistics saved to: {stats_file}")
        
        # Step 4: Generate config file for K-BERT
        logger.info("\n" + "-"*60)
        logger.info("GENERATING K-BERT CONFIG")
        logger.info("-"*60 + "\n")
        
        config_template = f"""# K-BERT WikiANN Spanish Configuration
# Generated automatically by download_prepare_wikiann_es.py

data:
  train_path: {self.output_dir}/train.tsv
  dev_path: {self.output_dir}/validation.tsv
  test_path: {self.output_dir}/test.tsv

dataset_info:
  name: WikiANN Spanish
  language: Spanish (es)
  source: https://huggingface.co/datasets/unimelb-nlp/wikiann
  format: IOB2
  entities: [PER, ORG, LOC]
  tag_mapping:
    O: 0
    B-PER: 1
    I-PER: 2
    B-ORG: 3
    I-ORG: 4
    B-LOC: 5
    I-LOC: 6
"""
        
        config_file = self.output_dir / "kbert_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_template)
        logger.info(f"✓ K-BERT config saved to: {config_file}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("✓ PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"\nOutput files created in: {self.output_dir}")
        logger.info(f"  - train.tsv (training data)")
        logger.info(f"  - validation.tsv (development data)")
        logger.info(f"  - test.tsv (test data)")
        logger.info(f"  - dataset_statistics.json (statistics)")
        logger.info(f"  - kbert_config.yaml (K-BERT configuration)")
        
        logger.info("\nNext steps:")
        logger.info("1. Update your K-BERT config.yaml with the paths above")
        logger.info("2. Ensure WikidataES_CLEAN_v251109.spo is registered in brain/config.py")
        logger.info("3. Run: python train_kbert_ner_monitored.py --config config.yaml")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare WikiANN Spanish dataset for K-BERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration
  python download_prepare_wikiann_es.py
  
  # Custom output directory
  python download_prepare_wikiann_es.py --output /path/to/output
  
  # Different language (e.g., English)
  python download_prepare_wikiann_es.py --lang en
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./pipeline-dataset/outputs/wikiann',
        help='Output directory for TSV files (default: ./pipeline-dataset/outputs/wikiann)'
    )
    
    parser.add_argument(
        '--lang',
        default='es',
        help='Language code (default: es for Spanish)'
    )
    
    parser.add_argument(
        '--cache',
        default=None,
        help='Cache directory for downloaded dataset (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate language code
    if len(args.lang) != 2:
        logger.error(f"Invalid language code: {args.lang}")
        logger.error("Use 2-letter ISO 639-1 codes (e.g., 'es' for Spanish)")
        return 1
    
    # Create processor
    processor = WikiANNProcessor(
        output_dir=args.output,
        language=args.lang,
        cache_dir=args.cache
    )
    
    # Run processing
    success = processor.process()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
