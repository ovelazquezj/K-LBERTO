#!/usr/bin/env python3
"""
CORRECTED: Convert WikiANN to K-BERT format with proper sentence separation
"""

import sys
from pathlib import Path
from datasets import load_dataset

# Tag mapping
ID_TO_TAG = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC'
}

def convert_split(split_name, input_dir, output_dir):
    """Convert a single split"""
    print(f"\nProcessing {split_name}...")
    
    # Load dataset
    dataset = load_dataset("unimelb-nlp/wikiann", "es", trust_remote_code=True)
    split_data = dataset[split_name]
    
    output_file = Path(output_dir) / f"{split_name}.tsv"
    
    num_sentences = 0
    num_tokens = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample_idx, sample in enumerate(split_data):
            tokens = sample['tokens']
            tag_ids = sample['ner_tags']
            
            # Convert tag IDs to labels
            tag_labels = [ID_TO_TAG.get(tag_id, 'O') for tag_id in tag_ids]
            
            # Write each token-tag pair
            for token, tag in zip(tokens, tag_labels):
                f.write(f"{token}\t{tag}\n")
                num_tokens += 1
            
            # Write blank line to separate sentences
            f.write("\n")
            num_sentences += 1
            
            if (sample_idx + 1) % 1000 == 0:
                print(f"  Processed {sample_idx + 1}/{len(split_data)} samples")
    
    print(f"✓ {split_name}: {num_sentences} sentences, {num_tokens} tokens")
    return num_sentences, num_tokens

def main():
    output_dir = Path("./pipeline-dataset/outputs/wikiann_kbert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("WikiANN to K-BERT Format Converter (CORRECTED)")
    print("="*60)
    
    stats = {}
    
    for split_name in ['train', 'validation', 'test']:
        num_sentences, num_tokens = convert_split(
            split_name,
            "./pipeline-dataset/outputs/wikiann",
            output_dir
        )
        stats[split_name] = {'sentences': num_sentences, 'tokens': num_tokens}
    
    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    
    for split_name, data in stats.items():
        print(f"  {split_name}: {data['sentences']} sentences, {data['tokens']} tokens")
    
    # Verify blank lines exist
    print("\nVerifying blank lines...")
    for split_name in ['train', 'validation', 'test']:
        file_path = output_dir / f"{split_name}.tsv"
        with open(file_path, 'r') as f:
            blank_lines = sum(1 for line in f if line.strip() == '')
        print(f"  {split_name}.tsv: {blank_lines} blank lines ✓")
    
    print("\nNext step: Train with:")
    print("python3 run_kbert_ner_spanish.py \\")
    print("  --pretrained_model_path ./models/beto_uer_model/pytorch_model.bin \\")
    print("  --config_path ./models/beto_uer_model/config.json \\")
    print("  --vocab_path ./models/beto_uer_model/vocab.txt \\")
    print("  --train_path ./pipeline-dataset/outputs/wikiann_kbert/train.tsv \\")
    print("  --dev_path ./pipeline-dataset/outputs/wikiann_kbert/validation.tsv \\")
    print("  --test_path ./pipeline-dataset/outputs/wikiann_kbert/test.tsv \\")
    print("  --epochs_num 8 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 2e-05 \\")
    print("  --kg_name WikidataES_CLEAN_v251109 \\")
    print("  --output_model_path ./outputs/kbert_ner/kbert_beto_ner_wikiann.bin \\")
    print("  --seq_length 128")

if __name__ == '__main__':
    main()
