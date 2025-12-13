#!/usr/bin/env python3
"""
Convert WikiANN to format expected by run_kbert_ner_spanish.py

Expected format:
token1 token2 token3<TAB>B-PER O B-LOC
token4 token5<TAB>O O

(All tokens in a sentence on one line, separated by spaces)
(All tags in a sentence on one line, separated by spaces)
"""

from pathlib import Path
from datasets import load_dataset

ID_TO_TAG = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC'
}

def convert_split(split_name, output_dir):
    """Convert a single split"""
    print(f"Processing {split_name}...")
    
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
            
            # Format: all tokens on one line (space-separated)
            #         TAB
            #         all tags on one line (space-separated)
            tokens_str = ' '.join(tokens)
            tags_str = ' '.join(tag_labels)
            
            f.write(f"{tokens_str}\t{tags_str}\n")
            
            num_sentences += 1
            num_tokens += len(tokens)
            
            if (sample_idx + 1) % 1000 == 0:
                print(f"  Processed {sample_idx + 1}/{len(split_data)} samples")
    
    print(f"✓ {split_name}: {num_sentences} sentences, {num_tokens} tokens")
    return num_sentences, num_tokens

def main():
    output_dir = Path("./pipeline-dataset/outputs/wikiann_kbert_format")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("WikiANN to run_kbert_ner_spanish.py Format")
    print("="*60)
    
    stats = {}
    
    for split_name in ['train', 'validation', 'test']:
        num_sentences, num_tokens = convert_split(split_name, output_dir)
        stats[split_name] = {'sentences': num_sentences, 'tokens': num_tokens}
    
    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    
    for split_name, data in stats.items():
        print(f"  {split_name}: {data['sentences']} sentences, {data['tokens']} tokens")
    
    # Show example
    print("\nExample format:")
    with open(output_dir / "train.tsv", 'r') as f:
        for i in range(3):
            line = f.readline()
            if line:
                parts = line.strip().split('\t')
                print(f"  Tokens: {parts[0][:50]}...")
                print(f"  Tags:   {parts[1][:50]}...")
                print()
    
    print("Next step: Train with:")
    print("python3 run_kbert_ner_spanish.py \\")
    print("  --pretrained_model_path ./models/beto_uer_model/pytorch_model.bin \\")
    print("  --config_path ./models/beto_uer_model/config.json \\")
    print("  --vocab_path ./models/beto_uer_model/vocab.txt \\")
    print(f"  --train_path {output_dir}/train.tsv \\")
    print(f"  --dev_path {output_dir}/validation.tsv \\")
    print(f"  --test_path {output_dir}/test.tsv \\")
    print("  --epochs_num 8 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 2e-05 \\")
    print("  --kg_name WikidataES_CLEAN_v251109 \\")
    print("  --output_model_path ./outputs/kbert_ner/kbert_beto_ner_wikiann.bin \\")
    print("  --seq_length 128")

if __name__ == '__main__':
    main()
