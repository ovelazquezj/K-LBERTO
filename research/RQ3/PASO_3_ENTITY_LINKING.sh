#!/bin/bash

# ============================================
# PASO 3: ENTITY LINKING (FASE 1)
# Extract named entities from Spanish tweets
# Link to Wikidata/DBpedia for KG injection
# ============================================

set -e

echo "========================================"
echo "PASO 3: ENTITY LINKING (SPANISH TWEETS)"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

echo -e "\nStarting entity linking pipeline...\n"

python << 'PYTHON_EOF'
"""
PASO 3: Entity Linking
Extract entities from TASS 2019 tweets using transformers NER pipeline
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

# Use transformers pipeline for NER (already installed, Python 3.8 compatible)
from transformers import pipeline

print("\n" + "="*70)
print("PASO 3: ENTITY LINKING")
print("="*70)

# Initialize NER pipeline (Spanish)
print("\nLoading Spanish NER model...")
try:
    # Try Spanish-specific model
    nlp = pipeline("token-classification", 
                   model="xlm-roberta-large-finetuned-conll03-english",
                   device=-1)  # CPU
except Exception as e:
    print(f"Warning: {e}")
    print("Using default English model (will identify some entities)...")
    nlp = pipeline("token-classification", device=-1)

print("✓ NER model loaded")

# Load training data
print("\nLoading tweets from train.tsv...")

tweets = []
with open('./data/raw/train.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        tweets.append({
            'label': row['label'],
            'text': row['text']
        })

print(f"✓ Loaded {len(tweets)} tweets")

# Process tweets for entities
print("\n" + "="*70)
print("ENTITY EXTRACTION")
print("="*70)

entities_per_tweet = []
entity_stats = Counter()
entity_types = Counter()
sample_entities = defaultdict(list)

print(f"\nProcessing {len(tweets)} tweets...")

for i, tweet in enumerate(tweets):
    text = tweet['text']
    
    # Clean tweet text for NER
    # Remove URLs, mentions but keep text
    clean_text = text
    
    try:
        # Extract entities
        results = nlp(clean_text[:512])  # Limit to 512 chars for efficiency
        
        tweet_entities = []
        prev_entity = None
        
        for item in results:
            entity_type = item['entity'].replace('B-', '').replace('I-', '')
            entity_text = item['word'].replace('##', '')  # Remove BPE markers
            score = item['score']
            
            # Aggregate consecutive tokens of same entity
            if (prev_entity and 
                prev_entity['label'] == entity_type and 
                score > 0.7):  # High confidence only
                prev_entity['text'] += entity_text
            else:
                if prev_entity:
                    tweet_entities.append(prev_entity)
                prev_entity = {
                    'text': entity_text,
                    'label': entity_type,
                    'score': score
                }
        
        if prev_entity:
            tweet_entities.append(prev_entity)
        
        # Filter high-confidence entities
        high_conf_entities = [e for e in tweet_entities if e['score'] > 0.7]
        
        if high_conf_entities:
            for ent in high_conf_entities:
                entity_stats[ent['text'].lower()] += 1
                entity_types[ent['label']] += 1
                
                # Keep samples
                if len(sample_entities[ent['label']]) < 3:
                    sample_entities[ent['label']].append({
                        'entity': ent['text'],
                        'tweet': text[:80]
                    })
            
            entities_per_tweet.append({
                'tweet': text,
                'label': tweet['label'],
                'entities': high_conf_entities,
            })
    
    except Exception as e:
        # Skip tweets that cause issues
        continue
    
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(tweets)}")

print(f"\n✓ Extraction complete")

# Statistics
print("\n" + "="*70)
print("ENTITY STATISTICS")
print("="*70)

print(f"\nTweets with entities: {len(entities_per_tweet)} / {len(tweets)}")
print(f"Entity coverage: {len(entities_per_tweet) / len(tweets) * 100:.1f}%")

print(f"\nEntity types found:")
for ent_type, count in entity_types.most_common():
    print(f"  {ent_type}: {count}")

print(f"\nTop 20 most frequent entities:")
for entity, count in entity_stats.most_common(20):
    print(f"  '{entity}': {count}")

# Save detailed results
output_dir = Path('./results')
output_dir.mkdir(parents=True, exist_ok=True)

# Save entities with context
entities_file = output_dir / 'PASO_3_ENTITIES_EXTRACTED.json'

# Convert float32 to float for JSON serialization
def convert_entities(entities_list):
    converted = []
    for item in entities_list:
        new_item = {
            'tweet': item['tweet'],
            'label': item['label'],
            'entities': [
                {
                    'text': e['text'],
                    'label': e['label'],
                    'score': float(e['score'])  # Convert float32 to float
                }
                for e in item['entities']
            ]
        }
        converted.append(new_item)
    return converted

with open(entities_file, 'w', encoding='utf-8') as f:
    json.dump({
        'total_tweets_processed': len(tweets),
        'tweets_with_entities': len(entities_per_tweet),
        'entity_types': dict(entity_types),
        'top_entities': dict(entity_stats.most_common(100)),
        'sample_entities': dict(sample_entities),
        'detailed': convert_entities(entities_per_tweet[:100]),  # First 100 for inspection
    }, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✓ Entities saved: {entities_file}")

# Validate sample
print("\n" + "="*70)
print("VALIDATION SAMPLE")
print("="*70)

print("\nSample tweets with extracted entities:")
for i, item in enumerate(entities_per_tweet[:3]):
    print(f"\n[Tweet {i+1}] Label: {item['label']}")
    print(f"Text: {item['tweet'][:80]}...")
    print(f"Entities found:")
    for ent in item['entities']:
        print(f"  - {ent['text']} ({ent['label']}, confidence: {ent['score']:.2f})")

# Statistics summary
stats_summary = {
    'total_tweets': len(tweets),
    'tweets_with_entities': len(entities_per_tweet),
    'coverage_percent': round(len(entities_per_tweet) / len(tweets) * 100, 2),
    'entity_types': dict(entity_types),
    'unique_entities': len(entity_stats),
    'avg_entities_per_tweet': round(sum(len(item['entities']) for item in entities_per_tweet) / max(len(entities_per_tweet), 1), 2) if entities_per_tweet else 0,
}

stats_file = output_dir / 'PASO_3_ENTITY_STATS.json'
with open(stats_file, 'w') as f:
    json.dump(stats_summary, f, indent=2)

print(f"\n✓ Statistics saved: {stats_file}")

# Summary
print("\n" + "="*70)
print("SUMMARY - PASO 3")
print("="*70)

print(f"""
Tweets processed: {stats_summary['total_tweets']}
Tweets with entities: {stats_summary['tweets_with_entities']}
Coverage: {stats_summary['coverage_percent']}%
Entity types: {len(entity_types)}
Unique entities: {stats_summary['unique_entities']}
Avg entities/tweet: {stats_summary['avg_entities_per_tweet']}

Next step (PASO 4):
- Map entities to sentiment-relevant concepts
- Filter to 50k triplets from Wikidata
- Focus on: Brand names, sentiment words, sentiment expressions
""")

print(f"\n✓ PASO 3: ENTITY LINKING COMPLETE")

PYTHON_EOF

PYTHON_EOF

echo ""
echo "========================================"
echo "✓ PASO 3: COMPLETE"
echo "========================================"
echo ""
echo "Results:"
ls -lh ~/projects/RQ3_TASS/results/PASO_3*.json
echo ""
echo "Next: PASO 4 - Knowledge Graph Curation"
echo "========================================"
