#!/bin/bash

# ============================================
# PASO 4: KNOWLEDGE GRAPH CURATION
# Create sentiment-relevant triplets for K-BERT injection
# ============================================

set -e

echo "========================================"
echo "PASO 4: KNOWLEDGE GRAPH CURATION"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo_kbert_jetson

cd ~/projects/RQ3_TASS

echo -e "\nCurating knowledge graph for Spanish sentiment analysis...\n"

python << 'PYTHON_EOF'
"""
PASO 4: Knowledge Graph Curation
Create triplets for K-BERT injection
Focus on: Sentiment-relevant entities and relations
"""

import json
from pathlib import Path
from collections import defaultdict

print("\n" + "="*70)
print("PASO 4: KNOWLEDGE GRAPH CURATION")
print("="*70)

# Load extracted entities
print("\nLoading extracted entities from PASO 3...")

entities_file = Path('./results/PASO_3_ENTITIES_EXTRACTED.json')
with open(entities_file, 'r', encoding='utf-8') as f:
    entities_data = json.load(f)

top_entities = entities_data['top_entities']
entity_types = entities_data['entity_types']

print(f"✓ Loaded {len(top_entities)} entities")

# Define sentiment-relevant triplets
print("\n" + "="*70)
print("DEFINING SENTIMENT TRIPLETS")
print("="*70)

# Strategy: Create triplets linking entities to sentiment/context
# Format: (entity, relation, concept)

triplets = []

# 1. Entity relations to sentiment indicators
sentiment_relations = {
    # Brands/Companies
    'mercadona': [
        ('mercadona', 'is_brand', 'retail_company'),
        ('mercadona', 'associated_with', 'shopping'),
        ('mercadona', 'sentiment_context', 'commercial'),
    ],
    'barcelona': [
        ('barcelona', 'is_location', 'city'),
        ('barcelona', 'associated_with', 'sports'),
        ('barcelona', 'sentiment_context', 'sports_team'),
    ],
    'real': [
        ('real', 'is_brand', 'football_team'),
        ('real', 'associated_with', 'sports'),
        ('real', 'sentiment_context', 'competitive'),
    ],
    'europa': [
        ('europa', 'is_location', 'continent'),
        ('europa', 'associated_with', 'travel'),
        ('europa', 'sentiment_context', 'geographic'),
    ],
}

# 2. Generic sentiment-relevant triplets for Spanish
generic_triplets = [
    # Positive sentiment indicators
    ('excelente', 'is_adjective', 'positive_sentiment'),
    ('excelente', 'sentiment_polarity', 'positive'),
    ('mejor', 'is_adjective', 'positive_sentiment'),
    ('mejor', 'sentiment_polarity', 'positive'),
    ('fantástico', 'is_adjective', 'positive_sentiment'),
    ('fantástico', 'sentiment_polarity', 'positive'),
    ('increíble', 'is_adjective', 'positive_sentiment'),
    ('increíble', 'sentiment_polarity', 'positive'),
    ('espectacular', 'is_adjective', 'positive_sentiment'),
    ('espectacular', 'sentiment_polarity', 'positive'),
    
    # Negative sentiment indicators
    ('malo', 'is_adjective', 'negative_sentiment'),
    ('malo', 'sentiment_polarity', 'negative'),
    ('horrible', 'is_adjective', 'negative_sentiment'),
    ('horrible', 'sentiment_polarity', 'negative'),
    ('terrible', 'is_adjective', 'negative_sentiment'),
    ('terrible', 'sentiment_polarity', 'negative'),
    ('pésimo', 'is_adjective', 'negative_sentiment'),
    ('pésimo', 'sentiment_polarity', 'negative'),
    ('dreadful', 'is_adjective', 'negative_sentiment'),
    ('dreadful', 'sentiment_polarity', 'negative'),
    
    # Neutral/Opinion indicators
    ('neutral', 'is_concept', 'neutral_sentiment'),
    ('normal', 'is_adjective', 'neutral_sentiment'),
    ('regular', 'is_adjective', 'neutral_sentiment'),
    ('mediocre', 'is_adjective', 'neutral_sentiment'),
    
    # Action verbs
    ('recomendar', 'is_verb', 'positive_action'),
    ('recomendar', 'sentiment_association', 'approval'),
    ('criticar', 'is_verb', 'negative_action'),
    ('criticar', 'sentiment_association', 'disapproval'),
    ('amar', 'is_verb', 'positive_action'),
    ('amar', 'sentiment_association', 'strong_positive'),
    ('odiar', 'is_verb', 'negative_action'),
    ('odiar', 'sentiment_association', 'strong_negative'),
    
    # Brands/Companies (generic)
    ('empresa', 'is_concept', 'business_entity'),
    ('marca', 'is_concept', 'brand_concept'),
    ('producto', 'is_concept', 'product_entity'),
    ('servicio', 'is_concept', 'service_entity'),
]

# 3. Add entity-specific triplets
print("\nCurating entity-specific triplets...")
for entity in list(top_entities.keys())[:50]:  # Top 50 entities
    entity_lower = entity.lower().strip()
    
    # Skip BPE markers and very short tokens
    if entity_lower.startswith('▁') or len(entity_lower) < 2:
        continue
    
    # Add basic triplets for extracted entities
    triplets.append((entity, 'is_entity', 'named_entity'))
    triplets.append((entity, 'found_in_tweets', 'tweet_corpus'))
    
    # Add specific triplets if in sentiment_relations
    if entity_lower in sentiment_relations:
        triplets.extend(sentiment_relations[entity_lower])

# Combine all triplets
all_triplets = generic_triplets + triplets

# Remove duplicates while preserving order
seen = set()
unique_triplets = []
for t in all_triplets:
    t_tuple = tuple(t)
    if t_tuple not in seen:
        seen.add(t_tuple)
        unique_triplets.append(t)

print(f"✓ Curated {len(unique_triplets)} unique triplets")

# Statistics
print("\n" + "="*70)
print("KNOWLEDGE GRAPH STATISTICS")
print("="*70)

relations = defaultdict(int)
subjects = defaultdict(int)
objects_set = defaultdict(int)

for s, r, o in unique_triplets:
    relations[r] += 1
    subjects[s] += 1
    objects_set[o] += 1

print(f"\nTotal triplets: {len(unique_triplets)}")
print(f"Unique relations: {len(relations)}")
print(f"Unique subjects: {len(subjects)}")
print(f"Unique objects: {len(objects_set)}")

print(f"\nTop 10 relations:")
for rel, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {rel}: {count}")

# Save triplets
output_dir = Path('./results')
output_dir.mkdir(parents=True, exist_ok=True)

triplets_file = output_dir / 'PASO_4_KNOWLEDGE_GRAPH_50K.json'

kg_data = {
    'metadata': {
        'source': 'Spanish TASS 2019 sentiment entities + generic sentiment relations',
        'total_triplets': len(unique_triplets),
        'total_relations': len(relations),
        'total_subjects': len(subjects),
        'total_objects': len(objects_set),
        'strategy': 'Sentiment-focused curation for Spanish NLP',
    },
    'triplets': [
        {
            'subject': s,
            'relation': r,
            'object': o,
        }
        for s, r, o in unique_triplets
    ],
    'relation_types': dict(relations),
}

with open(triplets_file, 'w', encoding='utf-8') as f:
    json.dump(kg_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Knowledge graph saved: {triplets_file}")
print(f"  File size: {triplets_file.stat().st_size / 1024:.1f} KB")

# Create summary
summary_file = output_dir / 'PASO_4_KG_SUMMARY.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("KNOWLEDGE GRAPH CURATION SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("Source: Spanish TASS 2019 + Sentiment taxonomy\n")
    f.write("Task: Provide sentiment-relevant knowledge for K-BERT injection\n\n")
    
    f.write("STATISTICS:\n")
    f.write(f"  Total triplets: {len(unique_triplets)}\n")
    f.write(f"  Unique relations: {len(relations)}\n")
    f.write(f"  Unique subjects: {len(subjects)}\n")
    f.write(f"  Unique objects: {len(objects_set)}\n\n")
    
    f.write("STRATEGY:\n")
    f.write("  1. Entity extraction from tweets (PASO 3)\n")
    f.write("  2. Sentiment-relevant relations (is_brand, is_adjective, sentiment_polarity)\n")
    f.write("  3. Generic sentiment triplets (excellent→positive, bad→negative)\n")
    f.write("  4. Curated for Spanish domain (brands, expressions)\n\n")
    
    f.write("NEXT STEP (PASO 5):\n")
    f.write("  - Use this KG as knowledge for K-BERT injection\n")
    f.write("  - Implement word-level visible matrix\n")
    f.write("  - Fine-tune K-BERT with KG on TASS training set\n")

print(f"✓ Summary saved: {summary_file}")

# Validation samples
print("\n" + "="*70)
print("VALIDATION SAMPLES")
print("="*70)

print(f"\nSample triplets:")
for i, (s, r, o) in enumerate(unique_triplets[:10]):
    print(f"  ({s}, {r}, {o})")

print(f"\n" + "="*70)
print("✓ PASO 4: KNOWLEDGE GRAPH CURATION COMPLETE")
print("="*70)

PYTHON_EOF

echo ""
echo "========================================"
echo "✓ PASO 4: COMPLETE"
echo "========================================"
echo ""
echo "Knowledge Graph created:"
ls -lh ~/projects/RQ3_TASS/results/PASO_4*.json
echo ""
echo "Next: PASO 5 - K-BERT Implementation"
echo "========================================"
