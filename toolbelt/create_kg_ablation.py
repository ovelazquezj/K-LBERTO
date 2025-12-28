#!/usr/bin/env python3
"""
Create KG ablation files: 0, 50k, 500k triplets from WikidataES
"""

import random
from pathlib import Path

kg_source = "./brain/kgs/WikidataES_CLEAN_v251109.spo"
output_dir = Path("./brain/kgs/ablation")
output_dir.mkdir(exist_ok=True)

print("Loading full KG...")
with open(kg_source, 'r', encoding='utf-8') as f:
    triplets = f.readlines()

print(f"Total triplets: {len(triplets)}")

# Shuffle for random sampling
random.shuffle(triplets)

# Create ablation versions
ablation_sizes = [0, 50000, 500000]

for size in ablation_sizes:
    output_file = output_dir / f"WikidataES_{size}_triplets.spo"
    
    if size == 0:
        # Empty KG
        with open(output_file, 'w') as f:
            pass
        print(f"✓ {output_file} created (0 triplets)")
    else:
        # Sample size triplets
        sample = triplets[:min(size, len(triplets))]
        with open(output_file, 'w') as f:
            f.writelines(sample)
        print(f"✓ {output_file} created ({len(sample)} triplets)")

print("\n✓ Ablation KGs ready")
