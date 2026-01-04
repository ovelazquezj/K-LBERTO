#!/usr/bin/env python3
"""Ver qué labels predice el modelo en dev set"""

import torch
import sys
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from brain import KnowledgeGraph
import numpy as np

# Cargar labels_map
labels_map = {"[PAD]": 0, "[ENT]": 1, 'O': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'B-LOC': 6, 'I-LOC': 7, 'I-ORG': 8}
id2label = {v: k for k, v in labels_map.items()}

# Cargar vocab
vocab = Vocab()
vocab.load("models/beto_model/vocab.txt")

# Cargar KG
kg = KnowledgeGraph(spo_files=["brain/kgs/wikidata_ner_spanish.spo"], predicate=False)

# Leer primeras 10 líneas de dev
print("Simulando predicciones del modelo...")
print("\nDistribución de labels en primeras 100 líneas de DEV:")

all_labels = []
with open("data/wikiann_tsv/dev.tsv", 'r') as f:
    f.readline()  # Skip header
    for i, line in enumerate(f):
        if i >= 100:
            break
        labels_str, tokens_str = line.strip().split("\t")
        labels = [labels_map[l] for l in labels_str.split()]
        all_labels.extend(labels)

# Contar
from collections import Counter
label_counts = Counter(all_labels)

print("\nDistribución gold en dev (primeras 100 líneas):")
for label_id, count in sorted(label_counts.items()):
    label_name = id2label[label_id]
    print(f"  {label_name:10s} (ID={label_id}): {count:4d}")

print("\n" + "="*50)
print("Si el modelo NUNCA predice ID=3 (B-PER),")
print("entonces stats_per_type['PER']['tp'] = 0")
print("y por lo tanto P=R=F1=0.0")
print("="*50)
