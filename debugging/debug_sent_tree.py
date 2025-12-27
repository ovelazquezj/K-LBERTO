from brain import KnowledgeGraph
import sys

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

# Agregar print temporalmente en knowgraph.py
text = "[CLS] excelente"
split_sent = text.split()

# Simular sent_tree
sent_tree = []
for token in split_sent:
    entities = list(kg.lookup_table.get(token, []))[:30]  # max_entities
    sent_tree.append((token, entities))
    print(f"Token: '{token}' → entities: {entities}")

print("\nsent_tree:")
for i, (word, ents) in enumerate(sent_tree):
    print(f"  [{i}] word='{word}', entities={ents}, len(entities)={len(ents)}")

# Ahora verificar el loop
print("\nLoop execution:")
for i, (word, ents) in enumerate(sent_tree):
    print(f"Loop i={i}: len(sent_tree[{i}][1]) = {len(sent_tree[i][1])}")
    for j in range(len(sent_tree[i][1])):
        print(f"  j={j}: entity = '{sent_tree[i][1][j]}'")
