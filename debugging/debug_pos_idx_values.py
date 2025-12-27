from brain import KnowledgeGraph

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

text = "[CLS] excelente"
split_sent = text.split()

# Reconstruir pos_idx_tree manualmente como lo hace knowgraph.py
pos_idx = -1
pos_idx_tree = []

for token in split_sent:
    entities = list(kg.lookup_table.get(token, []))[:30]
    
    # Word position
    token_pos_idx = [pos_idx + 1]  # word-level: 1 position per word
    
    # Entity positions
    entities_pos_idx = []
    for ent in entities:
        ent_pos_idx = [token_pos_idx[-1] + 1]  # Next position after token
        entities_pos_idx.append(ent_pos_idx)
    
    pos_idx_tree.append((token_pos_idx, entities_pos_idx))
    pos_idx = token_pos_idx[-1]
    
    print(f"Token '{token}':")
    print(f"  token_pos_idx: {token_pos_idx}")
    print(f"  entities_pos_idx: {entities_pos_idx}")

print("\n=== pos_idx_tree ===")
for i, (word_pos, ent_pos) in enumerate(pos_idx_tree):
    print(f"pos_idx_tree[{i}] = ({word_pos}, {ent_pos})")
    
print("\n=== Ahora simular agregar posiciones a pos ===")
pos = []
for i, (word, ents) in enumerate(split_sent):
    entities = list(kg.lookup_table.get(word, []))[:30]
    
    # Agregar word position
    pos += pos_idx_tree[i][0]
    print(f"Después de agregar word {i}: pos = {pos}")
    
    # Agregar entity positions
    for j in range(len(entities)):
        print(f"  Agregando entity {j}: pos_idx_tree[{i}][1][{j}] = {pos_idx_tree[i][1][j]}")
        pos += list(pos_idx_tree[i][1][j])
        print(f"  pos ahora = {pos}")
        
print(f"\npos final: {pos}")
