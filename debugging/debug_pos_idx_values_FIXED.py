from brain import KnowledgeGraph

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

text = "[CLS] excelente"
split_sent = text.split()

# Reconstruir pos_idx_tree manualmente
pos_idx = -1
pos_idx_tree = []

for token in split_sent:
    entities = list(kg.lookup_table.get(token, []))[:30]
    
    # Word position
    token_pos_idx = [pos_idx + 1]
    
    # Entity positions
    entities_pos_idx = []
    for ent in entities:
        ent_pos_idx = [token_pos_idx[-1] + 1]
        entities_pos_idx.append(ent_pos_idx)
    
    pos_idx_tree.append((token_pos_idx, entities_pos_idx))
    pos_idx = token_pos_idx[-1]
    
    print(f"Token '{token}':")
    print(f"  token_pos_idx: {token_pos_idx}")
    print(f"  entities_pos_idx: {entities_pos_idx}")

print("\n=== pos_idx_tree ===")
for i, (word_pos, ent_pos) in enumerate(pos_idx_tree):
    print(f"pos_idx_tree[{i}] = ({word_pos}, {ent_pos})")
    
print("\n=== Simular agregar a pos ===")
pos = []
for i in range(len(split_sent)):
    word = split_sent[i]
    entities = list(kg.lookup_table.get(word, []))[:30]
    
    # Agregar word position
    pos += pos_idx_tree[i][0]
    print(f"i={i} word='{word}': pos += {pos_idx_tree[i][0]} → pos = {pos}")
    
    # Agregar entity positions
    for j in range(len(entities)):
        to_add = list(pos_idx_tree[i][1][j])
        pos += to_add
        print(f"  j={j} entity='{entities[j]}': pos += {to_add} → pos = {pos}")
        
print(f"\npos final: {pos}")
print(f"Esperado para 3 tokens: [0, 1, 2]")
