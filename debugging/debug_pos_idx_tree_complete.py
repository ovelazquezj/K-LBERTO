from brain import KnowledgeGraph
import brain.config as config

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

text = "[CLS] excelente"
split_sent = text.split()

print("=== PASO 1: CONSTRUCCIÓN pos_idx_tree ===")
print(f"split_sent: {split_sent}")

pos_idx = -1
pos_idx_tree = []

for i, token in enumerate(split_sent):
    entities = list(kg.lookup_table.get(token, []))[:30]
    
    token_pos_idx = [pos_idx + 1]
    
    entities_pos_idx = []
    for ent in entities:
        ent_pos_idx = [token_pos_idx[-1] + 1]
        entities_pos_idx.append(ent_pos_idx)
    
    pos_idx_tree.append((token_pos_idx, entities_pos_idx))
    
    print(f"\nToken {i}: '{token}'")
    print(f"  pos_idx before: {pos_idx}")
    print(f"  token_pos_idx: {token_pos_idx}")
    print(f"  entities_pos_idx: {entities_pos_idx}")
    
    pos_idx = token_pos_idx[-1]
    print(f"  pos_idx after: {pos_idx}")
    print(f"  pos_idx_tree[{i}]: {pos_idx_tree[i]}")

print("\n=== RESULTADO pos_idx_tree ===")
for i, (word_pos, ent_pos) in enumerate(pos_idx_tree):
    print(f"pos_idx_tree[{i}]: word={word_pos}, ents={ent_pos}")

print("\n=== PASO 2: AGREGAR A pos ===")
pos = []
for i in range(len(pos_idx_tree)):
    if i == 0:
        pos += pos_idx_tree[i][0]
        print(f"i={i} ([CLS]): pos += {pos_idx_tree[i][0]} → pos = {pos}")
    else:
        pos += pos_idx_tree[i][0]
        print(f"i={i}: pos += {pos_idx_tree[i][0]} → pos = {pos}")
        for j in range(len(pos_idx_tree[i][1])):
            pos += list(pos_idx_tree[i][1][j])
            print(f"  j={j}: pos += {pos_idx_tree[i][1][j]} → pos = {pos}")

print(f"\npos final: {pos}")
print(f"ESPERADO: [0, 1, 2]")
print(f"¿COINCIDE? {pos == [0, 1, 2]}")
