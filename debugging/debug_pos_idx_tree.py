from brain import KnowledgeGraph

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

# Inyectar código de debug en knowgraph.py
text = "[CLS] excelente"

# Simulamos manualmente lo que hace knowgraph
split_sent = text.split()
print("Split sent:", split_sent)

for i, token in enumerate(split_sent):
    entities = list(kg.lookup_table.get(token, []))
    print(f"\nToken {i}: '{token}'")
    print(f"  Entities found: {entities}")
    print(f"  Number of entities: {len(entities)}")
    if entities:
        for j, ent in enumerate(entities):
            print(f"    Entity {j}: '{ent}'")

# Ahora ejecutar add_knowledge_with_vm y ver resultado
tokens, pos, vm, seg = kg.add_knowledge_with_vm([text], add_pad=True, max_length=128)
print("\n=== RESULTADO ===")
print("Tokens:", tokens[0][:10])
print("Positions:", pos[0][:10])
print("Segment:", seg[0][:10])
print("\nLongitud know_sent:", len([t for t in tokens[0] if t != '[PAD]']))
print("Longitud pos:", len([p for p in pos[0] if p != 127]))
