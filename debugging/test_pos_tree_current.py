from brain import KnowledgeGraph

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

text = "[CLS] excelente"
tokens, pos, vm, seg = kg.add_knowledge_with_vm([text], add_pad=True, max_length=128)

print("RESULTADO:")
print(f"Tokens: {tokens[0][:5]}")
print(f"Positions: {pos[0][:5]}")
print(f"\nESPERADO: [0, 1, 2]")
print(f"ACTUAL: {pos[0][:3]}")
