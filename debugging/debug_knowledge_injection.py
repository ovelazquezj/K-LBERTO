from brain import KnowledgeGraph

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

# Test sin knowledge
text_simple = "[CLS] excelente"
tokens, pos, vm, seg = kg.add_knowledge_with_vm([text_simple], add_pad=True, max_length=128)

print("Simple text:", text_simple)
print("Tokens:", tokens[0][:15])
print("Token count (no padding):", sum(1 for t in tokens[0] if t != '[PAD]'))
print("\nExpected: [CLS], excelente, positivo (si knowledge inyectado)")
print("Actual tokens:", tokens[0][:10])
