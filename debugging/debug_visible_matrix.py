from brain import KnowledgeGraph
import numpy as np

kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                    predicate=True, word_level=True)

text = "[CLS] Mercadona es excelente"
tokens, pos, vm, seg = kg.add_knowledge_with_vm([text], add_pad=True, max_length=128)

print("Text:", text)
print("Tokens:", tokens[0][:20])
print("Positions:", pos[0][:20])
print("Visible matrix shape:", vm[0].shape)
print("Visible matrix type:", type(vm[0]))
print("Visible matrix sample (primeras 10x10):")
print(vm[0][:10, :10])
print("Segment:", seg[0][:20])
print("\nCRITICAL CHECK:")
print("VM sum (should be small):", vm[0].sum())
print("VM dtype:", vm[0].dtype)
