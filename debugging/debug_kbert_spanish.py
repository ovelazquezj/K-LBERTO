#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/omar/projects/K-LBERTO')

from brain import KnowledgeGraph
from uer.utils.vocab import Vocab

print("="*70)
print("VERIFICACIÓN: ¿Funcionan los valores españoles limpios?")
print("="*70)

# USAR EL ARCHIVO LIMPIO
kg = KnowledgeGraph(spo_files=['./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo'], 
                     predicate=True, word_level=True)

print(f"\n1. LOOKUP TABLE (Knowledge Graph LIMPIO)")
print(f"   Tamaño: {len(kg.lookup_table)}")
print(f"   Ejemplo valores:")
for key in list(kg.lookup_table.keys())[:5]:
    print(f"     '{key}' → {kg.lookup_table[key]}")

# Cargar vocab
vocab = Vocab()
vocab.load('./models/beto_uer_model/vocab.txt')

# Test
test_text = "[CLS] excelente producto muy bueno"
tokens, pos, vm, seg = kg.add_knowledge_with_vm([test_text], add_pad=True, max_length=128)
tokens = tokens[0][:15]

print(f"\n2. TOKENS INYECTADOS")
print(f"   Input: '{test_text}'")
print(f"   Output tokens: {tokens}")

print(f"\n3. VOCAB MATCHING")
for token in tokens[:10]:
    if token == '[PAD]':
        continue
    token_id = vocab.get(token)
    status = "✓" if token_id != 100 else "✗ [UNK]"
    print(f"   {status} '{token}' → ID {token_id}")

print("\n" + "="*70 + "\n")
