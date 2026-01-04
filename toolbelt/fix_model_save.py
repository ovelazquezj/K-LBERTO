#!/usr/bin/env python3
"""Fix: Cambiar condición de guardado de modelo"""

with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

# Cambiar > por >= para que guarde en primera época
content = content.replace(
    'if f1 > best_f1:',
    'if f1 >= best_f1:'
)

with open('run_kbert_ner.py', 'w') as f:
    f.write(content)

print("✓ Condición cambiada: f1 > best_f1 → f1 >= best_f1")
print("  Ahora guardará modelo incluso si f1=0.0")
