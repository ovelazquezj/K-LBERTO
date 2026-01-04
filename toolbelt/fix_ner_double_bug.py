with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

# FIX #1: Invertir orden de variables (línea ~222)
content = content.replace(
    '                tokens, labels = line.strip().split("\\t")',
    '                labels, tokens = line.strip().split("\\t")'
)

# FIX #2: Usar vocab.get('[PAD]') en lugar de PAD_ID
content = content.replace(
    '                    if tag[i] == 0 and tokens[i] != PAD_ID:',
    '                    if tag[i] == 0 and tokens[i] != vocab.get("[PAD]"):'
)

content = content.replace(
    '                    elif tag[i] == 1 and tokens[i] != PAD_ID:',
    '                    elif tag[i] == 1 and tokens[i] != vocab.get("[PAD]"):'
)

with open('run_kbert_ner.py', 'w') as f:
    f.write(content)

print("✓ Doble fix aplicado:")
print("  1. Invertido orden: labels, tokens = split()")
print("  2. Cambiado PAD_ID → vocab.get('[PAD]')")
