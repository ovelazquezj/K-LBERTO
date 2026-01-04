with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Encontrar líneas de división en evaluate()
    if '        p = correct/pred_entities_num' in line:
        # Reemplazar con versión segura
        lines[i] = '        p = correct/pred_entities_num if pred_entities_num > 0 else 0.0\n'
        print(f"✓ Línea {i+1}: Fix precision")
    
    if '        r = correct/gold_entities_num' in line:
        lines[i] = '        r = correct/gold_entities_num if gold_entities_num > 0 else 0.0\n'
        print(f"✓ Línea {i+1}: Fix recall")
    
    if '        f1 = 2*p*r/(p+r)' in line:
        lines[i] = '        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0\n'
        print(f"✓ Línea {i+1}: Fix F1")

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ División por cero corregida")
