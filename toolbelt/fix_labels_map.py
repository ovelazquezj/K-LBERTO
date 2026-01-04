with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Encontrar y corregir la línea que construye labels_map (~línea 155)
for i, line in enumerate(lines):
    # Fix construcción de labels_map
    if 'labels = line.strip().split("\\t")[1].split()' in line:
        lines[i] = line.replace('[1].split()', '[0].split()')
        print(f"✓ Línea {i+1}: Corregido construcción labels_map")

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Fix aplicado")
