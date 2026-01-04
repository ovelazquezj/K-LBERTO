#!/usr/bin/env python3
with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

if 'import os' not in content:
    # Agregar después de los primeros imports
    content = content.replace(
        'import random\n',
        'import os\nimport random\n'
    )
    with open('run_kbert_ner.py', 'w') as f:
        f.write(content)
    print("✓ import os agregado")
else:
    print("✓ import os ya existe")
