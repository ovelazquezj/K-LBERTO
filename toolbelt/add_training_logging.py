#!/usr/bin/env python3
"""Agregar logging de loss en training loop"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Buscar el print de "Epoch id: X, Training steps: Y"
for i, line in enumerate(lines):
    if 'print("Epoch id: {}, Training steps: {}, Avg loss:' in line:
        # Agregar logging después del print
        indent = ' ' * 16  # Mismo indent que el print
        log_line = f'{indent}if logger:\n{indent}    logger.log_loss(epoch, i+1, total_loss / args.report_steps)\n'
        lines.insert(i+1, log_line)
        print(f"✓ Loss logging agregado en línea {i+2}")
        break

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("✓ Loss logging agregado")
