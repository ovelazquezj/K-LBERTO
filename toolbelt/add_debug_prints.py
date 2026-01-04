#!/usr/bin/env python3
"""Agregar prints de debug para ver qué recibe calculate_ner_metrics"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Encontrar la línea donde se llama calculate_ner_metrics
for i, line in enumerate(lines):
    if 'metrics = calculate_ner_metrics' in line and 'all_preds_cat' not in line:
        # Agregar debug ANTES de la llamada
        indent = ' ' * 8
        debug_code = f'''
{indent}# ========== DEBUG START ==========
{indent}print(f"\\n[DEBUG] Llamando calculate_ner_metrics:")
{indent}print(f"  pred shape: {{pred.shape}}")
{indent}print(f"  gold shape: {{gold.shape}}")
{indent}print(f"  pred primeros 20: {{pred[:20].tolist()}}")
{indent}print(f"  gold primeros 20: {{gold[:20].tolist()}}")
{indent}print(f"  labels_map: {{labels_map}}")
{indent}print(f"  begin_ids: {{begin_ids}}")
{indent}print(f"  Unique en pred: {{torch.unique(pred).tolist()}}")
{indent}print(f"  Unique en gold: {{torch.unique(gold).tolist()}}")
{indent}# ========== DEBUG END ==========

'''
        lines.insert(i, debug_code)
        print(f"✓ Debug agregado antes de línea {i+1}")
        break

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Debug prints agregados")
