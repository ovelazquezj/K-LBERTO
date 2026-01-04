#!/usr/bin/env python3
"""Fix: Acumular todas las predicciones antes de calcular métricas"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Encontrar inicio de evaluate() y agregar acumuladores
for i, line in enumerate(lines):
    if 'confusion = torch.zeros(len(labels_map)' in line:
        # Agregar acumuladores ANTES del batch loop
        indent = ' ' * 8
        accumulator_code = f'''{indent}# Acumuladores para todas las predicciones
{indent}all_preds = []
{indent}all_golds = []

'''
        lines.insert(i+1, accumulator_code)
        print(f"✓ Acumuladores agregados en línea {i+2}")
        break

# Encontrar dentro del batch loop donde se obtiene pred, gold
# y agregar la acumulación
for i, line in enumerate(lines):
    if 'loss, _, pred, gold = model(' in line:
        # Encontrar el siguiente bloque (después de esta línea)
        # Agregar acumulación
        indent = ' ' * 12
        accumulate_code = f'''{indent}# Acumular predicciones
{indent}all_preds.append(pred)
{indent}all_golds.append(gold)

'''
        lines.insert(i+1, accumulate_code)
        print(f"✓ Acumulación agregada en línea {i+2}")
        break

# Encontrar donde se llama calculate_ner_metrics y reemplazar
for i, line in enumerate(lines):
    if 'metrics = calculate_ner_metrics(pred, gold, labels_map, begin_ids)' in line:
        indent = ' ' * 8
        new_code = f'''{indent}# Concatenar todas las predicciones
{indent}if len(all_preds) > 0:
{indent}    all_preds_cat = torch.cat(all_preds, dim=0)
{indent}    all_golds_cat = torch.cat(all_golds, dim=0)
{indent}    metrics = calculate_ner_metrics(all_preds_cat, all_golds_cat, labels_map, begin_ids)
{indent}else:
{indent}    # Fallback: métricas vacías
{indent}    metrics = {{
{indent}        'overall': {{'precision': 0.0, 'recall': 0.0, 'f1': 0.0}},
{indent}        'PER': {{'precision': 0.0, 'recall': 0.0, 'f1': 0.0}},
{indent}        'LOC': {{'precision': 0.0, 'recall': 0.0, 'f1': 0.0}},
{indent}        'ORG': {{'precision': 0.0, 'recall': 0.0, 'f1': 0.0}},
{indent}        'counts': {{'correct': 0, 'pred_entities': 0, 'gold_entities': 0}}
{indent}    }}
'''
        lines[i] = new_code
        print(f"✓ Llamada a calculate_ner_metrics reemplazada en línea {i+1}")
        break

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Fix de acumulación aplicado")
print("Ahora calculate_ner_metrics() recibirá TODAS las predicciones")
