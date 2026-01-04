#!/usr/bin/env python3
"""Integrar métricas NER avanzadas en evaluate()"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# 1. Agregar import
import_added = False
for i, line in enumerate(lines):
    if 'from utils.metrics_logger import MetricsLogger' in line and not import_added:
        lines.insert(i+1, 'from utils.ner_metrics import calculate_ner_metrics\n')
        import_added = True
        print(f"✓ Import ner_metrics agregado en línea {i+2}")
        break

# 2. Encontrar función evaluate() y modificar el return
# Buscar "return f1" dentro de evaluate
for i, line in enumerate(lines):
    if 'def evaluate(args, is_test):' in line:
        # Buscar el final de evaluate donde está "return f1"
        for j in range(i, min(i+200, len(lines))):
            if '        return f1' in lines[j]:
                # Reemplazar con cálculo de métricas avanzadas
                indent = '        '
                new_code = f'''{indent}# Calculate detailed metrics
{indent}metrics = calculate_ner_metrics(pred, gold, labels_map, begin_ids)
{indent}
{indent}# Log metrics if logger available
{indent}if logger and not is_test:
{indent}    logger.log_metrics(epoch=0, split='dev', metrics=metrics)
{indent}elif logger and is_test:
{indent}    logger.log_metrics(epoch=0, split='test', metrics=metrics)
{indent}
{indent}return metrics['overall']['f1'], metrics
'''
                lines[j] = new_code
                print(f"✓ Return modificado en línea {j+1}")
                break
        break

# 3. Actualizar llamadas a evaluate() para recibir dos valores
for i, line in enumerate(lines):
    if '        f1 = evaluate(args, False)' in line:
        lines[i] = line.replace('f1 = evaluate(args, False)', 'f1, dev_metrics = evaluate(args, False)')
        print(f"✓ Llamada a evaluate() actualizada en línea {i+1}")

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Integración completada")
