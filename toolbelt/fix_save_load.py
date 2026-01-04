#!/usr/bin/env python3
"""Fix completo: guardado y carga de modelo"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Fix 1: Cambiar > a >= 
for i, line in enumerate(lines):
    if 'if f1 > best_f1:' in line:
        lines[i] = line.replace('if f1 > best_f1:', 'if f1 >= best_f1:')
        print(f"✓ Línea {i+1}: Cambiado f1 > best_f1 a f1 >= best_f1")

# Fix 2: Agregar check antes de cargar modelo
for i, line in enumerate(lines):
    if 'model.load_state_dict(torch.load(args.output_model_path))' in line:
        # Obtener indentación
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent
        
        # Agregar check de existencia
        check_code = f'''{spaces}if os.path.exists(args.output_model_path):
{spaces}    model.load_state_dict(torch.load(args.output_model_path))
{spaces}    print("Loaded best model from {{}}".format(args.output_model_path))
{spaces}else:
{spaces}    print("Warning: No saved model found at {{}}. Using current model.".format(args.output_model_path))
'''
        # Reemplazar línea original
        lines[i] = check_code
        print(f"✓ Línea {i+1}: Agregado check de existencia antes de cargar")
        break

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Fixes aplicados:")
print("  1. Guardará modelo cuando f1 >= best_f1 (incluye 0.0)")
print("  2. Verificará existencia antes de cargar")
