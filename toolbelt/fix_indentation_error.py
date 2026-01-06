#!/usr/bin/env python3
"""Fix: Agregar pass después de comentar save_model"""

with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

# Fix el bloque if que quedó vacío
old_block = '''        if f1 >= best_f1:
            best_f1 = f1
            # save_model(model, args.output_model_path)  # DESHABILITADO: ahorro de espacio
        else:
            continue'''

new_block = '''        if f1 >= best_f1:
            best_f1 = f1
            # save_model(model, args.output_model_path)  # DESHABILITADO: ahorro de espacio
            pass  # Placeholder para evitar IndentationError'''

content = content.replace(old_block, new_block)

# Fix el bloque del final también
old_final = '''    if torch.cuda.device_count() > 1:
        # model.module.load_state_dict(torch.load(args.output_model_path))  # DESHABILITADO: ahorro de espacio
    else:
        if os.path.exists(args.output_model_path):
            # model.load_state_dict(torch.load(args.output_model_path))  # DESHABILITADO: ahorro de espacio'''

new_final = '''    if torch.cuda.device_count() > 1:
        pass  # model.module.load_state_dict deshabilitado
    else:
        if os.path.exists(args.output_model_path):
            pass  # model.load_state_dict deshabilitado'''

content = content.replace(old_final, new_final)

with open('run_kbert_ner.py', 'w') as f:
    f.write(content)

print("✓ IndentationError corregido")
print("✓ Agregado 'pass' en bloques comentados")
