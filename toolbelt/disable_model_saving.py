#!/usr/bin/env python3
"""Modificar run_kbert_ner.py para NO guardar modelos"""

with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

# Comentar línea de guardado de modelo
content = content.replace(
    'save_model(model, args.output_model_path)',
    '# save_model(model, args.output_model_path)  # DESHABILITADO: ahorro de espacio'
)

# Comentar carga final de modelo
content = content.replace(
    'model.load_state_dict(torch.load(args.output_model_path))',
    '# model.load_state_dict(torch.load(args.output_model_path))  # DESHABILITADO: ahorro de espacio'
)

content = content.replace(
    'model.module.load_state_dict(torch.load(args.output_model_path))',
    '# model.module.load_state_dict(torch.load(args.output_model_path))  # DESHABILITADO: ahorro de espacio'
)

with open('run_kbert_ner.py', 'w') as f:
    f.write(content)

print("✓ Guardado de modelos deshabilitado")
print("✓ Solo se guardarán métricas (CSV) y configs (JSON)")
print("✓ Espacio requerido: ~500 MB en lugar de ~21 GB")
