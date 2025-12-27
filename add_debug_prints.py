import sys

# Leer archivo
with open('./run_kbert_cls.py', 'r') as f:
    lines = f.readlines()

# Encontrar la línea donde se calcula softmax (línea ~400)
# Insertar prints ANTES de argmax

insert_pos = None
for i, line in enumerate(lines):
    if 'logits = nn.Softmax(dim=1)(logits)' in line:
        insert_pos = i + 1
        break

if insert_pos:
    debug_code = '''                # DEBUG: Print logits y predicciones para primeros ejemplos
                if i == 0:  # Solo primer batch
                    print("\\n[DEBUG] PRIMEROS 5 EJEMPLOS DEL BATCH:")
                    for j in range(min(5, logits.size(0))):
                        print(f"  Ejemplo {j}:")
                        print(f"    Label real: {label_ids_batch[j].item()}")
                        print(f"    Logits softmax: {logits[j].cpu().numpy()}")
                        print(f"    Predicción: {torch.argmax(logits[j]).item()}")
                        print(f"    Confianza: {logits[j].max().item():.4f}")
                    print()
'''
    lines.insert(insert_pos, debug_code)
    
    with open('./run_kbert_cls.py', 'w') as f:
        f.writelines(lines)
    
    print("✓ Debug prints agregados")
else:
    print("❌ No se encontró la línea de softmax")
