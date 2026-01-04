#!/usr/bin/env python3
"""Agregar argumentos de logging al parser"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Buscar la línea con --kg_name y agregar después
found = False
for i, line in enumerate(lines):
    if 'parser.add_argument("--kg_name"' in line:
        # Avanzar hasta encontrar la siguiente línea que no sea continuación
        j = i + 1
        while j < len(lines) - 1 and lines[j].strip() and not lines[j].strip().startswith('parser.add_argument'):
            j += 1
        
        # Agregar nuevos argumentos en la línea j
        new_args = '''
    # Logging options
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging (e.g., D500_sqrt_seed42)")
    parser.add_argument("--output_dir", type=str, default="resultados",
                        help="Output directory for logs and results")

'''
        lines.insert(j, new_args)
        print(f"✓ Argumentos agregados en línea {j}")
        found = True
        break

if not found:
    print("❌ No se encontró --kg_name")
    print("Agregando manualmente antes de args = parser.parse_args()")
    
    # Buscar args = parser.parse_args()
    for i, line in enumerate(lines):
        if 'args = parser.parse_args()' in line:
            new_args = '''
    # Logging options
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging (e.g., D500_sqrt_seed42)")
    parser.add_argument("--output_dir", type=str, default="resultados",
                        help="Output directory for logs and results")

'''
            lines.insert(i, new_args)
            print(f"✓ Argumentos agregados antes de parse_args() en línea {i}")
            found = True
            break

if found:
    with open('run_kbert_ner.py', 'w') as f:
        f.writelines(lines)
    print("✓ Archivo actualizado")
else:
    print("❌ No se pudo agregar - revisa manualmente")
