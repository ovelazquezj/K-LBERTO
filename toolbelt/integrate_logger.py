#!/usr/bin/env python3
"""Script para integrar MetricsLogger en run_kbert_ner.py"""

with open('run_kbert_ner.py', 'r') as f:
    lines = f.readlines()

# Encontrar línea de imports y agregar logger
import_added = False
for i, line in enumerate(lines):
    if 'from brain import KnowledgeGraph' in line and not import_added:
        lines.insert(i+1, 'from utils.metrics_logger import MetricsLogger\n')
        import_added = True
        print(f"✓ Import agregado en línea {i+2}")
        break

# Encontrar función main() y agregar inicialización del logger
# Buscar después de set_seed
for i, line in enumerate(lines):
    if 'set_seed(args.seed)' in line:
        # Agregar logger después de set_seed
        indent = '    '
        logger_init = f'''
{indent}# Initialize metrics logger
{indent}if hasattr(args, 'experiment_name') and args.experiment_name:
{indent}    logger = MetricsLogger(
{indent}        output_dir=args.output_dir if hasattr(args, 'output_dir') else 'resultados',
{indent}        experiment_name=args.experiment_name
{indent}    )
{indent}    # Save experiment config
{indent}    config_dict = {{
{indent}        'train_path': args.train_path,
{indent}        'batch_size': args.batch_size,
{indent}        'learning_rate': args.learning_rate,
{indent}        'epochs_num': args.epochs_num,
{indent}        'seq_length': args.seq_length,
{indent}        'seed': args.seed,
{indent}        'kg_name': args.kg_name
{indent}    }}
{indent}    logger.save_config(config_dict)
{indent}    print(f"[Logger] Initialized: {{logger.experiment_name}}")
{indent}else:
{indent}    logger = None
{indent}    print("[Logger] No experiment_name provided - logging disabled")

'''
        lines.insert(i+1, logger_init)
        print(f"✓ Logger init agregado en línea {i+2}")
        break

with open('run_kbert_ner.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Integración básica completada")
print("Ahora necesitas agregar manualmente:")
print("  1. Argumentos --experiment_name y --output_dir en parser")
print("  2. logger.log_loss() en loop de training")
print("  3. Modificar evaluate() para retornar métricas detalladas")
print("  4. logger.log_metrics() después de evaluate()")
