#!/usr/bin/env python3
"""Fix: Pasar epoch real a logger.log_metrics"""

with open('run_kbert_ner.py', 'r') as f:
    content = f.read()

# Fix 1: Cambiar firma de evaluate() para recibir epoch
content = content.replace(
    'def evaluate(args, is_test):',
    'def evaluate(args, is_test, epoch=0):'
)
print("✓ Firma de evaluate() actualizada")

# Fix 2: Pasar epoch real a logger.log_metrics (en evaluate)
content = content.replace(
    'logger.log_metrics(epoch=0, split=\'dev\', metrics=metrics)',
    'logger.log_metrics(epoch=epoch, split=\'dev\', metrics=metrics)'
)
content = content.replace(
    'logger.log_metrics(epoch=0, split=\'test\', metrics=metrics)',
    'logger.log_metrics(epoch=epoch, split=\'test\', metrics=metrics)'
)
print("✓ epoch hardcoded → epoch variable")

# Fix 3: Actualizar llamadas a evaluate() en el training loop
# Buscar: f1, dev_metrics = evaluate(args, False)
# Reemplazar: f1, dev_metrics = evaluate(args, False, epoch)
content = content.replace(
    'f1, dev_metrics = evaluate(args, False)',
    'f1, dev_metrics = evaluate(args, False, epoch)'
)
print("✓ Llamada a evaluate() en training loop actualizada")

# Fix 4: Actualizar llamada final a evaluate() para test
# Buscar: evaluate(args, True)
# Reemplazar: evaluate(args, True, args.epochs_num)
import re
# Buscar el último evaluate(args, True) que está en el final
content = re.sub(
    r'(Final evaluation on test dataset\..*?)\n\s+evaluate\(args, True\)',
    r'\1\n    evaluate(args, True, args.epochs_num)',
    content,
    flags=re.DOTALL
)
print("✓ Llamada final a evaluate() actualizada")

with open('run_kbert_ner.py', 'w') as f:
    f.write(content)

print("\n✓ Fixes aplicados - métricas ahora guardarán epoch correcto")
