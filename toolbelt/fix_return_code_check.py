#!/usr/bin/env python3
"""Fix: Verificar éxito por archivos generados, no por return code"""

with open('run_experiments.py', 'r') as f:
    content = f.read()

# Reemplazar chequeo de returncode
old_check = '''                if process.returncode == 0:
                    elapsed = time.time() - start_time
                    print(f"✅ COMPLETADO: {exp_name} ({elapsed/60:.1f} min)")
                    self.progress['completed'].append(exp_name)
                    self.progress['current'] = None
                    self.save_progress()
                    return True
                else:
                    print(f"❌ ERROR: {exp_name} (return code: {process.returncode})")
                    self.progress['failed'].append({
                        'name': exp_name,
                        'error': f'Return code {process.returncode}',
                        'log': log_file
                    })
                    self.progress['current'] = None
                    self.save_progress()
                    return False'''

new_check = '''                elapsed = time.time() - start_time
                
                # Verificar éxito por existencia de archivos, no return code
                metrics_file = f"{self.base_config['output_dir']}/{exp_name}_metrics.csv"
                if os.path.exists(metrics_file):
                    print(f"✅ COMPLETADO: {exp_name} ({elapsed/60:.1f} min)")
                    self.progress['completed'].append(exp_name)
                    self.progress['current'] = None
                    self.save_progress()
                    return True
                else:
                    print(f"❌ ERROR: {exp_name} (no metrics file, return code: {process.returncode})")
                    self.progress['failed'].append({
                        'name': exp_name,
                        'error': f'No metrics file, return code {process.returncode}',
                        'log': log_file
                    })
                    self.progress['current'] = None
                    self.save_progress()
                    return False'''

content = content.replace(old_check, new_check)

with open('run_experiments.py', 'w') as f:
    f.write(content)

print("✓ Fix aplicado: Verifica existencia de archivos en lugar de return code")
