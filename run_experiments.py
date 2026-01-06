#!/usr/bin/env python3
"""
Script maestro para ejecutar experimentos de forma automatizada
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
import subprocess
import os
import time
import math
from datetime import datetime
from pathlib import Path

class ExperimentRunner:
    def __init__(self, config_file='experiments_config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.base_config = self.config['base_config']
        self.progress_file = 'experiments_progress.json'
        self.load_progress()
    
    def load_progress(self):
        """Cargar progreso previo si existe"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'started_at': datetime.now().isoformat(),
                'completed': [],
                'failed': [],
                'skipped': [],
                'current': None
            }
    
    def save_progress(self):
        """Guardar progreso actual"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def calculate_lr(self, dataset_size, config_type, base_lr=2e-5, base_size=500):
        """Calcular learning rate según configuración"""
        if config_type == 'base':
            return base_lr
        elif config_type == 'sqrt':
            ratio = dataset_size / base_size
            return base_lr / math.sqrt(ratio)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def generate_experiments_rq1(self):
        """Generar lista de experimentos para RQ1"""
        rq1 = self.config['research_questions']['RQ1']
        experiments = []
        
        for dataset_size in rq1['datasets']:
            for seed in rq1['seeds']:
                for config in rq1['configs']:
                    lr = self.calculate_lr(
                        dataset_size, 
                        config['name'],
                        rq1['base_lr']
                    )
                    
                    exp = {
                        'name': f"RQ1_D{dataset_size}_{config['name']}_seed{seed}",
                        'dataset_size': dataset_size,
                        'config_type': config['name'],
                        'learning_rate': lr,
                        'seed': seed,
                        'train_path': f"data/wikiann_tsv/train_{dataset_size}.tsv"
                    }
                    experiments.append(exp)
        
        return experiments
    
    def build_command(self, exp):
        """Construir comando de ejecución"""
        cmd = [
            'python', 'run_kbert_ner.py',
            '--pretrained_model_path', self.base_config['pretrained_model_path'],
            '--vocab_path', self.base_config['vocab_path'],
            '--config_path', self.base_config['config_path'],
            '--train_path', exp['train_path'],
            '--dev_path', self.base_config['dev_path'],
            '--test_path', self.base_config['test_path'],
            '--kg_name', self.base_config['kg_name'],
            '--output_model_path', f"outputs/{exp['name']}.bin",
            '--experiment_name', exp['name'],
            '--output_dir', self.base_config['output_dir'],
            '--epochs_num', str(self.base_config['epochs_num']),
            '--batch_size', str(self.base_config['batch_size']),
            '--seq_length', str(self.base_config['seq_length']),
            '--learning_rate', f"{exp['learning_rate']:.2e}",
            '--report_steps', str(self.base_config['report_steps']),
            '--seed', str(exp['seed'])
        ]
        return cmd
    
    def run_experiment(self, exp):
        """Ejecutar un experimento individual"""
        exp_name = exp['name']
        
        # Verificar si ya completó
        if exp_name in self.progress['completed']:
            print(f"⏭️  SKIP: {exp_name} (ya completado)")
            self.progress['skipped'].append(exp_name)
            self.save_progress()
            return True
        
        print(f"\n{'='*80}")
        print(f"🚀 INICIANDO: {exp_name}")
        print(f"   Dataset: {exp['dataset_size']} samples")
        print(f"   Config: {exp['config_type']}")
        print(f"   LR: {exp['learning_rate']:.2e}")
        print(f"   Seed: {exp['seed']}")
        print(f"{'='*80}\n")
        
        self.progress['current'] = {
            'name': exp_name,
            'started_at': datetime.now().isoformat()
        }
        self.save_progress()
        
        # Construir comando
        cmd = self.build_command(exp)
        
        # Ejecutar
        log_file = f"logs/{exp_name}.log"
        os.makedirs('logs', exist_ok=True)
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Esperar con timeout
                timeout = self.config['hardware_limits']['timeout_minutes'] * 60
                process.wait(timeout=timeout)
                
                elapsed = time.time() - start_time
                
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
                    return False
        
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"⏰ TIMEOUT: {exp_name} (>{timeout/60:.0f} min)")
            self.progress['failed'].append({
                'name': exp_name,
                'error': 'Timeout',
                'log': log_file
            })
            self.progress['current'] = None
            self.save_progress()
            return False
        
        except Exception as e:
            print(f"💥 EXCEPCIÓN: {exp_name}")
            print(f"   Error: {str(e)}")
            self.progress['failed'].append({
                'name': exp_name,
                'error': str(e),
                'log': log_file
            })
            self.progress['current'] = None
            self.save_progress()
            return False
    
    def run_all_rq1(self):
        """Ejecutar todos los experimentos de RQ1"""
        experiments = self.generate_experiments_rq1()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENTOS RQ1: {len(experiments)} total")
        print(f"{'='*80}\n")
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Experimento: {exp['name']}")
            self.run_experiment(exp)
        
        # Resumen final
        print(f"\n{'='*80}")
        print("RESUMEN FINAL")
        print(f"{'='*80}")
        print(f"✅ Completados: {len(self.progress['completed'])}")
        print(f"⏭️  Saltados: {len(self.progress['skipped'])}")
        print(f"❌ Fallados: {len(self.progress['failed'])}")
        print(f"{'='*80}\n")
        
        if self.progress['failed']:
            print("Experimentos fallados:")
            for fail in self.progress['failed']:
                print(f"  - {fail['name']}: {fail['error']}")
                print(f"    Log: {fail['log']}")


def main():
    runner = ExperimentRunner()
    runner.run_all_rq1()


if __name__ == '__main__':
    main()
