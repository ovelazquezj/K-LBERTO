#!/usr/bin/env python3
"""
Script para ejecutar experimentos restantes RQ1
Usa experiments_remaining.json
"""

import json
import subprocess
import sys
import os
from datetime import datetime

class RemainingExperimentsRunner:
    def __init__(self, config_file='experiments_remaining.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.base_config = self.config['base_config']
        self.experiments = self.config['experiments']
        self.hardware_limits = self.config.get('hardware_limits', {})
        
        # Progress tracking
        self.progress_file = 'experiments_progress_remaining.json'
        self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'completed': [],
                'failed': [],
                'skipped': [],
                'current': None
            }
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def run_experiment(self, exp):
        print("\n" + "="*80)
        print(f"🚀 INICIANDO: {exp['name']}")
        print(f"   Dataset: {exp.get('dataset_size', 'N/A')} samples")
        print(f"   Config: {exp.get('config_type', 'N/A')}")
        print(f"   LR: {exp['learning_rate']:.2e}")
        print(f"   Seed: {exp['seed']}")
        print("="*80)
        
        # Marcar como actual
        self.progress['current'] = {
            'name': exp['name'],
            'start_time': datetime.now().isoformat()
        }
        self.save_progress()
        
        # Construir comando
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
            '--learning_rate', str(exp['learning_rate']),
            '--report_steps', str(self.base_config['report_steps']),
            '--seed', str(exp['seed'])
        ]
        
        # Log
        log_file = f"logs/{exp['name']}.log"
        
        # Ejecutar
        try:
            with open(log_file, 'w') as log:
                process = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.hardware_limits.get('timeout_minutes', 210) * 60
                )
            
            # Verificar éxito
            metrics_file = f"{self.base_config['output_dir']}/{exp['name']}_metrics.csv"
            if os.path.exists(metrics_file):
                print(f"✅ ÉXITO: {exp['name']}")
                self.progress['completed'].append(exp['name'])
            else:
                print(f"❌ FALLADO: {exp['name']} (sin métricas)")
                self.progress['failed'].append({
                    'name': exp['name'],
                    'error': 'No metrics file'
                })
        
        except subprocess.TimeoutExpired:
            print(f"⏱️ TIMEOUT: {exp['name']}")
            self.progress['failed'].append({
                'name': exp['name'],
                'error': 'Timeout'
            })
        
        except Exception as e:
            print(f"❌ ERROR: {exp['name']}: {str(e)}")
            self.progress['failed'].append({
                'name': exp['name'],
                'error': str(e)
            })
        
        self.progress['current'] = None
        self.save_progress()
    
    def run_all(self):
        print("\n" + "="*80)
        print(f"EXPERIMENTOS RESTANTES RQ1: {len(self.experiments)} total")
        print("="*80)
        
        for i, exp in enumerate(self.experiments, 1):
            print(f"\n[{i}/{len(self.experiments)}] Experimento: {exp['name']}")
            
            # Skip si ya completado
            if exp['name'] in self.progress['completed']:
                print(f"⏭️  SKIP: {exp['name']} (ya completado)")
                self.progress['skipped'].append(exp['name'])
                continue
            
            self.run_experiment(exp)
        
        print("\n" + "="*80)
        print("EJECUCIÓN COMPLETADA")
        print("="*80)
        print(f"✅ Completados: {len(self.progress['completed'])}")
        print(f"❌ Fallados: {len(self.progress['failed'])}")
        print(f"⏭️  Saltados: {len(self.progress['skipped'])}")

if __name__ == '__main__':
    runner = RemainingExperimentsRunner()
    runner.run_all()
