#!/usr/bin/env python3
"""
Consolidar resultados de todos los experimentos en un solo CSV
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import os
import json
import csv
import glob
from pathlib import Path
from datetime import datetime

class ResultsConsolidator:
    def __init__(self, results_dir='resultados', output_file='consolidated_results.csv'):
        self.results_dir = results_dir
        self.output_file = output_file
    
    def find_all_metrics_files(self):
        """Encontrar todos los archivos de métricas"""
        pattern = os.path.join(self.results_dir, '*_metrics.csv')
        return glob.glob(pattern)
    
    def extract_experiment_info(self, filename):
        """Extraer información del nombre del experimento"""
        # Formato: RQ1_D500_base_seed42_metrics.csv
        basename = os.path.basename(filename).replace('_metrics.csv', '')
        parts = basename.split('_')
        
        info = {
            'experiment_name': basename,
            'rq': parts[0] if len(parts) > 0 else 'unknown',
            'dataset_size': parts[1][1:] if len(parts) > 1 and parts[1].startswith('D') else 'unknown',
            'config_type': parts[2] if len(parts) > 2 else 'unknown',
            'seed': parts[3].replace('seed', '') if len(parts) > 3 else 'unknown'
        }
        return info
    
    def read_metrics_file(self, filepath):
        """Leer archivo de métricas y extraer datos"""
        results = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        
        return results
    
    def consolidate(self):
        """Consolidar todos los resultados"""
        all_results = []
        
        metrics_files = self.find_all_metrics_files()
        print(f"📊 Encontrados {len(metrics_files)} archivos de métricas")
        
        for filepath in sorted(metrics_files):
            exp_info = self.extract_experiment_info(filepath)
            metrics = self.read_metrics_file(filepath)
            
            # Solo tomar métricas finales (última evaluación en test)
            # Buscar la última entrada de 'test'
            test_metrics = [m for m in metrics if m['split'] == 'test']
            if not test_metrics:
                print(f"⚠️  No test metrics en {exp_info['experiment_name']}")
                continue
            
            final_metrics = test_metrics[-1]  # Última evaluación en test
            
            # Combinar info del experimento con métricas
            result = {
                'experiment_name': exp_info['experiment_name'],
                'rq': exp_info['rq'],
                'dataset_size': exp_info['dataset_size'],
                'config_type': exp_info['config_type'],
                'seed': exp_info['seed'],
                'final_epoch': final_metrics['epoch'],
                'precision_overall': final_metrics['precision_overall'],
                'recall_overall': final_metrics['recall_overall'],
                'f1_overall': final_metrics['f1_overall'],
                'precision_PER': final_metrics['precision_PER'],
                'recall_PER': final_metrics['recall_PER'],
                'f1_PER': final_metrics['f1_PER'],
                'precision_LOC': final_metrics['precision_LOC'],
                'recall_LOC': final_metrics['recall_LOC'],
                'f1_LOC': final_metrics['f1_LOC'],
                'precision_ORG': final_metrics['precision_ORG'],
                'recall_ORG': final_metrics['recall_ORG'],
                'f1_ORG': final_metrics['f1_ORG'],
                'correct': final_metrics['correct'],
                'pred_entities': final_metrics['pred_entities'],
                'gold_entities': final_metrics['gold_entities'],
                'timestamp': final_metrics['timestamp']
            }
            
            all_results.append(result)
        
        # Escribir archivo consolidado
        if all_results:
            fieldnames = list(all_results[0].keys())
            
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"\n✅ Consolidado guardado en: {self.output_file}")
            print(f"   Total experimentos: {len(all_results)}")
            
            # Resumen por configuración
            self.print_summary(all_results)
        else:
            print("❌ No se encontraron resultados para consolidar")
    
    def print_summary(self, results):
        """Imprimir resumen de resultados"""
        print("\n" + "="*80)
        print("RESUMEN POR CONFIGURACIÓN")
        print("="*80)
        
        # Agrupar por dataset_size y config_type
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for r in results:
            key = f"D{r['dataset_size']}_{r['config_type']}"
            grouped[key].append(float(r['f1_overall']))
        
        print(f"\n{'Config':<20} {'N':>5} {'F1 Mean':>10} {'F1 Std':>10} {'F1 Min':>10} {'F1 Max':>10}")
        print("-" * 80)
        
        for key in sorted(grouped.keys()):
            f1_scores = grouped[key]
            n = len(f1_scores)
            mean = sum(f1_scores) / n if n > 0 else 0
            
            if n > 1:
                variance = sum((x - mean) ** 2 for x in f1_scores) / (n - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            
            min_f1 = min(f1_scores) if f1_scores else 0
            max_f1 = max(f1_scores) if f1_scores else 0
            
            print(f"{key:<20} {n:>5} {mean:>10.4f} {std:>10.4f} {min_f1:>10.4f} {max_f1:>10.4f}")
        
        print("="*80 + "\n")


def main():
    consolidator = ResultsConsolidator()
    consolidator.consolidate()


if __name__ == '__main__':
    main()
