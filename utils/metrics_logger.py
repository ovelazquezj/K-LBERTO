#!/usr/bin/env python3
"""
Sistema de Logging de Métricas para Investigación Dataset Scaling
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import os
import csv
import json
from datetime import datetime
import numpy as np


class MetricsLogger:
    """Logger para métricas de entrenamiento NER con K-LBERTO"""
    
    def __init__(self, output_dir, experiment_name):
        """
        Args:
            output_dir: Directorio base para resultados
            experiment_name: Nombre del experimento (ej: "D500_LR_sqrt_seed42")
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths de archivos
        self.loss_file = os.path.join(output_dir, f"{experiment_name}_loss.csv")
        self.metrics_file = os.path.join(output_dir, f"{experiment_name}_metrics.csv")
        self.config_file = os.path.join(output_dir, f"{experiment_name}_config.json")
        
        # Inicializar archivos CSV
        self._init_loss_file()
        self._init_metrics_file()
    
    def _init_loss_file(self):
        """Inicializar archivo de loss"""
        with open(self.loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'step', 'loss', 'timestamp'
            ])
    
    def _init_metrics_file(self):
        """Inicializar archivo de métricas"""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'split', 
                'precision_overall', 'recall_overall', 'f1_overall',
                'precision_PER', 'recall_PER', 'f1_PER',
                'precision_LOC', 'recall_LOC', 'f1_LOC',
                'precision_ORG', 'recall_ORG', 'f1_ORG',
                'correct', 'pred_entities', 'gold_entities',
                'timestamp'
            ])
    
    def log_loss(self, epoch, step, loss):
        """
        Registrar loss de training
        
        Args:
            epoch: Número de época
            step: Paso dentro de época
            loss: Valor del loss
        """
        with open(self.loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                step, 
                f"{loss:.6f}",
                datetime.now().isoformat()
            ])
    
    def log_metrics(self, epoch, split, metrics):
        """
        Registrar métricas de evaluación
        
        Args:
            epoch: Número de época
            split: 'dev' o 'test'
            metrics: Dict con estructura:
                {
                    'overall': {'precision': float, 'recall': float, 'f1': float},
                    'PER': {'precision': float, 'recall': float, 'f1': float},
                    'LOC': {...},
                    'ORG': {...},
                    'counts': {'correct': int, 'pred_entities': int, 'gold_entities': int}
                }
        """
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Overall metrics
            overall = metrics.get('overall', {'precision': 0, 'recall': 0, 'f1': 0})
            
            # Per-entity metrics
            per_metrics = metrics.get('PER', {'precision': 0, 'recall': 0, 'f1': 0})
            loc_metrics = metrics.get('LOC', {'precision': 0, 'recall': 0, 'f1': 0})
            org_metrics = metrics.get('ORG', {'precision': 0, 'recall': 0, 'f1': 0})
            
            # Counts
            counts = metrics.get('counts', {
                'correct': 0, 
                'pred_entities': 0, 
                'gold_entities': 0
            })
            
            writer.writerow([
                epoch,
                split,
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
                f"{overall['f1']:.4f}",
                f"{per_metrics['precision']:.4f}",
                f"{per_metrics['recall']:.4f}",
                f"{per_metrics['f1']:.4f}",
                f"{loc_metrics['precision']:.4f}",
                f"{loc_metrics['recall']:.4f}",
                f"{loc_metrics['f1']:.4f}",
                f"{org_metrics['precision']:.4f}",
                f"{org_metrics['recall']:.4f}",
                f"{org_metrics['f1']:.4f}",
                counts['correct'],
                counts['pred_entities'],
                counts['gold_entities'],
                datetime.now().isoformat()
            ])
    
    def save_config(self, config_dict):
        """
        Guardar configuración del experimento
        
        Args:
            config_dict: Diccionario con configuración experimental
        """
        # Agregar timestamp
        config_dict['experiment_name'] = self.experiment_name
        config_dict['logged_at'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_summary(self):
        """Obtener resumen de métricas del experimento"""
        summary = {
            'experiment_name': self.experiment_name,
            'files': {
                'loss': self.loss_file,
                'metrics': self.metrics_file,
                'config': self.config_file
            }
        }
        
        # Leer última métrica si existe
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Hay datos además del header
                    last_line = lines[-1].strip().split(',')
                    summary['last_f1'] = float(last_line[4])  # f1_overall
                    summary['last_epoch'] = int(last_line[0])
        
        return summary


def test_logger():
    """Test del MetricsLogger"""
    print("="*60)
    print("TESTING MetricsLogger")
    print("="*60)
    
    # Crear logger de prueba
    logger = MetricsLogger(
        output_dir='test_logs',
        experiment_name='test_D500_sqrt_seed42'
    )
    
    # Guardar configuración
    config = {
        'dataset_size': 500,
        'learning_rate': 2e-5,
        'lr_strategy': 'sqrt',
        'batch_size': 4,
        'epochs': 3,
        'seed': 42
    }
    logger.save_config(config)
    print(f"✓ Config guardado: {logger.config_file}")
    
    # Simular logging de loss
    for epoch in range(1, 4):
        for step in range(1, 6):
            loss = 2.0 / (epoch * step)  # Loss simulado
            logger.log_loss(epoch, step, loss)
    print(f"✓ Loss logged: {logger.loss_file}")
    
    # Simular logging de métricas
    for epoch in range(1, 4):
        metrics = {
            'overall': {
                'precision': 0.5 + epoch * 0.1,
                'recall': 0.4 + epoch * 0.1,
                'f1': 0.45 + epoch * 0.1
            },
            'PER': {
                'precision': 0.6 + epoch * 0.05,
                'recall': 0.5 + epoch * 0.05,
                'f1': 0.55 + epoch * 0.05
            },
            'LOC': {
                'precision': 0.4 + epoch * 0.08,
                'recall': 0.3 + epoch * 0.08,
                'f1': 0.35 + epoch * 0.08
            },
            'ORG': {
                'precision': 0.5 + epoch * 0.06,
                'recall': 0.4 + epoch * 0.06,
                'f1': 0.45 + epoch * 0.06
            },
            'counts': {
                'correct': 100 * epoch,
                'pred_entities': 200 * epoch,
                'gold_entities': 180 * epoch
            }
        }
        
        logger.log_metrics(epoch, 'dev', metrics)
        logger.log_metrics(epoch, 'test', metrics)
    
    print(f"✓ Metrics logged: {logger.metrics_file}")
    
    # Obtener resumen
    summary = logger.get_summary()
    print("\n" + "="*60)
    print("RESUMEN:")
    print("="*60)
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*60)
    print("TEST COMPLETADO - Revisa test_logs/")
    print("="*60)


if __name__ == "__main__":
    test_logger()
