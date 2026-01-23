#!/usr/bin/env python3
"""
Análisis Parcial de Grid Search LR
==================================
Analiza resultados parciales del grid search para caracterizar η_óptimo = f(D)

Uso:
    python analyze_gridsearch_partial.py [--results-dir resultados] [--progress-file experiments_progress_gridsearch.json]
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Análisis parcial de Grid Search LR')
    parser.add_argument('--results-dir', default='resultados', help='Directorio de resultados')
    parser.add_argument('--progress-file', default='experiments_progress_gridsearch.json', help='Archivo de progreso')
    parser.add_argument('--output', default=None, help='Archivo de salida para el reporte')
    return parser.parse_args()

def load_progress(progress_file):
    """Carga el archivo de progreso"""
    if not os.path.exists(progress_file):
        return {'completed': [], 'failed': [], 'current': None}
    with open(progress_file, 'r') as f:
        return json.load(f)

def parse_experiment_name(name):
    """Extrae dataset_size y learning_rate del nombre del experimento
    
    Formato: GS_D{size}_LR{lr}
    Ejemplos: GS_D500_LR5e6, GS_D1000_LR2e5
    """
    try:
        parts = name.split('_')
        # D500, D1000, etc
        d_part = parts[1]
        dataset_size = int(d_part[1:])
        
        # LR5e6, LR1e5, etc
        lr_part = parts[2]
        lr_str = lr_part[2:]  # Quita "LR"
        
        # Convertir notación como "5e6" a float
        if 'e' in lr_str:
            base, exp = lr_str.split('e')
            learning_rate = float(base) * (10 ** (-int(exp)))
        else:
            learning_rate = float(lr_str)
        
        return dataset_size, learning_rate
    except Exception as e:
        print(f"Error parsing {name}: {e}")
        return None, None

def load_metrics(results_dir, experiment_name):
    """Carga métricas de un experimento completado"""
    metrics_file = os.path.join(results_dir, f"{experiment_name}_metrics.csv")
    
    if not os.path.exists(metrics_file):
        return None
    
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'best_f1': 0,
        'best_epoch': 0
    }
    
    with open(metrics_file, 'r') as f:
        header = f.readline().strip().split(',')
        
        # Encontrar índices de columnas
        epoch_idx = header.index('epoch') if 'epoch' in header else 0
        f1_idx = None
        loss_idx = None
        
        for i, h in enumerate(header):
            if 'f1' in h.lower() and f1_idx is None:
                f1_idx = i
            if 'loss' in h.lower() and 'val' not in h.lower() and loss_idx is None:
                loss_idx = i
        
        if f1_idx is None:
            f1_idx = 4  # Default based on previous format
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > f1_idx:
                try:
                    epoch = int(parts[epoch_idx])
                    f1 = float(parts[f1_idx])
                    
                    metrics['epochs'].append(epoch)
                    metrics['val_f1'].append(f1)
                    
                    if f1 > metrics['best_f1']:
                        metrics['best_f1'] = f1
                        metrics['best_epoch'] = epoch
                except (ValueError, IndexError):
                    continue
    
    return metrics

def calculate_sqrt_prediction(dataset_size, base_lr=2e-5, base_d=1000):
    """Calcula el LR predicho por √-scaling"""
    ratio = dataset_size / base_d
    return base_lr / math.sqrt(ratio)

def find_closest_lr(target_lr, available_lrs):
    """Encuentra el LR más cercano al target entre los disponibles"""
    return min(available_lrs, key=lambda x: abs(x - target_lr))

def generate_heatmap_ascii(results_matrix, datasets, lrs):
    """Genera un heatmap ASCII de los resultados"""
    
    # Header
    header = "         "
    for lr in lrs:
        header += f" {lr:.0e}  "
    
    lines = [header, "-" * len(header)]
    
    for d in datasets:
        row = f"D{d:4d} | "
        for lr in lrs:
            key = (d, lr)
            if key in results_matrix:
                f1 = results_matrix[key]['best_f1']
                # Color coding via symbols
                if f1 >= 0.6:
                    symbol = "█"
                elif f1 >= 0.5:
                    symbol = "▓"
                elif f1 >= 0.4:
                    symbol = "▒"
                elif f1 >= 0.3:
                    symbol = "░"
                else:
                    symbol = "·"
                row += f" {f1:.3f}{symbol}"
            else:
                row += "   ---  "
        lines.append(row)
    
    return "\n".join(lines)

def analyze_results(results_dir, progress):
    """Analiza los resultados completados"""
    
    completed = progress.get('completed', [])
    failed = progress.get('failed', [])
    
    # Estructuras de datos
    results_matrix = {}  # (dataset_size, lr) -> metrics
    datasets = set()
    lrs = set()
    
    # Cargar resultados
    for exp_name in completed:
        d, lr = parse_experiment_name(exp_name)
        if d is None:
            continue
        
        metrics = load_metrics(results_dir, exp_name)
        if metrics is None:
            continue
        
        datasets.add(d)
        lrs.add(lr)
        results_matrix[(d, lr)] = metrics
    
    datasets = sorted(datasets)
    lrs = sorted(lrs)
    
    return {
        'results_matrix': results_matrix,
        'datasets': datasets,
        'lrs': lrs,
        'completed': completed,
        'failed': failed
    }

def find_optimal_lr_per_dataset(results_matrix, datasets, lrs):
    """Encuentra el LR óptimo para cada dataset"""
    optimal = {}
    
    for d in datasets:
        best_f1 = -1
        best_lr = None
        all_f1s = {}
        
        for lr in lrs:
            key = (d, lr)
            if key in results_matrix:
                f1 = results_matrix[key]['best_f1']
                all_f1s[lr] = f1
                if f1 > best_f1:
                    best_f1 = f1
                    best_lr = lr
        
        if best_lr is not None:
            optimal[d] = {
                'best_lr': best_lr,
                'best_f1': best_f1,
                'all_f1s': all_f1s,
                'sqrt_predicted_lr': calculate_sqrt_prediction(d)
            }
    
    return optimal

def compare_with_sqrt_scaling(optimal, results_matrix):
    """Compara LR óptimo empírico vs predicción √-scaling"""
    comparison = []
    
    for d, opt in optimal.items():
        sqrt_lr = opt['sqrt_predicted_lr']
        
        # Encontrar el LR más cercano al sqrt predicho
        available_lrs = list(opt['all_f1s'].keys())
        closest_lr = find_closest_lr(sqrt_lr, available_lrs)
        
        f1_optimal = opt['best_f1']
        f1_sqrt = opt['all_f1s'].get(closest_lr, None)
        
        comparison.append({
            'dataset': d,
            'lr_optimal': opt['best_lr'],
            'f1_optimal': f1_optimal,
            'lr_sqrt_predicted': sqrt_lr,
            'lr_sqrt_closest': closest_lr,
            'f1_sqrt': f1_sqrt,
            'delta_f1': f1_optimal - f1_sqrt if f1_sqrt else None
        })
    
    return comparison

def generate_report(analysis, comparison, optimal):
    """Genera el reporte de análisis"""
    
    lines = []
    lines.append("=" * 70)
    lines.append("ANÁLISIS PARCIAL DE GRID SEARCH LR")
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Progreso
    total = 20
    completed = len(analysis['completed'])
    failed = len(analysis['failed'])
    
    lines.append(f"\n📊 PROGRESO: {completed}/{total} experimentos completados")
    if failed > 0:
        lines.append(f"❌ Fallados: {failed}")
        for f in analysis['failed']:
            lines.append(f"   - {f}")
    
    lines.append(f"\n📈 COBERTURA:")
    lines.append(f"   Datasets: {analysis['datasets']}")
    lines.append(f"   Learning Rates: {[f'{lr:.0e}' for lr in analysis['lrs']]}")
    
    # Heatmap
    if analysis['results_matrix']:
        lines.append(f"\n" + "=" * 70)
        lines.append("HEATMAP: F1 vs (Dataset Size × Learning Rate)")
        lines.append("=" * 70)
        lines.append(generate_heatmap_ascii(
            analysis['results_matrix'], 
            analysis['datasets'], 
            analysis['lrs']
        ))
        lines.append("\nLeyenda: █ ≥0.6 | ▓ ≥0.5 | ▒ ≥0.4 | ░ ≥0.3 | · <0.3")
    
    # LR Óptimo por Dataset
    if optimal:
        lines.append(f"\n" + "=" * 70)
        lines.append("LR ÓPTIMO POR DATASET SIZE")
        lines.append("=" * 70)
        lines.append(f"{'Dataset':<10} {'LR Óptimo':<12} {'Best F1':<10} {'√-Predicted':<12}")
        lines.append("-" * 50)
        
        for d in sorted(optimal.keys()):
            opt = optimal[d]
            lines.append(
                f"D{d:<9} {opt['best_lr']:<12.2e} {opt['best_f1']:<10.4f} {opt['sqrt_predicted_lr']:<12.2e}"
            )
    
    # Comparación con √-scaling
    if comparison:
        lines.append(f"\n" + "=" * 70)
        lines.append("COMPARACIÓN: LR ÓPTIMO vs √-SCALING")
        lines.append("=" * 70)
        lines.append(f"{'Dataset':<8} {'LR_opt':<10} {'F1_opt':<8} {'LR_√':<10} {'F1_√':<8} {'ΔF1':<8}")
        lines.append("-" * 60)
        
        total_delta = 0
        valid_comparisons = 0
        
        for c in comparison:
            delta_str = f"{c['delta_f1']:+.4f}" if c['delta_f1'] is not None else "N/A"
            f1_sqrt_str = f"{c['f1_sqrt']:.4f}" if c['f1_sqrt'] is not None else "N/A"
            
            lines.append(
                f"D{c['dataset']:<7} {c['lr_optimal']:<10.2e} {c['f1_optimal']:<8.4f} "
                f"{c['lr_sqrt_closest']:<10.2e} {f1_sqrt_str:<8} {delta_str:<8}"
            )
            
            if c['delta_f1'] is not None:
                total_delta += c['delta_f1']
                valid_comparisons += 1
        
        if valid_comparisons > 0:
            avg_delta = total_delta / valid_comparisons
            lines.append("-" * 60)
            lines.append(f"Promedio ΔF1 (óptimo - √-scaling): {avg_delta:+.4f}")
            
            if avg_delta > 0.02:
                lines.append("\n⚠️  LR óptimo empírico SUPERA √-scaling por >2%")
            elif avg_delta < -0.02:
                lines.append("\n⚠️  √-scaling SUPERA LR óptimo empírico por >2%")
            else:
                lines.append("\n✅ Diferencia dentro del margen de ruido (<2%)")
    
    # Análisis de patrones
    if len(optimal) >= 2:
        lines.append(f"\n" + "=" * 70)
        lines.append("ANÁLISIS DE PATRONES")
        lines.append("=" * 70)
        
        # ¿El LR óptimo es constante?
        best_lrs = [optimal[d]['best_lr'] for d in sorted(optimal.keys())]
        unique_lrs = set(best_lrs)
        
        if len(unique_lrs) == 1:
            lines.append(f"\n🔍 PATRÓN DETECTADO: LR óptimo CONSTANTE = {best_lrs[0]:.2e}")
            lines.append("   → Esto sugiere que √-scaling NO es necesario")
        else:
            lines.append(f"\n🔍 LR óptimo varía entre datasets:")
            for d in sorted(optimal.keys()):
                lines.append(f"   D{d}: {optimal[d]['best_lr']:.2e}")
            
            # ¿Sigue algún patrón?
            # Verificar si decrece con D (como √-scaling predeciría)
            sorted_d = sorted(optimal.keys())
            lr_sequence = [optimal[d]['best_lr'] for d in sorted_d]
            
            is_decreasing = all(lr_sequence[i] >= lr_sequence[i+1] for i in range(len(lr_sequence)-1))
            is_increasing = all(lr_sequence[i] <= lr_sequence[i+1] for i in range(len(lr_sequence)-1))
            
            if is_decreasing:
                lines.append("   → Tendencia DECRECIENTE (consistente con scaling)")
            elif is_increasing:
                lines.append("   → Tendencia CRECIENTE (OPUESTO a √-scaling)")
            else:
                lines.append("   → Sin tendencia clara")
    
    # Conclusiones preliminares
    lines.append(f"\n" + "=" * 70)
    lines.append("CONCLUSIONES PRELIMINARES")
    lines.append("=" * 70)
    
    if completed < 10:
        lines.append("\n⚠️  Datos insuficientes para conclusiones robustas")
        lines.append(f"   Completados: {completed}/20")
        lines.append("   Recomendación: Esperar más experimentos")
    else:
        lines.append(f"\n📊 Basado en {completed}/20 experimentos:")
        
        if comparison:
            avg_delta = sum(c['delta_f1'] for c in comparison if c['delta_f1']) / len([c for c in comparison if c['delta_f1']])
            
            if avg_delta > 0.01:
                lines.append("   1. LR empírico supera √-scaling en promedio")
                lines.append("   2. √-scaling parece ser SUBÓPTIMO para este task")
            elif avg_delta < -0.01:
                lines.append("   1. √-scaling es competitivo o mejor que grid search")
                lines.append("   2. Puede indicar que √-scaling es útil (inesperado)")
            else:
                lines.append("   1. Diferencias marginales entre estrategias")
                lines.append("   2. LR puede ser relativamente robusto")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)

def main():
    args = parse_args()
    
    # Cargar progreso
    progress = load_progress(args.progress_file)
    
    # Analizar resultados
    analysis = analyze_results(args.results_dir, progress)
    
    if not analysis['results_matrix']:
        print("❌ No hay resultados completados aún")
        print(f"   Progreso: {len(progress.get('completed', []))}/20")
        if progress.get('current'):
            print(f"   En ejecución: {progress['current'].get('name', 'unknown')}")
        return
    
    # Encontrar LR óptimo por dataset
    optimal = find_optimal_lr_per_dataset(
        analysis['results_matrix'], 
        analysis['datasets'], 
        analysis['lrs']
    )
    
    # Comparar con √-scaling
    comparison = compare_with_sqrt_scaling(optimal, analysis['results_matrix'])
    
    # Generar reporte
    report = generate_report(analysis, comparison, optimal)
    
    # Mostrar o guardar
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n📄 Reporte guardado en: {args.output}")

if __name__ == "__main__":
    main()
