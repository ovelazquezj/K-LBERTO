#!/usr/bin/env python3
"""
Análisis preliminar RQ1: Comparación base vs sqrt scaling
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_metrics(results_dir="resultados"):
    """Carga todas las métricas de RQ1"""
    data = []
    skipped = []
    
    for metrics_file in Path(results_dir).glob("RQ1_*_metrics.csv"):
        name = metrics_file.stem.replace("_metrics", "")
        parts = name.split("_")
        
        # Parse: RQ1_D{size}_{config}_{seed}
        dataset_size = int(parts[1][1:])  # D500 -> 500
        config_type = parts[2]  # base o sqrt
        seed = int(parts[3].replace("seed", ""))
        
        try:
            df = pd.read_csv(metrics_file)
            
            # F1 final (época 10, test)
            final_rows = df[(df['epoch'] == 10) & (df['split'] == 'test')]
            
            if len(final_rows) == 0:
                skipped.append(f"{name} (incompleto: max epoch {df['epoch'].max()})")
                continue
                
            final_row = final_rows.iloc[-1]
            
            data.append({
                'name': name,
                'dataset_size': dataset_size,
                'config': config_type,
                'seed': seed,
                'f1_final': final_row['f1_overall'],
                'precision': final_row['precision_overall'],
                'recall': final_row['recall_overall'],
                'f1_PER': final_row['f1_PER'],
                'f1_LOC': final_row['f1_LOC'],
                'f1_ORG': final_row['f1_ORG']
            })
        except Exception as e:
            skipped.append(f"{name} (error: {e})")
    
    if skipped:
        print(f"⚠️  Archivos saltados: {len(skipped)}")
        for s in skipped:
            print(f"   - {s}")
        print()
    
    return pd.DataFrame(data)

def analyze_preliminary(df):
    """Análisis estadístico preliminar"""
    
    print("=" * 60)
    print("ANÁLISIS PRELIMINAR RQ1 - Dataset Size Scaling")
    print("=" * 60)
    
    # 1. Resumen de datos
    print(f"\n📊 DATOS CARGADOS: {len(df)} experimentos completos")
    print(f"   Datasets: {sorted(df['dataset_size'].unique())}")
    print(f"   Configs: {df['config'].unique().tolist()}")
    print(f"   Seeds: {sorted(df['seed'].unique())}")
    
    # 2. Tabla comparativa por dataset size
    print("\n" + "=" * 60)
    print("📈 F1 TEST FINAL POR DATASET SIZE Y CONFIGURACIÓN")
    print("=" * 60)
    
    print("\n--- Media F1 ---")
    means = df.pivot_table(values='f1_final', index='dataset_size', columns='config', aggfunc='mean')
    print(means.round(4).to_string())
    
    print("\n--- Desviación Estándar ---")
    stds = df.pivot_table(values='f1_final', index='dataset_size', columns='config', aggfunc='std')
    print(stds.round(4).to_string())
    
    # 3. Diferencia sqrt vs base
    print("\n" + "=" * 60)
    print("📊 DIFERENCIA: sqrt - base (positivo = sqrt mejor)")
    print("=" * 60)
    
    comparison = []
    for size in sorted(df['dataset_size'].unique()):
        base_f1 = df[(df['dataset_size'] == size) & (df['config'] == 'base')]['f1_final']
        sqrt_f1 = df[(df['dataset_size'] == size) & (df['config'] == 'sqrt')]['f1_final']
        
        if len(base_f1) > 0 and len(sqrt_f1) > 0:
            diff = sqrt_f1.mean() - base_f1.mean()
            comparison.append({
                'dataset_size': size,
                'base_mean': round(base_f1.mean(), 4),
                'sqrt_mean': round(sqrt_f1.mean(), 4),
                'diff': round(diff, 4),
                'pct_diff': round((diff / base_f1.mean()) * 100, 2) if base_f1.mean() > 0 else 0,
                'winner': '√' if diff > 0 else 'base'
            })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # 4. Resumen por entity type
    print("\n" + "=" * 60)
    print("📋 F1 POR ENTITY TYPE (promedio global)")
    print("=" * 60)
    
    for config in ['base', 'sqrt']:
        subset = df[df['config'] == config]
        print(f"\n{config.upper()}:")
        print(f"  PER: {subset['f1_PER'].mean():.4f} ± {subset['f1_PER'].std():.4f}")
        print(f"  LOC: {subset['f1_LOC'].mean():.4f} ± {subset['f1_LOC'].std():.4f}")
        print(f"  ORG: {subset['f1_ORG'].mean():.4f} ± {subset['f1_ORG'].std():.4f}")
    
    # 5. Tendencia: ¿sqrt mejora con datasets más grandes?
    print("\n" + "=" * 60)
    print("📈 TENDENCIA: Ventaja de sqrt vs dataset size")
    print("=" * 60)
    
    if len(comp_df) > 2:
        corr = np.corrcoef(comp_df['dataset_size'], comp_df['diff'])[0, 1]
        print(f"\nCorrelación (dataset_size vs diff): {corr:.4f}")
        if corr > 0.3:
            print("→ Tendencia POSITIVA: sqrt mejora más en datasets grandes")
        elif corr < -0.3:
            print("→ Tendencia NEGATIVA: sqrt mejora más en datasets pequeños")
        else:
            print("→ Sin tendencia clara")
    
    # 6. Guardar resultados
    output = {
        'summary': comp_df.to_dict('records'),
        'total_experiments': len(df),
        'datasets_analyzed': sorted(df['dataset_size'].unique().tolist())
    }
    
    with open('analysis_rq1_preliminary.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Resultados guardados en analysis_rq1_preliminary.json")
    
    return df, comp_df

if __name__ == "__main__":
    df = load_metrics()
    if len(df) > 0:
        df, comp = analyze_preliminary(df)
    else:
        print("❌ No se encontraron archivos de métricas completos")
