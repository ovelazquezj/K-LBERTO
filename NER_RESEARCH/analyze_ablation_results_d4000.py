#!/usr/bin/env python3
"""
Análisis de resultados - Ablation Study KG Quality vs Data Curation
Compatible con Jetson (sin scipy/matplotlib)

Uso:
    python3 analyze_ablation_results.py [--partial]

Salidas (5 archivos):
    - all_results.csv: Todos los resultados individuales
    - summary_stats.csv: Estadísticas por condición
    - interaction_data.csv: Datos para gráfico de interacción
    - main_effects_data.csv: Datos para gráficos de efectos principales
    - heatmap_matrix.csv: Matriz para heatmap
"""

import argparse
from pathlib import Path
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Configuración
BASE_DIR = Path.home() / "projects" / "K-LBERTO"
RESULTS_DIR = BASE_DIR / "resultados_ablation_d4000"
OUTPUT_DIR = BASE_DIR / "ablation_analysis_d4000"

DATA_LEVELS = {"CUR": "Curated", "RAW": "Raw"}
KG_LEVELS = {"NOKG": "None", "GEN": "Generic (3.4M)", "CUR": "Curated (15k)"}
KG_LABELS_ORDER = ["None", "Generic (3.4M)", "Curated (15k)"]


def parse_experiment_id(exp_id: str) -> dict:
    parts = exp_id.split("_")
    return {
        "exp_id": exp_id,
        "data": parts[1],
        "kg": parts[2],
        "seed": int(parts[3].replace("s", ""))
    }


def load_results() -> pd.DataFrame:
    results = []

    if not RESULTS_DIR.exists():
        print(f"❌ Directorio no encontrado: {RESULTS_DIR}")
        return pd.DataFrame()

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("ABL_"):
            continue

        # Buscar archivo *_metrics.csv
        metrics_files = list(exp_dir.glob("*_metrics.csv"))
        
        if not metrics_files:
            continue
        
        metrics_file = metrics_files[0]

        try:
            df_metrics = pd.read_csv(metrics_file)
            
            if df_metrics.empty:
                continue
            
            # Buscar mejor F1 en test
            test_rows = df_metrics[df_metrics['split'] == 'test']
            if test_rows.empty:
                test_rows = df_metrics
            
            best_row = test_rows.loc[test_rows['f1_overall'].idxmax()]
            exp_info = parse_experiment_id(exp_dir.name)

            results.append({
                **exp_info,
                "data_label": DATA_LEVELS.get(exp_info["data"], exp_info["data"]),
                "kg_label": KG_LEVELS.get(exp_info["kg"], exp_info["kg"]),
                "f1": float(best_row['f1_overall']),
                "precision": float(best_row['precision_overall']),
                "recall": float(best_row['recall_overall']),
                "epoch": int(best_row['epoch'])
            })
            print(f"  ✅ {exp_dir.name}: F1={best_row['f1_overall']:.4f}")

        except Exception as e:
            print(f"  ❌ Error {exp_dir.name}: {e}")

    return pd.DataFrame(results)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["data", "kg", "data_label", "kg_label"]).agg({
        "f1": ["mean", "std", "count", "min", "max"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"]
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def generate_interaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Genera datos para gráfico de interacción Data × KG"""
    interaction = df.groupby(['data', 'data_label', 'kg', 'kg_label'])['f1'].agg(
        ['mean', 'std', 'count']
    ).round(4).reset_index()
    interaction.columns = ['data', 'data_label', 'kg', 'kg_label', 'f1_mean', 'f1_std', 'n']
    return interaction


def generate_main_effects_data(df: pd.DataFrame) -> pd.DataFrame:
    """Genera datos para gráficos de efectos principales"""
    # Efecto de Data
    data_effect = df.groupby(['data', 'data_label'])['f1'].agg(['mean', 'std', 'count']).round(4).reset_index()
    data_effect.columns = ['factor_code', 'factor_label', 'f1_mean', 'f1_std', 'n']
    data_effect['factor'] = 'Data'
    
    # Efecto de KG
    kg_effect = df.groupby(['kg', 'kg_label'])['f1'].agg(['mean', 'std', 'count']).round(4).reset_index()
    kg_effect.columns = ['factor_code', 'factor_label', 'f1_mean', 'f1_std', 'n']
    kg_effect['factor'] = 'KG'
    
    return pd.concat([data_effect, kg_effect], ignore_index=True)


def generate_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Genera matriz para heatmap (filas=Data, columnas=KG)"""
    pivot = df.groupby(['data_label', 'kg_label'])['f1'].mean().unstack()
    # Ordenar columnas
    ordered_cols = [k for k in KG_LABELS_ORDER if k in pivot.columns]
    pivot = pivot[ordered_cols]
    return pivot.round(4)


def t_test_manual(group1, group2):
    """T-test independiente manual"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return float('nan'), float('nan'), float('nan')
    
    t_stat = (mean1 - mean2) / se
    pooled_std = math.sqrt((var1 + var2) / 2)
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Aproximación p-value
    z = abs(t_stat)
    if z > 6:
        p_value = 0.0001
    else:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    
    return t_stat, p_value, cohens_d


def f_cdf_approx(f_stat, df1, df2):
    """Aproximación p-value para F-test"""
    if df2 <= 0 or f_stat <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    if x >= 1:
        return 1.0
    if x <= 0:
        return 0.0001
    p = x ** (df2 / 2)
    return max(0.0001, min(0.9999, p))


def two_way_anova_manual(df: pd.DataFrame) -> dict:
    """ANOVA factorial 2×3 manual"""
    print("\n📊 ANOVA Factorial 2×3 (Data × KG)")
    print("=" * 60)

    grand_mean = df['f1'].mean()
    n_total = len(df)
    ss_total = ((df['f1'] - grand_mean) ** 2).sum()

    data_means = df.groupby('data')['f1'].mean()
    data_counts = df.groupby('data')['f1'].count()
    ss_data = sum(data_counts[l] * (data_means[l] - grand_mean) ** 2 for l in data_means.index)
    df_data = len(data_means) - 1

    kg_means = df.groupby('kg')['f1'].mean()
    kg_counts = df.groupby('kg')['f1'].count()
    ss_kg = sum(kg_counts[l] * (kg_means[l] - grand_mean) ** 2 for l in kg_means.index)
    df_kg = len(kg_means) - 1

    cell_means = df.groupby(['data', 'kg'])['f1'].mean()
    cell_counts = df.groupby(['data', 'kg'])['f1'].count()
    ss_cells = sum(cell_counts[c] * (cell_means[c] - grand_mean) ** 2 for c in cell_means.index)
    ss_interaction = ss_cells - ss_data - ss_kg
    df_interaction = df_data * df_kg

    ss_error = ss_total - ss_data - ss_kg - ss_interaction
    df_error = n_total - (len(data_means) * len(kg_means))

    ms_data = ss_data / df_data if df_data > 0 else 0
    ms_kg = ss_kg / df_kg if df_kg > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0.0001

    f_data = ms_data / ms_error if ms_error > 0 else 0
    f_kg = ms_kg / ms_error if ms_error > 0 else 0
    f_interaction = ms_interaction / ms_error if ms_error > 0 else 0

    p_data = f_cdf_approx(f_data, df_data, df_error)
    p_kg = f_cdf_approx(f_kg, df_kg, df_error)
    p_interaction = f_cdf_approx(f_interaction, df_interaction, df_error)

    eta2_data = ss_data / ss_total if ss_total > 0 else 0
    eta2_kg = ss_kg / ss_total if ss_total > 0 else 0
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    anova_table = pd.DataFrame({
        'Source': ['Data', 'KG', 'Data × KG', 'Error', 'Total'],
        'SS': [round(ss_data, 4), round(ss_kg, 4), round(ss_interaction, 4), round(ss_error, 4), round(ss_total, 4)],
        'df': [df_data, df_kg, df_interaction, df_error, n_total - 1],
        'MS': [round(ms_data, 4), round(ms_kg, 4), round(ms_interaction, 4), round(ms_error, 4), ''],
        'F': [round(f_data, 2), round(f_kg, 2), round(f_interaction, 2), '', ''],
        'p': [round(p_data, 4), round(p_kg, 4), round(p_interaction, 4), '', ''],
        'eta2': [round(eta2_data, 4), round(eta2_kg, 4), round(eta2_interaction, 4), '', '']
    })

    print(anova_table.to_string(index=False))

    def effect_label(e):
        return "(pequeño)" if e < 0.06 else "(medio)" if e < 0.14 else "(grande)"

    print(f"\n📏 Effect Sizes: Data η²={eta2_data:.4f} {effect_label(eta2_data)}, "
          f"KG η²={eta2_kg:.4f} {effect_label(eta2_kg)}, "
          f"Interacción η²={eta2_interaction:.4f} {effect_label(eta2_interaction)}")

    return {
        'anova_table': anova_table,
        'effects': {
            'data': {'F': f_data, 'p': p_data, 'eta2': eta2_data, 'significant': p_data < 0.05},
            'kg': {'F': f_kg, 'p': p_kg, 'eta2': eta2_kg, 'significant': p_kg < 0.05},
            'interaction': {'F': f_interaction, 'p': p_interaction, 'eta2': eta2_interaction, 'significant': p_interaction < 0.05}
        }
    }


def run_pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Comparaciones pareadas usando t-tests manuales"""
    print("\n🎯 Comparaciones Pareadas (Research Questions)")
    print("=" * 60)

    comparisons = []

    def get_f1(data, kg):
        return df[(df['data'] == data) & (df['kg'] == kg)]['f1'].values

    def compare(name, rq, d1, k1, d2, k2):
        f1_a, f1_b = get_f1(d1, k1), get_f1(d2, k2)
        if len(f1_a) < 2 or len(f1_b) < 2:
            return None
        t_stat, p_val, cohens_d = t_test_manual(f1_a, f1_b)
        return {
            "RQ": rq, "Comparison": name,
            "Cond_A": f"{d1}_{k1}", "Cond_B": f"{d2}_{k2}",
            "Mean_A": round(np.mean(f1_a), 4), "Mean_B": round(np.mean(f1_b), 4),
            "Diff": round(np.mean(f1_a) - np.mean(f1_b), 4),
            "t": round(t_stat, 3), "p": round(p_val, 4), "Cohen_d": round(cohens_d, 2),
            "Significant": "Yes" if p_val < 0.05 else "No"
        }

    tests = [
        ("CUR: NoKG vs Generic", "RQ1", "CUR", "NOKG", "CUR", "GEN"),
        ("CUR: NoKG vs Curated", "RQ2", "CUR", "NOKG", "CUR", "CUR"),
        ("CUR: Generic vs Curated", "RQ3", "CUR", "GEN", "CUR", "CUR"),
        ("NoKG: Curated vs Raw", "RQ4", "CUR", "NOKG", "RAW", "NOKG"),
        ("GEN: Curated vs Raw", "RQ4b", "CUR", "GEN", "RAW", "GEN"),
        ("CUR_KG: Curated vs Raw", "RQ4c", "CUR", "CUR", "RAW", "CUR"),
        ("RAW: NoKG vs Curated KG", "RQ6", "RAW", "NOKG", "RAW", "CUR"),
    ]

    for args in tests:
        r = compare(*args)
        if r:
            comparisons.append(r)

    df_comp = pd.DataFrame(comparisons)
    if not df_comp.empty:
        print(df_comp.to_string(index=False))

    return df_comp


def print_ascii_table(df: pd.DataFrame):
    print("\n📊 Tabla de Resultados (F1 Score)")
    print("=" * 60)
    print(f"{'Data':<12} | {'None':^12} | {'Generic':^12} | {'Curated':^12}")
    print("-" * 60)

    pivot = df.groupby(['data_label', 'kg_label'])['f1'].mean().unstack()

    for data_label in ['Curated', 'Raw']:
        if data_label in pivot.index:
            row = pivot.loc[data_label]
            vals = [f"{row.get(k, 0):.4f}" if k in row else "---" for k in KG_LABELS_ORDER]
            print(f"{data_label:<12} | {vals[0]:^12} | {vals[1]:^12} | {vals[2]:^12}")
        else:
            print(f"{data_label:<12} | {'---':^12} | {'---':^12} | {'---':^12}")


def generate_report(df, summary, anova_results, comparisons, output_dir):
    report = []
    report.append("# Ablation Study Results: KG Quality vs Data Curation\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Experiments:** {len(df)}/30\n\n")

    report.append("## Summary\n\n| Data | KG | F1 Mean | F1 Std | N |\n|------|-----|---------|--------|---|\n")
    for _, row in summary.iterrows():
        report.append(f"| {row['data_label']} | {row['kg_label']} | {row['f1_mean']:.4f} | {row['f1_std']:.4f} | {int(row['f1_count'])} |\n")

    if anova_results:
        report.append("\n## ANOVA\n\n")
        for factor, vals in anova_results['effects'].items():
            sig = "✓" if vals['significant'] else "✗"
            report.append(f"- **{factor}:** F={vals['F']:.2f}, p={vals['p']:.4f}, η²={vals['eta2']:.4f} {sig}\n")

    if not comparisons.empty:
        report.append("\n## Pairwise Comparisons\n\n| RQ | Comparison | Δ F1 | p | Cohen's d | Sig |\n|---|---|---|---|---|---|\n")
        for _, row in comparisons.iterrows():
            report.append(f"| {row['RQ']} | {row['Comparison']} | {row['Diff']:+.4f} | {row['p']:.4f} | {row['Cohen_d']:.2f} | {row['Significant']} |\n")

    with open(output_dir / "ablation_report.md", 'w') as f:
        f.writelines(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true', help='Analizar resultados parciales')
    args = parser.parse_args()

    print("=" * 60)
    print("ANÁLISIS DE ABLACIÓN: KG Quality vs Data Curation")
    print("=" * 60)

    print("\n📂 Cargando resultados...")
    df = load_results()

    if df.empty:
        print("❌ No hay resultados para analizar")
        return

    print(f"\n   Total: {len(df)} experimentos")
    print(f"   Condiciones: {df.groupby(['data', 'kg']).size().to_dict()}")

    if len(df) < 30 and not args.partial:
        print(f"\n⚠️  Solo {len(df)}/30. Usa --partial para análisis parcial.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generar estadísticas
    summary = compute_summary_stats(df)
    print_ascii_table(df)

    # Guardar 5 archivos CSV
    print("\n📁 Generando archivos CSV...")
    
    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"  ✅ all_results.csv")
    
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)
    print(f"  ✅ summary_stats.csv")
    
    interaction_data = generate_interaction_data(df)
    interaction_data.to_csv(OUTPUT_DIR / "interaction_data.csv", index=False)
    print(f"  ✅ interaction_data.csv")
    
    main_effects_data = generate_main_effects_data(df)
    main_effects_data.to_csv(OUTPUT_DIR / "main_effects_data.csv", index=False)
    print(f"  ✅ main_effects_data.csv")
    
    heatmap_matrix = generate_heatmap_matrix(df)
    heatmap_matrix.to_csv(OUTPUT_DIR / "heatmap_matrix.csv")
    print(f"  ✅ heatmap_matrix.csv")

    # Análisis estadístico
    anova_results = {}
    if len(df) >= 6:
        anova_results = two_way_anova_manual(df)

    comparisons = run_pairwise_comparisons(df)
    if not comparisons.empty:
        comparisons.to_csv(OUTPUT_DIR / "pairwise_comparisons.csv", index=False)
        print(f"\n  ✅ pairwise_comparisons.csv")

    # Reporte
    print("\n📝 Generando reporte...")
    generate_report(df, summary, anova_results, comparisons, OUTPUT_DIR)
    print(f"  ✅ ablation_report.md")

    print("\n" + "=" * 60)
    print("✅ ANÁLISIS COMPLETADO")
    print(f"   Resultados en: {OUTPUT_DIR}")
    print("   CSVs listos para graficar en Colab/laptop")
    print("=" * 60)


if __name__ == "__main__":
    main()
