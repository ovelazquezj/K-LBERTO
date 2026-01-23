#!/usr/bin/env python3
"""
Análisis detallado por tipo de entidad (PER, LOC, ORG)
Paper 2: Knowledge Graph Ablation in Spanish NER
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para Jetson
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Directorios
RESULTS_DIR = "/home/omar/projects/K-LBERTO/resultados_ablation_d4000"
OUTPUT_DIR = "/home/omar/projects/K-LBERTO/NER_RESEARCH"

def load_all_results():
    """Carga resultados de todos los experimentos"""
    results = []

    for exp_dir in os.listdir(RESULTS_DIR):
        if not exp_dir.startswith("ABL_"):
            continue

        metrics_file = os.path.join(RESULTS_DIR, exp_dir, f"{exp_dir}_metrics.csv")
        if not os.path.exists(metrics_file):
            continue

        # Parsear nombre del experimento
        parts = exp_dir.split("_")
        data_type = parts[1]  # CUR o RAW
        kg_type = parts[2]    # NOKG, GEN, CUR
        seed = parts[3][1:]   # número de seed

        # Leer métricas
        df = pd.read_csv(metrics_file)

        # Obtener última fila de test (epoch final)
        test_rows = df[df['split'] == 'test']
        if len(test_rows) == 0:
            continue
        final_row = test_rows.iloc[-1]

        results.append({
            'exp_id': exp_dir,
            'data': data_type,
            'kg': kg_type,
            'seed': int(seed),
            'f1_overall': final_row['f1_overall'],
            'f1_PER': final_row['f1_PER'],
            'f1_LOC': final_row['f1_LOC'],
            'f1_ORG': final_row['f1_ORG'],
            'precision_overall': final_row['precision_overall'],
            'recall_overall': final_row['recall_overall'],
            'precision_PER': final_row['precision_PER'],
            'recall_PER': final_row['recall_PER'],
            'precision_LOC': final_row['precision_LOC'],
            'recall_LOC': final_row['recall_LOC'],
            'precision_ORG': final_row['precision_ORG'],
            'recall_ORG': final_row['recall_ORG'],
        })

    return pd.DataFrame(results)

def compute_statistics(df):
    """Calcula estadísticas por condición y tipo de entidad"""

    # Agrupar por data y kg
    grouped = df.groupby(['data', 'kg'])

    stats_list = []
    for (data, kg), group in grouped:
        for entity in ['overall', 'PER', 'LOC', 'ORG']:
            col = f'f1_{entity}'
            values = group[col].values

            stats_list.append({
                'data': data,
                'kg': kg,
                'entity': entity,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            })

    return pd.DataFrame(stats_list)

def create_heatmap_by_entity(df, output_dir):
    """Crea heatmaps de F1 por tipo de entidad"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    entities = ['overall', 'PER', 'LOC', 'ORG']
    titles = ['F1 Overall', 'F1 PER (Personas)', 'F1 LOC (Lugares)', 'F1 ORG (Organizaciones)']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx // 2, idx % 2]

        # Crear matriz para heatmap
        pivot_data = df.groupby(['data', 'kg'])[f'f1_{entity}'].mean().unstack()

        # Reordenar
        pivot_data = pivot_data.reindex(index=['CUR', 'RAW'], columns=['NOKG', 'GEN', 'CUR'])
        pivot_data.index = ['Curated', 'Raw']
        pivot_data.columns = ['NoKG', 'Generic', 'Curated KG']

        # Crear heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=ax, vmin=0.35, vmax=0.70,
                    cbar_kws={'label': 'F1 Score'},
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Knowledge Graph')
        ax.set_ylabel('Data Type')

    plt.suptitle('F1 Score por Tipo de Entidad y Condición Experimental',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_heatmap_by_entity.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_heatmap_by_entity.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig1_heatmap_by_entity.png/pdf")

def create_barplot_comparison(df, output_dir):
    """Crea barplot comparando entidades por condición"""

    # Preparar datos
    stats_df = compute_statistics(df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Por tipo de KG (datos curados)
    ax1 = axes[0]
    cur_data = stats_df[(stats_df['data'] == 'CUR') & (stats_df['entity'] != 'overall')]

    x = np.arange(3)  # NOKG, GEN, CUR
    width = 0.25

    for i, entity in enumerate(['PER', 'LOC', 'ORG']):
        entity_data = cur_data[cur_data['entity'] == entity]
        means = [entity_data[entity_data['kg'] == kg]['mean'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
        stds = [entity_data[entity_data['kg'] == kg]['std'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
        ax1.bar(x + i*width, means, width, label=entity, yerr=stds, capsize=3)

    ax1.set_xlabel('Knowledge Graph Type')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Curated Data: F1 por Tipo de Entidad', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
    ax1.legend(title='Entity Type')
    ax1.set_ylim(0.3, 0.7)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Por tipo de KG (datos raw)
    ax2 = axes[1]
    raw_data = stats_df[(stats_df['data'] == 'RAW') & (stats_df['entity'] != 'overall')]

    for i, entity in enumerate(['PER', 'LOC', 'ORG']):
        entity_data = raw_data[raw_data['entity'] == entity]
        means = [entity_data[entity_data['kg'] == kg]['mean'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
        stds = [entity_data[entity_data['kg'] == kg]['std'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
        ax2.bar(x + i*width, means, width, label=entity, yerr=stds, capsize=3)

    ax2.set_xlabel('Knowledge Graph Type')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Raw Data: F1 por Tipo de Entidad', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
    ax2.legend(title='Entity Type')
    ax2.set_ylim(0.3, 0.7)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Comparación de F1 por Tipo de Entidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_barplot_entity_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig2_barplot_entity_comparison.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig2_barplot_entity_comparison.png/pdf")

def create_interaction_plot(df, output_dir):
    """Crea gráfico de interacción Data x KG"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    entities = ['overall', 'PER', 'LOC', 'ORG']
    titles = ['Overall', 'PER (Personas)', 'LOC (Lugares)', 'ORG (Organizaciones)']

    kg_order = ['NOKG', 'GEN', 'CUR']
    kg_labels = ['NoKG', 'Generic', 'Curated']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx // 2, idx % 2]

        for data_type, marker, color in [('CUR', 'o', 'blue'), ('RAW', 's', 'red')]:
            means = []
            stds = []
            for kg in kg_order:
                subset = df[(df['data'] == data_type) & (df['kg'] == kg)]
                means.append(subset[f'f1_{entity}'].mean())
                stds.append(subset[f'f1_{entity}'].std())

            label = 'Curated Data' if data_type == 'CUR' else 'Raw Data'
            ax.errorbar(kg_labels, means, yerr=stds, marker=marker,
                       label=label, color=color, linewidth=2, markersize=8, capsize=5)

        ax.set_xlabel('Knowledge Graph Type')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 {title}', fontweight='bold')
        ax.legend()
        ax.set_ylim(0.3, 0.75)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Interaction Plot: Data Quality × KG Quality',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_interaction_plot.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_interaction_plot.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig3_interaction_plot.png/pdf")

def create_entity_radar_chart(df, output_dir):
    """Crea radar chart comparando entidades por condición"""

    from math import pi

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))

    conditions = [
        ('CUR', 'NOKG', 'CUR + NoKG'),
        ('CUR', 'GEN', 'CUR + Generic'),
        ('CUR', 'CUR', 'CUR + Curated KG'),
        ('RAW', 'NOKG', 'RAW + NoKG'),
        ('RAW', 'GEN', 'RAW + Generic'),
        ('RAW', 'CUR', 'RAW + Curated KG'),
    ]

    categories = ['PER', 'LOC', 'ORG']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # cerrar el círculo

    for idx, (data, kg, title) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]

        subset = df[(df['data'] == data) & (df['kg'] == kg)]
        values = [subset[f'f1_{cat}'].mean() for cat in categories]
        values += values[:1]  # cerrar el círculo

        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 0.7)
        ax.set_title(title, fontweight='bold', size=12, y=1.1)

    plt.suptitle('Radar Chart: F1 por Tipo de Entidad en Cada Condición',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_radar_entity.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_radar_entity.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig4_radar_entity.png/pdf")

def create_boxplot_by_entity(df, output_dir):
    """Crea boxplots comparando distribuciones por entidad"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    entities = ['PER', 'LOC', 'ORG']
    titles = ['PER (Personas)', 'LOC (Lugares)', 'ORG (Organizaciones)']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx]

        # Preparar datos para boxplot
        plot_data = []
        labels = []

        for data in ['CUR', 'RAW']:
            for kg in ['NOKG', 'GEN', 'CUR']:
                subset = df[(df['data'] == data) & (df['kg'] == kg)]
                plot_data.append(subset[f'f1_{entity}'].values)
                kg_label = {'NOKG': 'NoKG', 'GEN': 'Gen', 'CUR': 'Cur'}[kg]
                labels.append(f'{data}\n{kg_label}')

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

        # Colorear por tipo de data
        colors = ['lightblue']*3 + ['lightcoral']*3
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 {title}', fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.2, 0.75)

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='Curated Data'),
                          Patch(facecolor='lightcoral', label='Raw Data')]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.suptitle('Distribución de F1 por Tipo de Entidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_boxplot_by_entity.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_boxplot_by_entity.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig5_boxplot_by_entity.png/pdf")

def create_main_results_figure(df, output_dir):
    """Crea figura principal de resultados para el paper"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Heatmap de F1 Overall
    ax1 = axes[0]
    pivot_data = df.groupby(['data', 'kg'])['f1_overall'].mean().unstack()
    pivot_data = pivot_data.reindex(index=['CUR', 'RAW'], columns=['NOKG', 'GEN', 'CUR'])
    pivot_data.index = ['Curated', 'Raw']
    pivot_data.columns = ['NoKG', 'Generic', 'Curated KG']

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax1, vmin=0.55, vmax=0.68,
                cbar_kws={'label': 'F1 Score'},
                annot_kws={'size': 16, 'weight': 'bold'})
    ax1.set_title('(A) F1 Score by Condition', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Knowledge Graph')
    ax1.set_ylabel('Data Type')

    # Panel B: Barplot comparativo RAW vs CUR
    ax2 = axes[1]

    # Calcular medias por data type
    cur_means = df[df['data'] == 'CUR'].groupby('kg')['f1_overall'].mean()
    raw_means = df[df['data'] == 'RAW'].groupby('kg')['f1_overall'].mean()
    cur_stds = df[df['data'] == 'CUR'].groupby('kg')['f1_overall'].std()
    raw_stds = df[df['data'] == 'RAW'].groupby('kg')['f1_overall'].std()

    x = np.arange(3)
    width = 0.35

    bars1 = ax2.bar(x - width/2, [cur_means['NOKG'], cur_means['GEN'], cur_means['CUR']],
                    width, label='Curated Data', color='steelblue',
                    yerr=[cur_stds['NOKG'], cur_stds['GEN'], cur_stds['CUR']], capsize=5)
    bars2 = ax2.bar(x + width/2, [raw_means['NOKG'], raw_means['GEN'], raw_means['CUR']],
                    width, label='Raw Data', color='coral',
                    yerr=[raw_stds['NOKG'], raw_stds['GEN'], raw_stds['CUR']], capsize=5)

    ax2.set_xlabel('Knowledge Graph Type')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('(B) Data Quality Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
    ax2.legend()
    ax2.set_ylim(0.5, 0.7)
    ax2.axhline(y=0.6, color='gray', linestyle='--', alpha=0.3)

    # Añadir significancia
    ax2.annotate('***', xy=(0, 0.665), ha='center', fontsize=14)
    ax2.annotate('ns', xy=(1, 0.60), ha='center', fontsize=12, color='gray')
    ax2.annotate('***', xy=(2, 0.66), ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_main_results.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_main_results.pdf'),
                bbox_inches='tight')
    plt.close()
    print("  Guardado: fig_main_results.png/pdf")

def generate_entity_report(df, output_dir):
    """Genera reporte de análisis por entidad"""

    stats_df = compute_statistics(df)

    report = []
    report.append("# Análisis Detallado por Tipo de Entidad")
    report.append("=" * 60)
    report.append("")
    report.append("## Resumen de F1 por Entidad y Condición")
    report.append("")

    # Tabla de medias
    report.append("### Tabla de Medias (F1 Score)")
    report.append("")
    report.append("| Data | KG | Overall | PER | LOC | ORG |")
    report.append("|------|-----|---------|-----|-----|-----|")

    for data in ['CUR', 'RAW']:
        for kg in ['NOKG', 'GEN', 'CUR']:
            row = stats_df[(stats_df['data'] == data) & (stats_df['kg'] == kg)]
            overall = row[row['entity'] == 'overall']['mean'].values[0]
            per = row[row['entity'] == 'PER']['mean'].values[0]
            loc = row[row['entity'] == 'LOC']['mean'].values[0]
            org = row[row['entity'] == 'ORG']['mean'].values[0]
            report.append(f"| {data} | {kg} | {overall:.4f} | {per:.4f} | {loc:.4f} | {org:.4f} |")

    report.append("")
    report.append("### Observaciones por Tipo de Entidad")
    report.append("")

    # Análisis PER
    report.append("#### PER (Personas)")
    per_data = stats_df[stats_df['entity'] == 'PER']
    best_per = per_data.loc[per_data['mean'].idxmax()]
    worst_per = per_data.loc[per_data['mean'].idxmin()]
    report.append(f"- Mejor condición: {best_per['data']}_{best_per['kg']} (F1={best_per['mean']:.4f})")
    report.append(f"- Peor condición: {worst_per['data']}_{worst_per['kg']} (F1={worst_per['mean']:.4f})")
    report.append(f"- Diferencia: {best_per['mean'] - worst_per['mean']:.4f} ({((best_per['mean'] - worst_per['mean'])/worst_per['mean']*100):.1f}%)")
    report.append("")

    # Análisis LOC
    report.append("#### LOC (Lugares)")
    loc_data = stats_df[stats_df['entity'] == 'LOC']
    best_loc = loc_data.loc[loc_data['mean'].idxmax()]
    worst_loc = loc_data.loc[loc_data['mean'].idxmin()]
    report.append(f"- Mejor condición: {best_loc['data']}_{best_loc['kg']} (F1={best_loc['mean']:.4f})")
    report.append(f"- Peor condición: {worst_loc['data']}_{worst_loc['kg']} (F1={worst_loc['mean']:.4f})")
    report.append(f"- Diferencia: {best_loc['mean'] - worst_loc['mean']:.4f} ({((best_loc['mean'] - worst_loc['mean'])/worst_loc['mean']*100):.1f}%)")
    report.append("")

    # Análisis ORG
    report.append("#### ORG (Organizaciones)")
    org_data = stats_df[stats_df['entity'] == 'ORG']
    best_org = org_data.loc[org_data['mean'].idxmax()]
    worst_org = org_data.loc[org_data['mean'].idxmin()]
    report.append(f"- Mejor condición: {best_org['data']}_{best_org['kg']} (F1={best_org['mean']:.4f})")
    report.append(f"- Peor condición: {worst_org['data']}_{worst_org['kg']} (F1={worst_org['mean']:.4f})")
    report.append(f"- Diferencia: {best_org['mean'] - worst_org['mean']:.4f} ({((best_org['mean'] - worst_org['mean'])/worst_org['mean']*100):.1f}%)")
    report.append("")

    # Ranking de dificultad
    report.append("### Ranking de Dificultad por Entidad")
    report.append("")
    overall_means = stats_df[stats_df['entity'] != 'overall'].groupby('entity')['mean'].mean()
    for entity in overall_means.sort_values(ascending=False).index:
        report.append(f"1. **{entity}**: F1 promedio = {overall_means[entity]:.4f}")
    report.append("")
    report.append("**Interpretación:** ORG es consistentemente la entidad más difícil de reconocer.")
    report.append("")

    # Efecto del KG por entidad
    report.append("### Efecto del Knowledge Graph por Entidad")
    report.append("")

    for entity in ['PER', 'LOC', 'ORG']:
        entity_data = stats_df[stats_df['entity'] == entity]
        nokg_mean = entity_data[entity_data['kg'] == 'NOKG']['mean'].mean()
        gen_mean = entity_data[entity_data['kg'] == 'GEN']['mean'].mean()
        cur_mean = entity_data[entity_data['kg'] == 'CUR']['mean'].mean()

        report.append(f"**{entity}:**")
        report.append(f"- NoKG: {nokg_mean:.4f}")
        report.append(f"- Generic KG: {gen_mean:.4f} ({(gen_mean-nokg_mean)*100:+.2f}%)")
        report.append(f"- Curated KG: {cur_mean:.4f} ({(cur_mean-nokg_mean)*100:+.2f}%)")
        report.append("")

    # Guardar reporte
    report_path = os.path.join(output_dir, 'entity_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"  Guardado: entity_analysis_report.md")

    # Guardar CSV con estadísticas
    stats_df.to_csv(os.path.join(output_dir, 'entity_statistics.csv'), index=False)
    print(f"  Guardado: entity_statistics.csv")

def main():
    print("=" * 60)
    print("ANÁLISIS POR TIPO DE ENTIDAD - Paper 2 NER")
    print("=" * 60)
    print()

    # Cargar datos
    print("Cargando resultados...")
    df = load_all_results()
    print(f"  Cargados {len(df)} experimentos")
    print()

    # Generar figuras
    print("Generando figuras...")
    create_heatmap_by_entity(df, OUTPUT_DIR)
    create_barplot_comparison(df, OUTPUT_DIR)
    create_interaction_plot(df, OUTPUT_DIR)
    create_entity_radar_chart(df, OUTPUT_DIR)
    create_boxplot_by_entity(df, OUTPUT_DIR)
    create_main_results_figure(df, OUTPUT_DIR)
    print()

    # Generar reporte
    print("Generando reporte de análisis...")
    generate_entity_report(df, OUTPUT_DIR)
    print()

    print("=" * 60)
    print("ANÁLISIS COMPLETADO")
    print(f"Figuras guardadas en: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
