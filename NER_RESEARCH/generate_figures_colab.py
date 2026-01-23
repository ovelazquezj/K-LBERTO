#!/usr/bin/env python3
"""
Script para generar figuras en Google Colab o laptop
Usar después de copiar los CSVs generados por analyze_entity_data.py

Instrucciones:
1. Subir entity_statistics.csv y entity_all_results.csv a Colab
2. Ejecutar este script
3. Descargar las figuras generadas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Directorio de salida (cambiar según necesidad)
OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Carga los CSVs generados"""
    stats_df = pd.read_csv('entity_statistics.csv')
    results_df = pd.read_csv('entity_all_results.csv')
    return stats_df, results_df

def fig1_heatmap_by_entity(stats_df):
    """Heatmap de F1 por tipo de entidad"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    entities = ['overall', 'PER', 'LOC', 'ORG']
    titles = ['F1 Overall', 'F1 PER (Personas)', 'F1 LOC (Lugares)', 'F1 ORG (Organizaciones)']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx // 2, idx % 2]

        # Crear matriz
        entity_data = stats_df[stats_df['entity'] == entity]
        matrix = np.zeros((2, 3))

        for i, data in enumerate(['CUR', 'RAW']):
            for j, kg in enumerate(['NOKG', 'GEN', 'CUR']):
                val = entity_data[(entity_data['data'] == data) & (entity_data['kg'] == kg)]['mean'].values
                matrix[i, j] = val[0] if len(val) > 0 else 0

        # Heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.35, vmax=0.70)

        # Etiquetas
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Curated', 'Raw'])

        # Valores
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                       fontsize=14, fontweight='bold')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Knowledge Graph')
        ax.set_ylabel('Data Type')

    plt.suptitle('F1 Score por Tipo de Entidad y Condición', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.colorbar(im, ax=axes, shrink=0.6, label='F1 Score')
    plt.savefig(OUTPUT_DIR / 'fig1_heatmap_by_entity.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_heatmap_by_entity.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig1_heatmap_by_entity")

def fig2_barplot_comparison(stats_df):
    """Barplot comparando entidades por condición"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(3)
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for ax_idx, (data_type, title) in enumerate([('CUR', 'Curated Data'), ('RAW', 'Raw Data')]):
        ax = axes[ax_idx]
        data_subset = stats_df[(stats_df['data'] == data_type) & (stats_df['entity'] != 'overall')]

        for i, entity in enumerate(['PER', 'LOC', 'ORG']):
            entity_data = data_subset[data_subset['entity'] == entity]
            means = [entity_data[entity_data['kg'] == kg]['mean'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
            stds = [entity_data[entity_data['kg'] == kg]['std'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
            ax.bar(x + i*width, means, width, label=entity, yerr=stds, capsize=3, color=colors[i])

        ax.set_xlabel('Knowledge Graph Type')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{title}: F1 por Tipo de Entidad', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
        ax.legend(title='Entity Type')
        ax.set_ylim(0.25, 0.70)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Comparación de F1 por Tipo de Entidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_barplot_entity_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_barplot_entity_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig2_barplot_entity_comparison")

def fig3_interaction_plot(stats_df):
    """Gráfico de interacción Data x KG"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    entities = ['overall', 'PER', 'LOC', 'ORG']
    titles = ['Overall', 'PER (Personas)', 'LOC (Lugares)', 'ORG (Organizaciones)']
    kg_labels = ['NoKG', 'Generic', 'Curated']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx // 2, idx % 2]
        entity_data = stats_df[stats_df['entity'] == entity]

        for data_type, marker, color in [('CUR', 'o', 'steelblue'), ('RAW', 's', 'coral')]:
            subset = entity_data[entity_data['data'] == data_type]
            means = [subset[subset['kg'] == kg]['mean'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]
            stds = [subset[subset['kg'] == kg]['std'].values[0] for kg in ['NOKG', 'GEN', 'CUR']]

            label = 'Curated Data' if data_type == 'CUR' else 'Raw Data'
            ax.errorbar(kg_labels, means, yerr=stds, marker=marker, label=label,
                       color=color, linewidth=2, markersize=10, capsize=5)

        ax.set_xlabel('Knowledge Graph Type')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 {title}', fontweight='bold')
        ax.legend()
        ax.set_ylim(0.30, 0.75)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Interaction Plot: Data Quality × KG Quality', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_interaction_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_interaction_plot.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig3_interaction_plot")

def fig4_boxplot_by_entity(results_df):
    """Boxplots comparando distribuciones"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    entities = ['PER', 'LOC', 'ORG']
    titles = ['PER (Personas)', 'LOC (Lugares)', 'ORG (Organizaciones)']

    for idx, (entity, title) in enumerate(zip(entities, titles)):
        ax = axes[idx]

        plot_data = []
        labels = []
        colors_list = []

        for data in ['CUR', 'RAW']:
            for kg in ['NOKG', 'GEN', 'CUR']:
                subset = results_df[(results_df['data'] == data) & (results_df['kg'] == kg)]
                plot_data.append(subset[f'f1_{entity}'].values)
                kg_label = {'NOKG': 'NoKG', 'GEN': 'Gen', 'CUR': 'Cur'}[kg]
                labels.append(f'{data}\n{kg_label}')
                colors_list.append('lightblue' if data == 'CUR' else 'lightcoral')

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)

        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 {title}', fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.15, 0.75)

    plt.suptitle('Distribución de F1 por Tipo de Entidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_boxplot_by_entity.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_boxplot_by_entity.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig4_boxplot_by_entity")

def fig5_main_results(stats_df):
    """Figura principal para el paper"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Heatmap Overall
    ax1 = axes[0]
    overall_data = stats_df[stats_df['entity'] == 'overall']

    matrix = np.zeros((2, 3))
    for i, data in enumerate(['CUR', 'RAW']):
        for j, kg in enumerate(['NOKG', 'GEN', 'CUR']):
            val = overall_data[(overall_data['data'] == data) & (overall_data['kg'] == kg)]['mean'].values
            matrix[i, j] = val[0] if len(val) > 0 else 0

    im = ax1.imshow(matrix, cmap='RdYlGn', vmin=0.55, vmax=0.68)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Curated', 'Raw'])

    for i in range(2):
        for j in range(3):
            ax1.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='black')

    ax1.set_title('(A) F1 Score by Condition', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Knowledge Graph')
    ax1.set_ylabel('Data Type')
    plt.colorbar(im, ax=ax1, shrink=0.8, label='F1 Score')

    # Panel B: Barplot RAW vs CUR
    ax2 = axes[1]
    x = np.arange(3)
    width = 0.35

    cur_means = [overall_data[(overall_data['data'] == 'CUR') & (overall_data['kg'] == kg)]['mean'].values[0]
                 for kg in ['NOKG', 'GEN', 'CUR']]
    raw_means = [overall_data[(overall_data['data'] == 'RAW') & (overall_data['kg'] == kg)]['mean'].values[0]
                 for kg in ['NOKG', 'GEN', 'CUR']]
    cur_stds = [overall_data[(overall_data['data'] == 'CUR') & (overall_data['kg'] == kg)]['std'].values[0]
                for kg in ['NOKG', 'GEN', 'CUR']]
    raw_stds = [overall_data[(overall_data['data'] == 'RAW') & (overall_data['kg'] == kg)]['std'].values[0]
                for kg in ['NOKG', 'GEN', 'CUR']]

    ax2.bar(x - width/2, cur_means, width, label='Curated Data', color='steelblue',
            yerr=cur_stds, capsize=5)
    ax2.bar(x + width/2, raw_means, width, label='Raw Data', color='coral',
            yerr=raw_stds, capsize=5)

    ax2.set_xlabel('Knowledge Graph Type')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('(B) Data Quality Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['NoKG', 'Generic', 'Curated KG'])
    ax2.legend()
    ax2.set_ylim(0.50, 0.70)

    # Significancia
    ax2.annotate('***', xy=(0, 0.665), ha='center', fontsize=14)
    ax2.annotate('ns', xy=(1, 0.60), ha='center', fontsize=12, color='gray')
    ax2.annotate('***', xy=(2, 0.665), ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_main_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_main_results.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig_main_results")

def fig6_entity_difficulty(stats_df):
    """Gráfico de dificultad por tipo de entidad"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calcular promedios por entidad
    entity_means = stats_df[stats_df['entity'] != 'overall'].groupby('entity')['mean'].mean()
    entity_stds = stats_df[stats_df['entity'] != 'overall'].groupby('entity')['std'].mean()

    entities = ['PER', 'LOC', 'ORG']
    means = [entity_means[e] for e in entities]
    stds = [entity_stds[e] for e in entities]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(entities, means, yerr=stds, capsize=10, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title('Entity Recognition Difficulty\n(Lower = Harder)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.65)

    # Valores sobre barras
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{mean_val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_entity_difficulty.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_entity_difficulty.pdf', bbox_inches='tight')
    plt.close()
    print("Guardado: fig6_entity_difficulty")

def main():
    print("=" * 60)
    print("GENERACIÓN DE FIGURAS - Paper 2 NER")
    print("=" * 60)
    print()

    # Cargar datos
    print("Cargando datos...")
    stats_df, results_df = load_data()
    print(f"  Stats: {len(stats_df)} filas")
    print(f"  Results: {len(results_df)} experimentos")
    print()

    # Generar figuras
    print("Generando figuras...")
    fig1_heatmap_by_entity(stats_df)
    fig2_barplot_comparison(stats_df)
    fig3_interaction_plot(stats_df)
    fig4_boxplot_by_entity(results_df)
    fig5_main_results(stats_df)
    fig6_entity_difficulty(stats_df)
    print()

    print("=" * 60)
    print(f"FIGURAS GUARDADAS EN: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
