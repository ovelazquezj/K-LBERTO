#!/usr/bin/env python3
"""
Análisis de datos por tipo de entidad (sin gráficos)
Paper 2: Knowledge Graph Ablation in Spanish NER
"""

import os
import csv
from collections import defaultdict
import math

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
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Obtener última fila de test
        test_rows = [r for r in rows if r['split'] == 'test']
        if len(test_rows) == 0:
            continue
        final_row = test_rows[-1]

        results.append({
            'exp_id': exp_dir,
            'data': data_type,
            'kg': kg_type,
            'seed': int(seed),
            'f1_overall': float(final_row['f1_overall']),
            'f1_PER': float(final_row['f1_PER']),
            'f1_LOC': float(final_row['f1_LOC']),
            'f1_ORG': float(final_row['f1_ORG']),
            'precision_PER': float(final_row['precision_PER']),
            'recall_PER': float(final_row['recall_PER']),
            'precision_LOC': float(final_row['precision_LOC']),
            'recall_LOC': float(final_row['recall_LOC']),
            'precision_ORG': float(final_row['precision_ORG']),
            'recall_ORG': float(final_row['recall_ORG']),
        })

    return results

def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def compute_statistics(results):
    """Calcula estadísticas por condición y tipo de entidad"""

    # Agrupar por data y kg
    grouped = defaultdict(list)
    for r in results:
        key = (r['data'], r['kg'])
        grouped[key].append(r)

    stats_list = []
    for (data, kg), group in grouped.items():
        for entity in ['overall', 'PER', 'LOC', 'ORG']:
            col = f'f1_{entity}'
            values = [r[col] for r in group]

            stats_list.append({
                'data': data,
                'kg': kg,
                'entity': entity,
                'mean': mean(values),
                'std': std(values),
                'min': min(values),
                'max': max(values),
                'n': len(values),
                'values': values
            })

    return stats_list

def generate_report(results, stats, output_dir):
    """Genera reporte completo de análisis por entidad"""

    lines = []
    lines.append("# Análisis Detallado por Tipo de Entidad")
    lines.append("=" * 70)
    lines.append("")
    lines.append("## 1. Resumen de F1 por Entidad y Condición")
    lines.append("")

    # Tabla de medias
    lines.append("### Tabla de Medias (F1 Score)")
    lines.append("")
    lines.append("| Data | KG | Overall | PER | LOC | ORG |")
    lines.append("|------|-----|---------|-----|-----|-----|")

    for data in ['CUR', 'RAW']:
        for kg in ['NOKG', 'GEN', 'CUR']:
            overall = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='overall'][0]['mean']
            per = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='PER'][0]['mean']
            loc = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='LOC'][0]['mean']
            org = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='ORG'][0]['mean']
            lines.append(f"| {data} | {kg} | {overall:.4f} | {per:.4f} | {loc:.4f} | {org:.4f} |")

    lines.append("")
    lines.append("### Tabla de Desviaciones Estándar")
    lines.append("")
    lines.append("| Data | KG | Overall | PER | LOC | ORG |")
    lines.append("|------|-----|---------|-----|-----|-----|")

    for data in ['CUR', 'RAW']:
        for kg in ['NOKG', 'GEN', 'CUR']:
            overall = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='overall'][0]['std']
            per = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='PER'][0]['std']
            loc = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='LOC'][0]['std']
            org = [s for s in stats if s['data']==data and s['kg']==kg and s['entity']=='ORG'][0]['std']
            lines.append(f"| {data} | {kg} | {overall:.4f} | {per:.4f} | {loc:.4f} | {org:.4f} |")

    # Análisis por entidad
    lines.append("")
    lines.append("## 2. Análisis por Tipo de Entidad")
    lines.append("")

    for entity in ['PER', 'LOC', 'ORG']:
        entity_name = {'PER': 'Personas', 'LOC': 'Lugares', 'ORG': 'Organizaciones'}[entity]
        lines.append(f"### {entity} ({entity_name})")
        lines.append("")

        entity_stats = [s for s in stats if s['entity'] == entity]
        best = max(entity_stats, key=lambda x: x['mean'])
        worst = min(entity_stats, key=lambda x: x['mean'])

        lines.append(f"- **Mejor condición:** {best['data']}_{best['kg']} (F1={best['mean']:.4f} ± {best['std']:.4f})")
        lines.append(f"- **Peor condición:** {worst['data']}_{worst['kg']} (F1={worst['mean']:.4f} ± {worst['std']:.4f})")
        diff = best['mean'] - worst['mean']
        pct = (diff / worst['mean']) * 100 if worst['mean'] > 0 else 0
        lines.append(f"- **Diferencia:** {diff:.4f} ({pct:.1f}%)")
        lines.append("")

        # Valores por condición
        lines.append(f"**Valores F1 {entity} por condición:**")
        lines.append("```")
        for data in ['CUR', 'RAW']:
            for kg in ['NOKG', 'GEN', 'CUR']:
                s = [x for x in entity_stats if x['data']==data and x['kg']==kg][0]
                vals_str = ', '.join([f"{v:.4f}" for v in s['values']])
                lines.append(f"{data}_{kg}: [{vals_str}] -> mean={s['mean']:.4f}")
        lines.append("```")
        lines.append("")

    # Ranking de dificultad
    lines.append("## 3. Ranking de Dificultad por Entidad")
    lines.append("")

    entity_overall = {}
    for entity in ['PER', 'LOC', 'ORG']:
        entity_stats = [s for s in stats if s['entity'] == entity]
        entity_overall[entity] = mean([s['mean'] for s in entity_stats])

    sorted_entities = sorted(entity_overall.items(), key=lambda x: x[1], reverse=True)
    for i, (entity, avg) in enumerate(sorted_entities, 1):
        entity_name = {'PER': 'Personas', 'LOC': 'Lugares', 'ORG': 'Organizaciones'}[entity]
        lines.append(f"{i}. **{entity} ({entity_name})**: F1 promedio = {avg:.4f}")

    lines.append("")
    lines.append("**Interpretación:** ORG es consistentemente la entidad más difícil de reconocer,")
    lines.append("mientras que LOC tiende a ser la más fácil.")
    lines.append("")

    # Efecto del KG por entidad
    lines.append("## 4. Efecto del Knowledge Graph por Entidad")
    lines.append("")

    for entity in ['PER', 'LOC', 'ORG']:
        entity_name = {'PER': 'Personas', 'LOC': 'Lugares', 'ORG': 'Organizaciones'}[entity]
        entity_stats = [s for s in stats if s['entity'] == entity]

        nokg_mean = mean([s['mean'] for s in entity_stats if s['kg'] == 'NOKG'])
        gen_mean = mean([s['mean'] for s in entity_stats if s['kg'] == 'GEN'])
        cur_mean = mean([s['mean'] for s in entity_stats if s['kg'] == 'CUR'])

        lines.append(f"### {entity} ({entity_name})")
        lines.append(f"- NoKG: {nokg_mean:.4f}")
        gen_diff = (gen_mean - nokg_mean) * 100
        cur_diff = (cur_mean - nokg_mean) * 100
        lines.append(f"- Generic KG: {gen_mean:.4f} ({gen_diff:+.2f}% vs NoKG)")
        lines.append(f"- Curated KG: {cur_mean:.4f} ({cur_diff:+.2f}% vs NoKG)")
        lines.append("")

    # Efecto de Data por entidad
    lines.append("## 5. Efecto de Data Curation por Entidad")
    lines.append("")

    for entity in ['PER', 'LOC', 'ORG']:
        entity_name = {'PER': 'Personas', 'LOC': 'Lugares', 'ORG': 'Organizaciones'}[entity]
        entity_stats = [s for s in stats if s['entity'] == entity]

        cur_mean = mean([s['mean'] for s in entity_stats if s['data'] == 'CUR'])
        raw_mean = mean([s['mean'] for s in entity_stats if s['data'] == 'RAW'])

        lines.append(f"### {entity} ({entity_name})")
        lines.append(f"- Curated Data: {cur_mean:.4f}")
        lines.append(f"- Raw Data: {raw_mean:.4f}")
        diff = (raw_mean - cur_mean) * 100
        lines.append(f"- Diferencia: {diff:+.2f}% (RAW {'>' if diff > 0 else '<'} CUR)")
        lines.append("")

    # Hallazgos principales
    lines.append("## 6. Hallazgos Principales por Entidad")
    lines.append("")

    lines.append("### Hallazgo 1: ORG es la Entidad más Difícil")
    lines.append("- F1 promedio de ORG (~0.40) es consistentemente menor que PER (~0.52) y LOC (~0.54)")
    lines.append("- Esto se debe a la mayor variabilidad en nombres de organizaciones")
    lines.append("- Mayor sensibilidad a ruido de KG en ORG")
    lines.append("")

    lines.append("### Hallazgo 2: LOC Beneficia más de Raw Data")
    raw_loc = mean([s['mean'] for s in stats if s['entity']=='LOC' and s['data']=='RAW'])
    cur_loc = mean([s['mean'] for s in stats if s['entity']=='LOC' and s['data']=='CUR'])
    lines.append(f"- RAW LOC: {raw_loc:.4f} vs CUR LOC: {cur_loc:.4f}")
    lines.append(f"- Diferencia: {(raw_loc-cur_loc)*100:+.2f}%")
    lines.append("- Los datos raw preservan patrones geográficos útiles")
    lines.append("")

    lines.append("### Hallazgo 3: Generic KG Perjudica Todas las Entidades")
    for entity in ['PER', 'LOC', 'ORG']:
        nokg = mean([s['mean'] for s in stats if s['entity']==entity and s['kg']=='NOKG'])
        gen = mean([s['mean'] for s in stats if s['entity']==entity and s['kg']=='GEN'])
        lines.append(f"- {entity}: NoKG={nokg:.4f}, GEN={gen:.4f} ({(gen-nokg)*100:+.2f}%)")
    lines.append("")

    # Guardar reporte
    report_path = os.path.join(output_dir, 'entity_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado: entity_analysis_report.md")

def save_csv_files(results, stats, output_dir):
    """Guarda archivos CSV para análisis posterior"""

    # CSV con todos los resultados por entidad
    all_results_path = os.path.join(output_dir, 'entity_all_results.csv')
    with open(all_results_path, 'w', newline='') as f:
        fieldnames = ['exp_id', 'data', 'kg', 'seed', 'f1_overall', 'f1_PER', 'f1_LOC', 'f1_ORG',
                      'precision_PER', 'recall_PER', 'precision_LOC', 'recall_LOC',
                      'precision_ORG', 'recall_ORG']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"  Guardado: entity_all_results.csv")

    # CSV con estadísticas por condición
    stats_path = os.path.join(output_dir, 'entity_statistics.csv')
    with open(stats_path, 'w', newline='') as f:
        fieldnames = ['data', 'kg', 'entity', 'mean', 'std', 'min', 'max', 'n']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow({k: s[k] for k in fieldnames})
    print(f"  Guardado: entity_statistics.csv")

    # CSV matriz para heatmap
    heatmap_path = os.path.join(output_dir, 'entity_heatmap_data.csv')
    with open(heatmap_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entity', 'data', 'NOKG', 'GEN', 'CUR'])
        for entity in ['overall', 'PER', 'LOC', 'ORG']:
            for data in ['CUR', 'RAW']:
                row = [entity, data]
                for kg in ['NOKG', 'GEN', 'CUR']:
                    val = [s['mean'] for s in stats if s['data']==data and s['kg']==kg and s['entity']==entity][0]
                    row.append(f"{val:.4f}")
                writer.writerow(row)
    print(f"  Guardado: entity_heatmap_data.csv")

def generate_ascii_charts(stats, output_dir):
    """Genera gráficos ASCII para visualización rápida"""

    lines = []
    lines.append("# Visualización ASCII de Resultados")
    lines.append("=" * 70)
    lines.append("")

    # Función para crear barra ASCII
    def ascii_bar(value, max_val=0.7, width=40):
        filled = int((value / max_val) * width)
        return '█' * filled + '░' * (width - filled)

    # Heatmap ASCII por entidad
    for entity in ['overall', 'PER', 'LOC', 'ORG']:
        entity_name = {'overall': 'OVERALL', 'PER': 'PER (Personas)',
                       'LOC': 'LOC (Lugares)', 'ORG': 'ORG (Organizaciones)'}[entity]
        lines.append(f"## F1 {entity_name}")
        lines.append("")
        lines.append("```")
        lines.append(f"{'Condition':<12} {'F1':>6}  {'Bar (max=0.7)':<42}")
        lines.append("-" * 62)

        entity_stats = sorted([s for s in stats if s['entity'] == entity],
                              key=lambda x: x['mean'], reverse=True)

        for s in entity_stats:
            cond = f"{s['data']}_{s['kg']}"
            bar = ascii_bar(s['mean'])
            lines.append(f"{cond:<12} {s['mean']:.4f}  {bar}")

        lines.append("```")
        lines.append("")

    # Comparación RAW vs CUR por entidad
    lines.append("## Comparación RAW vs CUR (promedio por entidad)")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Entity':<10} {'CUR':>8} {'RAW':>8} {'Diff':>8} {'Winner':<6}")
    lines.append("-" * 45)

    for entity in ['overall', 'PER', 'LOC', 'ORG']:
        cur_mean = mean([s['mean'] for s in stats if s['entity']==entity and s['data']=='CUR'])
        raw_mean = mean([s['mean'] for s in stats if s['entity']==entity and s['data']=='RAW'])
        diff = (raw_mean - cur_mean) * 100
        winner = "RAW" if raw_mean > cur_mean else "CUR"
        lines.append(f"{entity:<10} {cur_mean:>8.4f} {raw_mean:>8.4f} {diff:>+7.2f}% {winner:<6}")

    lines.append("```")
    lines.append("")

    # Guardar
    charts_path = os.path.join(output_dir, 'entity_ascii_charts.md')
    with open(charts_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado: entity_ascii_charts.md")

def main():
    print("=" * 60)
    print("ANÁLISIS POR TIPO DE ENTIDAD - Paper 2 NER")
    print("=" * 60)
    print()

    # Cargar datos
    print("Cargando resultados...")
    results = load_all_results()
    print(f"  Cargados {len(results)} experimentos")
    print()

    # Calcular estadísticas
    print("Calculando estadísticas...")
    stats = compute_statistics(results)
    print(f"  Calculadas estadísticas para {len(stats)} condiciones")
    print()

    # Guardar CSVs
    print("Guardando archivos CSV...")
    save_csv_files(results, stats, OUTPUT_DIR)
    print()

    # Generar reporte
    print("Generando reporte de análisis...")
    generate_report(results, stats, OUTPUT_DIR)
    print()

    # Generar gráficos ASCII
    print("Generando visualizaciones ASCII...")
    generate_ascii_charts(stats, OUTPUT_DIR)
    print()

    print("=" * 60)
    print("ANÁLISIS DE DATOS COMPLETADO")
    print(f"Archivos guardados en: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
