#!/usr/bin/env python3
"""
Extract entity-level metrics from the best epoch (max f1_overall on test)
for all 30 D=4000 ablation experiments.

This ensures consistency with analyze_ablation_results_d4000.py which uses
idxmax() to select the best test epoch, unlike analyze_entity_data.py
which uses the last test row.

Author: Omar Velázquez
Date: 2026-02-09
"""

import csv
import os
from collections import defaultdict
import math

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados_ablation_d4000")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "entity_results_best_epoch.csv")
SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "..", "entity_summary_best_epoch.csv")

# Reference values from all_results.csv (analyze_ablation_results_d4000.py)
REFERENCE = {
    "CUR_NOKG_s42": 0.6047, "CUR_NOKG_s123": 0.6157, "CUR_NOKG_s456": 0.6117,
    "CUR_NOKG_s789": 0.6089, "CUR_NOKG_s2024": 0.6085,
    "CUR_GEN_s42": 0.5766, "CUR_GEN_s123": 0.5954, "CUR_GEN_s456": 0.5799,
    "CUR_GEN_s789": 0.5800, "CUR_GEN_s2024": 0.5840,
    "CUR_CUR_s42": 0.6078, "CUR_CUR_s123": 0.6060, "CUR_CUR_s456": 0.6192,
    "CUR_CUR_s789": 0.6084, "CUR_CUR_s2024": 0.6085,
    "RAW_NOKG_s42": 0.6493, "RAW_NOKG_s123": 0.6652, "RAW_NOKG_s456": 0.6542,
    "RAW_NOKG_s789": 0.6592, "RAW_NOKG_s2024": 0.6546,
    "RAW_GEN_s42": 0.6236, "RAW_GEN_s123": 0.6510, "RAW_GEN_s456": 0.3458,
    "RAW_GEN_s789": 0.6584, "RAW_GEN_s2024": 0.6547,
    "RAW_CUR_s42": 0.6481, "RAW_CUR_s123": 0.6519, "RAW_CUR_s456": 0.6547,
    "RAW_CUR_s789": 0.6502, "RAW_CUR_s2024": 0.6549,
}

COLUMNS = [
    "exp_id", "data", "kg", "seed", "best_epoch",
    "f1_overall", "precision_overall", "recall_overall",
    "f1_PER", "f1_LOC", "f1_ORG",
    "precision_PER", "precision_LOC", "precision_ORG",
    "recall_PER", "recall_LOC", "recall_ORG",
]

CONDITIONS = ["CUR_NOKG", "CUR_GEN", "CUR_CUR", "RAW_NOKG", "RAW_GEN", "RAW_CUR"]
SEEDS = ["42", "123", "456", "789", "2024"]


def parse_experiment_id(dirname):
    """Parse ABL_CUR_NOKG_s42 -> (CUR, NOKG, 42)"""
    parts = dirname.replace("ABL_", "").split("_")
    data = parts[0]
    kg = parts[1]
    seed = parts[2].replace("s", "")
    return data, kg, seed


def process_experiment(exp_dir, dirname):
    """Read metrics CSV and return best test epoch row."""
    data, kg, seed = parse_experiment_id(dirname)
    condition = f"{data}_{kg}"
    exp_id = f"{condition}_s{seed}"

    metrics_file = os.path.join(exp_dir, f"ABL_{exp_id}_metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"  WARNING: {metrics_file} not found")
        return None

    # Read all test rows
    test_rows = []
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test':
                test_rows.append(row)

    if not test_rows:
        print(f"  WARNING: No test rows in {dirname}")
        return None

    # Find best test row by f1_overall (same logic as analyze_ablation_results_d4000.py)
    best_row = max(test_rows, key=lambda r: float(r['f1_overall']))

    result = {
        "exp_id": exp_id,
        "data": data,
        "kg": kg,
        "seed": seed,
        "best_epoch": best_row['epoch'],
        "f1_overall": float(best_row['f1_overall']),
        "precision_overall": float(best_row['precision_overall']),
        "recall_overall": float(best_row['recall_overall']),
        "f1_PER": float(best_row['f1_PER']),
        "f1_LOC": float(best_row['f1_LOC']),
        "f1_ORG": float(best_row['f1_ORG']),
        "precision_PER": float(best_row['precision_PER']),
        "precision_LOC": float(best_row['precision_LOC']),
        "precision_ORG": float(best_row['precision_ORG']),
        "recall_PER": float(best_row['recall_PER']),
        "recall_LOC": float(best_row['recall_LOC']),
        "recall_ORG": float(best_row['recall_ORG']),
    }

    return result


def verify_against_reference(results):
    """Verify f1_overall matches reference values."""
    print("\n" + "=" * 70)
    print("VERIFICACIÓN CONTRA VALORES DE REFERENCIA")
    print("=" * 70)

    discrepancies = []
    for r in results:
        exp_id = r['exp_id']
        computed = r['f1_overall']
        expected = REFERENCE.get(exp_id)

        if expected is None:
            print(f"  WARNING: {exp_id} not in reference")
            continue

        match = abs(computed - expected) < 0.00005
        status = "OK" if match else "MISMATCH"

        if not match:
            discrepancies.append((exp_id, computed, expected))
            print(f"  {status}: {exp_id}  computed={computed:.4f}  expected={expected:.4f}  diff={computed-expected:+.4f}")

    if not discrepancies:
        print(f"\n  Todos los 30 valores coinciden con la referencia.")
    else:
        print(f"\n  DISCREPANCIAS ENCONTRADAS: {len(discrepancies)}")
        for exp_id, computed, expected in discrepancies:
            print(f"    {exp_id}: {computed:.4f} vs {expected:.4f} (diff={computed-expected:+.4f})")

    return discrepancies


def compute_summary(results):
    """Compute mean and SD per condition."""
    by_condition = defaultdict(list)
    for r in results:
        cond = f"{r['data']}_{r['kg']}"
        by_condition[cond].append(r)

    summary = []
    for cond in CONDITIONS:
        rows = by_condition.get(cond, [])
        if not rows:
            continue

        n = len(rows)
        metrics = ['f1_overall', 'f1_PER', 'f1_LOC', 'f1_ORG']
        entry = {"condition": cond, "n": n}

        for m in metrics:
            values = [r[m] for r in rows]
            mean = sum(values) / n
            if n > 1:
                sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
            else:
                sd = 0.0
            entry[f"{m}_mean"] = mean
            entry[f"{m}_sd"] = sd

        summary.append(entry)

    return summary


def main():
    print("=" * 70)
    print("EXTRACCIÓN DE MÉTRICAS POR ENTIDAD - BEST EPOCH")
    print("=" * 70)

    # Process all 30 experiments
    results = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            dirname = f"ABL_{cond}_s{seed}"
            exp_dir = os.path.join(RESULTS_DIR, dirname)
            if not os.path.isdir(exp_dir):
                print(f"  WARNING: Directory not found: {dirname}")
                continue

            result = process_experiment(exp_dir, dirname)
            if result:
                results.append(result)
                print(f"  {result['exp_id']:20s}  epoch={result['best_epoch']:>2s}  "
                      f"f1={result['f1_overall']:.4f}  "
                      f"PER={result['f1_PER']:.4f}  LOC={result['f1_LOC']:.4f}  ORG={result['f1_ORG']:.4f}")

    print(f"\nTotal experiments processed: {len(results)}/30")

    # Verify against reference
    discrepancies = verify_against_reference(results)
    if discrepancies:
        print("\n*** HAY DISCREPANCIAS - revisar antes de continuar ***\n")

    # Write detailed CSV
    print(f"\nGuardando: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r[col] for col in COLUMNS})

    # Compute and write summary
    summary = compute_summary(results)

    print(f"Guardando: {SUMMARY_CSV}")
    summary_cols = ["condition", "n",
                    "f1_overall_mean", "f1_overall_sd",
                    "f1_PER_mean", "f1_PER_sd",
                    "f1_LOC_mean", "f1_LOC_sd",
                    "f1_ORG_mean", "f1_ORG_sd"]
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for s in summary:
            writer.writerow({col: s.get(col, '') for col in summary_cols})

    # Print summary table
    print("\n" + "=" * 70)
    print("RESUMEN POR CONDICIÓN (mean ± sd)")
    print("=" * 70)
    print(f"\n{'Condition':12s}  {'n':>2s}  {'F1 Overall':>14s}  {'F1 PER':>14s}  {'F1 LOC':>14s}  {'F1 ORG':>14s}")
    print("-" * 78)
    for s in summary:
        print(f"{s['condition']:12s}  {s['n']:2d}  "
              f"{s['f1_overall_mean']:.4f} ± {s['f1_overall_sd']:.4f}  "
              f"{s['f1_PER_mean']:.4f} ± {s['f1_PER_sd']:.4f}  "
              f"{s['f1_LOC_mean']:.4f} ± {s['f1_LOC_sd']:.4f}  "
              f"{s['f1_ORG_mean']:.4f} ± {s['f1_ORG_sd']:.4f}")

    # Print best epoch distribution
    print("\n" + "=" * 70)
    print("DISTRIBUCIÓN DE BEST EPOCH")
    print("=" * 70)
    epoch_counts = defaultdict(int)
    for r in results:
        epoch_counts[r['best_epoch']] += 1
    for epoch in sorted(epoch_counts.keys(), key=int):
        bar = '█' * epoch_counts[epoch]
        print(f"  Epoch {epoch:>2s}: {epoch_counts[epoch]:2d} {bar}")

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
