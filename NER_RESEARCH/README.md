# Paper 2: Knowledge Graph Ablation in Low-Resource Spanish NER

## When Curation Hurts: Knowledge Noise and Data Quality Trade-offs in Low-Resource Spanish NER

**Author:** Omar Francisco Velazquez Juarez
**Program:** PhD in Information and Knowledge Engineering - UAH
**Directors:** Dr. Garcia Cabot Antonio, Dra. Garcia Lopez Eva
**Status:** Experiments completed (30/30) - Analysis phase

---

## Overview

This research investigates the effectiveness of Knowledge Graph (KG) injection in Named Entity Recognition (NER) for low-resource Spanish, using K-LBERTO (an edge-optimized K-BERT variant). Through a factorial ablation study (2×3 design), we discovered surprising findings that challenge conventional assumptions about data curation and knowledge enhancement.

### Key Findings

| Finding | Evidence |
|---------|----------|
| **KG does NOT improve NER** | NoKG (0.61) >= Curated KG (0.61) > Generic KG (0.58) |
| **Raw data outperforms curated** | +4.7%, p<0.0001, Cohen's d=9.08 |
| **Task-dependency confirmed** | Sentiment benefits from KG, NER does not |
| **ORG hardest entity** | F1=0.39 vs LOC=0.54, PER=0.52 |

---

## Directory Structure

```
NER_RESEARCH/
├── README.md                          # This file
├── RESEARCH.md                        # Complete research documentation
├── BitacoraPaper2.md                  # Detailed experiment log (Spanish)
├── EstrategiaInvestigacionPaper2.md   # Research strategy (Spanish)
│
├── # Experiment Configuration
├── experiments_ablation_kg.json       # D=2000 ablation config
├── experiments_ablation_kg_d4000.json # D=4000 ablation config (final)
├── experiments_gridsearch_lr.json     # Learning rate grid search config
│
├── # Experiment Scripts
├── run_ablation_experiments.py        # D=2000 experiment runner
├── run_ablation_experiments_d4000.py  # D=4000 experiment runner (final)
├── safety_check_ablation.sh           # D=2000 monitoring script
├── safety_check_ablation_d4000.sh     # D=4000 monitoring script
├── monitor_gridsearch.sh              # Grid search monitoring
│
├── # Analysis Scripts
├── analyze_ablation_results.py        # D=2000 analysis
├── analyze_ablation_results_d4000.py  # D=4000 analysis (final)
├── analyze_entity_data.py             # Entity-level analysis (no matplotlib)
├── analyze_entity_types.py            # Entity analysis with plots
├── generate_figures_colab.py          # Figure generation for Colab/laptop
│
├── # Results Data
├── resultados_ablation/               # D=2000 experiment results
├── resultados_ablation_d4000/         # D=4000 experiment results (final)
├── ablation_analysis/                 # D=2000 analysis outputs
├── ablation_analysis_d4000/           # D=4000 analysis outputs
│
├── # Analysis Reports
├── entity_analysis_report.md          # Entity-level findings
├── entity_ascii_charts.md             # ASCII visualizations
├── entity_statistics.csv              # Statistics by condition
├── entity_all_results.csv             # Raw results (30 experiments)
├── entity_heatmap_data.csv            # Data for heatmap generation
│
├── # Logs
├── ablation_log.txt                   # D=2000 execution log
├── ablation_log_d4000.txt             # D=4000 execution log
├── ablation_progress.json             # D=2000 progress tracker
└── ablation_progress_d4000.json       # D=4000 progress tracker
```

---

## Experimental Design

### Factorial Design: 2 × 3

**Factor 1: Data Quality**
- `CUR` (Curated): Cleaned, validated, balanced dataset
- `RAW` (Raw): Original WikiANN without cleaning

**Factor 2: Knowledge Graph Quality**
- `NOKG` (None): Empty KG baseline
- `GEN` (Generic): Wikidata 3.4M triplets
- `CUR` (Curated): Domain-specific 15k triplets

### Experiment Matrix

| | NoKG | Generic | Curated KG |
|---|---|---|---|
| **Curated Data** | CUR_NOKG | CUR_GEN | CUR_CUR |
| **Raw Data** | RAW_NOKG | RAW_GEN | RAW_CUR |

**Total:** 6 conditions × 5 seeds = 30 experiments

### Fixed Parameters

| Parameter | Value |
|-----------|-------|
| Model | BETO (Spanish BERT) |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 10 |
| Sequence Length | 128 |
| Seeds | 42, 123, 456, 789, 2024 |
| Dataset Size | 4000 samples |

---

## Quick Start: Analysis Pipeline

### Step 1: Run Main Analysis

```bash
# Activate conda environment
conda activate geo_kbert_jetson

# Run ablation analysis (generates CSVs and report)
python analyze_ablation_results_d4000.py

# Output files in ablation_analysis_d4000/:
#   - all_results.csv
#   - summary_stats.csv
#   - pairwise_comparisons.csv
#   - ablation_report.md
```

### Step 2: Run Entity-Level Analysis

```bash
# Generate entity-level statistics (no matplotlib required)
python analyze_entity_data.py

# Output files:
#   - entity_statistics.csv
#   - entity_all_results.csv
#   - entity_heatmap_data.csv
#   - entity_analysis_report.md
#   - entity_ascii_charts.md
```

### Step 3: Monitor Running Experiments (if applicable)

```bash
# Check experiment status
./safety_check_ablation_d4000.sh

# View progress
cat ablation_progress_d4000.json | python -m json.tool
```

---

## Generating Figures for Paper

Due to libstdc++ incompatibility on Jetson, figures must be generated on Colab or a laptop.

### Option A: Google Colab

1. Upload these files to Colab:
   - `entity_statistics.csv`
   - `entity_all_results.csv`
   - `generate_figures_colab.py`
   - `ablation_analysis_d4000/all_results.csv`
   - `ablation_analysis_d4000/summary_stats.csv`

2. Run in Colab notebook:
```python
# Install dependencies if needed
!pip install pandas numpy matplotlib seaborn

# Run figure generation
!python generate_figures_colab.py

# Figures saved to ./figures/
```

3. Download generated figures (PNG + PDF)

### Option B: Local Machine (laptop/desktop)

```bash
# Copy files to local machine
scp -r user@jetson:/path/to/NER_RESEARCH/*.csv ./
scp user@jetson:/path/to/NER_RESEARCH/generate_figures_colab.py ./

# Run locally
python generate_figures_colab.py
```

### Generated Figures

| Figure | Description | Use in Paper |
|--------|-------------|--------------|
| `fig1_heatmap_by_entity.png` | F1 heatmap by entity type | Results section |
| `fig2_barplot_entity_comparison.png` | Entity comparison bars | Results section |
| `fig3_interaction_plot.png` | Data × KG interaction | Results section |
| `fig4_boxplot_by_entity.png` | Distribution boxplots | Appendix |
| `fig_main_results.png` | Main results figure | Abstract/Results |
| `fig6_entity_difficulty.png` | Entity difficulty ranking | Discussion |

---

## Generating Tables for Paper

### Table 1: Main Results (F1 Score)

```bash
# Generate from analysis
python analyze_ablation_results_d4000.py

# Extract table from ablation_analysis_d4000/summary_stats.csv
# Or use the printed table output
```

**Expected output:**
```
                |    NoKG    |   Generic   |   Curated
---------------------------------------------------------
Curated Data    |   0.6099   |   0.5832    |   0.6100
Raw Data        |   0.6565   |   0.5867    |   0.6520
```

### Table 2: Statistical Comparisons

```bash
# From ablation_analysis_d4000/pairwise_comparisons.csv
cat ablation_analysis_d4000/pairwise_comparisons.csv
```

### Table 3: Entity-Level Results

```bash
# From entity_statistics.csv
cat entity_statistics.csv
```

### Table 4: Sensitivity Analysis

For the sensitivity analysis (with/without outlier s456):

```bash
# Run analysis with --partial flag to see intermediate results
python analyze_ablation_results_d4000.py --partial

# Manual calculation for sensitivity:
# RAW_GEN with s456:    mean=0.5867, std=0.121
# RAW_GEN without s456: mean=0.6469, std=0.014
```

---

## Complete Asset Generation Workflow

### 1. Prepare Environment

```bash
cd /path/to/K-LBERTO/NER_RESEARCH
conda activate geo_kbert_jetson
```

### 2. Generate All Analysis Files

```bash
# Main ablation analysis
python analyze_ablation_results_d4000.py

# Entity-level analysis
python analyze_entity_data.py

# Verify outputs
ls -la ablation_analysis_d4000/
ls -la entity_*.csv entity_*.md
```

### 3. Collect Data for Colab

```bash
# Create assets bundle
mkdir -p paper_assets
cp ablation_analysis_d4000/*.csv paper_assets/
cp entity_*.csv paper_assets/
cp generate_figures_colab.py paper_assets/
cp entity_analysis_report.md paper_assets/
cp RESEARCH.md paper_assets/

# Create tarball for transfer
tar -czvf paper_assets.tar.gz paper_assets/
```

### 4. Generate Figures in Colab

```python
# In Colab notebook:

# Upload paper_assets.tar.gz and extract
!tar -xzvf paper_assets.tar.gz
%cd paper_assets

# Generate figures
!python generate_figures_colab.py

# List generated figures
!ls -la figures/
```

### 5. Generate LaTeX Tables (optional)

```python
import pandas as pd

# Load data
stats = pd.read_csv('entity_statistics.csv')

# Generate LaTeX
print(stats.to_latex(index=False, float_format='%.4f'))
```

---

## Key Files Reference

### For Understanding the Research

| File | Purpose |
|------|---------|
| `RESEARCH.md` | Complete methodology, findings, conclusions |
| `BitacoraPaper2.md` | Detailed chronological log |
| `EstrategiaInvestigacionPaper2.md` | Research strategy and RQs |

### For Reproducing Analysis

| File | Purpose |
|------|---------|
| `analyze_ablation_results_d4000.py` | Main statistical analysis |
| `analyze_entity_data.py` | Entity-level breakdown |
| `experiments_ablation_kg_d4000.json` | Experiment configuration |

### For Paper Writing

| File | Purpose |
|------|---------|
| `ablation_analysis_d4000/summary_stats.csv` | Main results table |
| `ablation_analysis_d4000/pairwise_comparisons.csv` | Statistical tests |
| `entity_statistics.csv` | Entity-level table |
| `generate_figures_colab.py` | All paper figures |

---

## Results Summary

### Final Results (30/30 Experiments)

```
F1 Score by Condition:

                NoKG      Generic    Curated KG
Curated Data    0.6099    0.5832     0.6100
Raw Data        0.6565    0.5867     0.6520

Best:  RAW + NoKG  (F1 = 0.6565)
Worst: CUR + GEN   (F1 = 0.5832)
```

### Statistical Significance

| Comparison | p-value | Significant |
|------------|---------|-------------|
| NoKG vs Generic (CUR) | <0.0001 | Yes |
| NoKG vs Curated (CUR) | 0.9786 | No |
| Generic vs Curated (CUR) | <0.0001 | Yes |
| CUR vs RAW (NoKG) | <0.0001 | Yes |
| CUR vs RAW (CUR_KG) | <0.0001 | Yes |
| CUR vs RAW (GEN) | 0.9537 | No* |

*Affected by outlier s456; significant without outlier (p<0.01)

### Entity-Level Summary

```
Entity Difficulty (hardest to easiest):
1. ORG: F1 = 0.3948
2. PER: F1 = 0.5201
3. LOC: F1 = 0.5434

RAW > CUR for ALL entities:
- ORG: +5.99%
- PER: +1.65%
- LOC: +1.28%
```

---

## Citation

```bibtex
@article{velazquez2026curation,
  title={When Curation Hurts: Knowledge Noise and Data Quality Trade-offs
         in Low-Resource Spanish NER},
  author={Velázquez Juárez, Omar Francisco and García Cabot, Antonio
          and García López, Eva},
  journal={TBD},
  year={2026}
}
```

---

## Contact

- **Author:** Omar Francisco Velázquez Juárez
- **Institution:** Universidad de Alcalá de Henares
- **Program:** PhD in Information and Knowledge Engineering

---

*Last updated: 2026-01-25*
