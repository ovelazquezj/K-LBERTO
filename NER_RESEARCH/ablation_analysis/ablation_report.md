# Ablation Study Results: KG Quality vs Data Curation

**Generated:** 2026-01-14 08:12:28
**Experiments:** 18/18

## Summary

| Data | KG | F1 Mean | F1 Std | N |
|------|-----|---------|--------|---|
| Curated | Curated (15k) | 0.6156 | 0.0117 | 3 |
| Curated | Generic (3.4M) | 0.5645 | 0.0979 | 3 |
| Curated | None | 0.6244 | 0.0127 | 3 |
| Raw | Curated (15k) | 0.5423 | 0.1091 | 3 |
| Raw | Generic (3.4M) | 0.4430 | 0.1364 | 3 |
| Raw | None | 0.5416 | 0.1190 | 3 |

## ANOVA

- **data:** F=4.24, p=0.1629, η²=0.2221 ✗
- **kg:** F=1.31, p=0.3048, η²=0.1377 ✗
- **interaction:** F=0.11, p=0.8990, η²=0.0113 ✗

## Pairwise Comparisons

| RQ | Comparison | Δ F1 | p | Cohen's d | Sig |
|---|---|---|---|---|---|
| RQ1 | CUR: NoKG vs Generic | +0.0599 | 0.2932 | 0.86 | No |
| RQ2 | CUR: NoKG vs Curated | +0.0088 | 0.3744 | 0.73 | No |
| RQ3 | CUR: Generic vs Curated | -0.0511 | 0.3695 | -0.73 | No |
| RQ4 | NoKG: Curated vs Raw | +0.0829 | 0.2304 | 0.98 | No |
| RQ4b | GEN: Curated vs Raw | +0.1215 | 0.2103 | 1.02 | No |
| RQ4c | CUR_KG: Curated vs Raw | +0.0733 | 0.2472 | 0.94 | No |
| RQ6 | RAW: NoKG vs Curated KG | -0.0008 | 0.9934 | -0.01 | No |
