# Ablation Study Results: KG Quality vs Data Curation

**Generated:** 2026-01-22 21:49:56
**Experiments:** 30/30

## Summary

| Data | KG | F1 Mean | F1 Std | N |
|------|-----|---------|--------|---|
| Curated | Curated (15k) | 0.6100 | 0.0053 | 5 |
| Curated | Generic (3.4M) | 0.5832 | 0.0073 | 5 |
| Curated | None | 0.6099 | 0.0041 | 5 |
| Raw | Curated (15k) | 0.6520 | 0.0029 | 5 |
| Raw | Generic (3.4M) | 0.5867 | 0.1354 | 5 |
| Raw | None | 0.6565 | 0.0060 | 5 |

## ANOVA

- **data:** F=2.30, p=0.3340, η²=0.0717 ✗
- **kg:** F=2.41, p=0.1111, η²=0.1506 ✗
- **interaction:** F=0.45, p=0.6402, η²=0.0284 ✗

## Pairwise Comparisons

| RQ | Comparison | Δ F1 | p | Cohen's d | Sig |
|---|---|---|---|---|---|
| RQ1 | CUR: NoKG vs Generic | +0.0267 | 0.0001 | 4.51 | Yes |
| RQ2 | CUR: NoKG vs Curated | -0.0001 | 0.9786 | -0.02 | No |
| RQ3 | CUR: Generic vs Curated | -0.0268 | 0.0001 | -4.21 | Yes |
| RQ4 | NoKG: Curated vs Raw | -0.0466 | 0.0001 | -9.08 | Yes |
| RQ4b | GEN: Curated vs Raw | -0.0035 | 0.9537 | -0.04 | No |
| RQ4c | CUR_KG: Curated vs Raw | -0.0420 | 0.0001 | -9.88 | Yes |
| RQ6 | RAW: NoKG vs Curated KG | +0.0045 | 0.1279 | 0.96 | No |
