# Paper 2 Research Documentation - Knowledge Graph Ablation in Spanish NER

## When Curation Hurts: Knowledge Noise and Data Quality Trade-offs in Low-Resource Spanish NER

**Researcher:** Omar Francisco Velazquez Juarez
**Research Directors:** Dr. Garcia Cabot Antonio, Dra. Garcia Lopez Eva
**Date:** January 22, 2026
**Duration:** ~115 hours (experiment execution) + analysis
**Application:** Empirical validation of Knowledge Graph effectiveness in token-level NLP tasks

---

## Methodological Notes

This research document was generated through an iterative experimental process that included:

1. Factorial ablation design (2x3: Data curation x KG quality)
2. Execution of 30 experiments on Jetson Orin NX hardware
3. Statistical analysis with ANOVA and pairwise comparisons
4. Critical validation through multiple random seeds (5 per condition)

The use of AI tools in research follows current academic standards for transparency and methodology. Claude AI was utilized for:
- Experiment monitoring and progress tracking
- Statistical analysis interpretation
- Documentation structure and content refinement
- Sensitivity analysis recommendations

All experimental work, data collection, model training, and results interpretation were conducted by the author. AI assistance served as a tool for optimization and documentation, not as a substitute for research rigor.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Context](#research-context)
3. [Experimental Design](#experimental-design)
4. [Results Overview](#results-overview)
5. [Key Findings](#key-findings)
6. [Statistical Analysis](#statistical-analysis)
7. [Sensitivity Analysis](#sensitivity-analysis)
8. [Discussion](#discussion)
9. [Conclusions](#conclusions)
10. [Recommendations](#recommendations)

---

# Executive Summary

## Scientific Journey: Paper 1 to Paper 2

```
Paper 1 (Sentiment Analysis - Sentence Level)
- Task: Classification (4 classes)
- Finding: KG curated IMPROVES performance (+12%)
- Finding: Data curation IMPROVES performance
- Finding: LR scaling is CRITICAL
- Conclusion: "Curation Over Scale" validated

Paper 2 (NER - Token Level)
- Task: Named Entity Recognition (PER, LOC, ORG)
- Finding: KG does NOT improve performance
- Finding: Generic KG HURTS performance (-2.7%)
- Finding: Raw data OUTPERFORMS curated data (+4.7%)
- Conclusion: Task-dependency is CRITICAL
```

## Key Discovery

The effectiveness of Knowledge Graph injection and data curation is TASK-DEPENDENT:

- **Sentence-level tasks (Sentiment):** KG helps, curation helps
- **Token-level tasks (NER):** KG neutral/hurts, curation HURTS

This contradicts the assumption that improvements generalize across NLP tasks.

## Main Results (30/30 Experiments)

| Data | NoKG | Generic KG | Curated KG |
|------|------|------------|------------|
| Curated | 0.6099 | 0.5832 | 0.6100 |
| Raw | **0.6565** | 0.5867 | **0.6520** |

Best condition: **RAW + NoKG (F1=0.6565)**
Worst condition: **CUR + Generic KG (F1=0.5832)**

---

# Research Context

## Background: Paper 1 Findings

Paper 1 established that for sentiment analysis:
1. Data curation is critical (removes noise)
2. KG curation improves knowledge injection
3. Hyperparameter scaling is required when dataset grows

## Research Gap

Paper 1 did not quantify:
1. What percentage of improvement comes from data curation
2. What percentage comes from KG quality
3. Whether findings generalize to other NLP tasks

## Director's Proposal (Gap #4)

> "Conduct an ablation study of knowledge, comparing K-LBERTO performance using a generic graph versus a specialized (curated) one under the same hyperparameter configuration. This would isolate the impact of 'semantic quality' from 'training data quality'."

## Emergent Finding

Preliminary results showed an UNEXPECTED pattern:

```
NoKG (0.6099) > Curated KG (0.6100) > Generic KG (0.5832)
```

This contradicts the expectation that KG improves performance (based on Paper 1 - Sentiment).

---

# Experimental Design

## Factorial Design: 2 x 3

**Factor 1: Data Quality**
- Curated (CUR): Cleaned, validated, balanced
- Raw (RAW): Original WikiANN without cleaning

**Factor 2: KG Quality**
- None (NoKG): Empty KG (baseline)
- Generic (GEN): Wikidata 3.4M triplets
- Curated (CUR): Domain-specific 15k triplets

## Experiment Matrix

```
                NoKG        Generic       Curated
Curated Data    CUR_NOKG    CUR_GEN       CUR_CUR
Raw Data        RAW_NOKG    RAW_GEN       RAW_CUR
```

**Total: 6 conditions x 5 seeds = 30 experiments**

## Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Batch Size | 8 | Hardware constraint (Jetson) |
| Epochs | 10 | Sufficient for convergence |
| Sequence Length | 128 | Standard for NER |
| Seeds | 42, 123, 456, 789, 2024 | Reproducibility |

## Dataset Configuration

**Training Data:**
- Curated: train_4000.tsv (4000 samples, cleaned)
- Raw: train_raw_4000.tsv (4000 samples, original)

**Evaluation:**
- dev.tsv: Validation set
- test.tsv: Final evaluation

## Knowledge Graphs

| KG | Triplets | Description |
|----|----------|-------------|
| empty.spo | 1 | Dummy (never matches) |
| WikidataES_CLEAN | 3.4M | Generic multi-domain |
| wikidata_ner_spanish | 15k | NER-specific curated |

---

# Results Overview

## Execution Summary

```
Start:              2026-01-17 ~21:00
End:                2026-01-22 16:06
Duration:           ~115 hours
Experiments:        30/30 (100%)
Failed:             0
OOM errors:         0
```

## Results by Condition (F1 Score)

### Curated Data (15/15 complete)

```
CUR_NOKG: 0.6047, 0.6157, 0.6117, 0.6089, 0.6085
          Mean: 0.6099 +/- 0.0043

CUR_GEN:  0.5766, 0.5954, 0.5799, 0.5800, 0.5840
          Mean: 0.5832 +/- 0.0076

CUR_CUR:  0.6078, 0.6060, 0.6192, 0.6084, 0.6085
          Mean: 0.6100 +/- 0.0053
```

### Raw Data (15/15 complete)

```
RAW_NOKG: 0.6493, 0.6652, 0.6542, 0.6592, 0.6546
          Mean: 0.6565 +/- 0.0058

RAW_GEN:  0.6236, 0.6510, 0.3458*, 0.6584, 0.6547
          Mean: 0.5867 +/- 0.1211
          *s456 = outlier (catastrophic collapse)
          Mean without s456: 0.6469 +/- 0.0137

RAW_CUR:  0.6481, 0.6519, 0.6547, 0.6502, 0.6549
          Mean: 0.6520 +/- 0.0028
```

## Summary Table

```
                |    NoKG    |   Generic   |   Curated   |
---------------------------------------------------------
Curated Data    |   0.6099   |   0.5832    |   0.6100    |
Raw Data        |   0.6565   |   0.5867    |   0.6520    |
---------------------------------------------------------
```

---

# Key Findings

## Finding 1: Knowledge Graph Does NOT Improve NER

**Evidence:**
- NoKG (0.6099) >= Curated KG (0.6100) > Generic KG (0.5832)
- Difference NoKG vs Generic: +2.67%, p<0.0001, Cohen's d=4.51
- Difference NoKG vs Curated: -0.01%, p=0.98 (not significant)

**Interpretation:**
The Knowledge Graph provides NO benefit for NER in low-resource Spanish.
The generic KG (3.4M triplets) HURTS performance.
The curated KG (15k triplets) is NEUTRAL.

**Contrast with Paper 1:**
- Paper 1 (Sentiment): Curated KG IMPROVES +12% vs baseline
- Paper 2 (NER): Curated KG is NEUTRAL, generic HURTS

**Proposed Text:**
> "Contrary to findings in sentence-level tasks, knowledge graph injection provides no benefit for token-level NER in low-resource Spanish. The generic KG with 3.4M triplets significantly degraded performance (F1=0.583 vs 0.610 baseline, p<0.0001), while the curated domain-specific KG with 15k triplets showed no improvement (F1=0.610 vs 0.610, p=0.98). This suggests that the knowledge noise introduced by entity injection disrupts the fine-grained boundary detection required for NER."

---

## Finding 2: Raw Data Outperforms Curated Data (Inversion)

**Evidence:**
- NoKG condition: RAW (0.6565) > CUR (0.6099) = +4.66%, p<0.0001, d=9.08
- CUR_KG condition: RAW (0.6520) > CUR (0.6100) = +4.20%, p<0.0001, d=9.88
- GEN condition: RAW (0.5867) > CUR (0.5832) = +0.35%, p=0.95, ns
- Without s456 outlier: RAW (0.6469) > CUR (0.5832) = +6.37% (SIGNIFICANT)

**Effect Size:**
- Cohen's d = 9.08 to 9.88 (HUGE effect, >0.8 is "large")
- Absolute difference: +4.2% to +4.7% F1

**Proposed Text:**
> "Surprisingly, models trained on raw uncurated data consistently outperformed those trained on carefully curated data. In the baseline condition without KG, raw data achieved F1=0.657 compared to F1=0.610 for curated data (+4.7%, p<0.0001, Cohen's d=9.08). This pattern held across KG conditions, suggesting that aggressive curation removes linguistically informative patterns that benefit NER boundary detection. This finding challenges the assumption that cleaner data necessarily produces better models."

---

## Finding 3: Task-Dependency (Sentence vs Token Level)

**Cross-study Evidence:**

| Aspect | Paper 1 (Sentiment) | Paper 2 (NER) |
|--------|---------------------|---------------|
| Curated KG | IMPROVES +12% | NEUTRAL |
| Generic KG | N/A | HURTS -2.7% |
| Data curation | IMPROVES | HURTS -4.7% |
| Task level | Sentence | Token |
| Representation | CLS token | Per-token |

**Proposed Mechanism:**

Sentence-level:
- CLS token aggregates global context
- More information = better
- KG enriches semantic representation

Token-level:
- Each token needs precise boundary
- Injected tokens INTERRUPT positions
- KG introduces NOISE in sequence

**Proposed Text:**
> "Our results reveal a fundamental task-dependency in knowledge graph effectiveness. While KG injection benefits sentence-level tasks like sentiment analysis where the CLS token aggregates enriched context, it fails or harms token-level tasks like NER. We hypothesize that injected knowledge tokens disrupt the positional patterns critical for entity boundary detection, even when using soft-position embeddings. This finding has important implications for practitioners: KG-enhanced models should not be assumed to generalize across NLP task types."

---

## Finding 4: Generic KG is Worse than Curated KG

**Evidence:**
- For curated data: Curated KG (0.6100) > Generic KG (0.5832) = +2.68%, p<0.0001
- For raw data: Curated KG (0.6520) > Generic KG (0.5867) = +6.53% (with outlier)

**Proposed Text:**
> "When knowledge graphs are employed, quality dramatically outweighs quantity. The generic Wikidata KG with 3.4M triplets consistently underperformed the domain-curated KG with 15k triplets by 2.7-6.5 percentage points. This confirms the knowledge noise phenomenon identified by Liu et al.: indiscriminate knowledge injection introduces irrelevant associations that confuse rather than inform the model."

---

## Finding 5: Seed 456 as Instability Indicator

**Evidence:**
- RAW + Generic KG: F1 = 0.3458 (catastrophic collapse)
- Other conditions with s456: F1 = 0.55-0.66 (normal)
- Deviation: >2 SD below mean

**Interpretation:**
The collapse occurs ONLY in RAW + Generic KG.
This combination is particularly UNSTABLE.
Noisy data + noisy KG amplifies problems.

**Proposed Text (Appendix):**
> "Seed 456 exhibited convergence failure exclusively in the RAW+Generic KG condition with F1=0.346 vs mean 0.647 for other seeds. This isolated failure suggests that the combination of uncurated training data and large generic knowledge graphs creates training instability. Following Bethard et al. 2022, we retain this seed in primary analysis but provide sensitivity analysis showing that conclusions regarding RQ4b change from non-significant to significant when excluded."

---

# Statistical Analysis

## ANOVA Factorial 2x3 (Data x KG)

```
   Source     SS      df      MS       F       p      eta2
-----------------------------------------------------------
   Data     0.0071    1    0.0071   2.30   0.334   0.072
   KG       0.0148    2    0.0074   2.41   0.111   0.151
   Data*KG  0.0028    2    0.0014   0.45   0.640   0.028
   Error    0.0739   24    0.0031
   Total    0.0986   29
```

**Effect Sizes:**
- Data: eta2 = 0.072 (medium)
- KG: eta2 = 0.151 (large)
- Interaction: eta2 = 0.028 (small)

## Pairwise Comparisons

```
  RQ   |        Comparison         | Mean_A | Mean_B |  Diff  |    t    |    p    | Cohen_d | Sig
-------------------------------------------------------------------------------------------------
 RQ1  | CUR: NoKG vs Generic       | 0.6099 | 0.5832 | +0.027 |   7.13  | 0.0001  |   4.51  |  Yes
 RQ2  | CUR: NoKG vs Curated       | 0.6099 | 0.6100 | -0.000 |  -0.03  | 0.9786  |  -0.02  |  No
 RQ3  | CUR: Generic vs Curated    | 0.5832 | 0.6100 | -0.027 |  -6.65  | 0.0001  |  -4.21  |  Yes
 RQ4  | NoKG: Curated vs Raw       | 0.6099 | 0.6565 | -0.047 | -14.36  | 0.0001  |  -9.08  |  Yes
 RQ4b | GEN: Curated vs Raw        | 0.5832 | 0.5867 | -0.004 |  -0.06  | 0.9537  |  -0.04  |  No
 RQ4c | CUR_KG: Curated vs Raw     | 0.6100 | 0.6520 | -0.042 | -15.62  | 0.0001  |  -9.88  |  Yes
 RQ6  | RAW: NoKG vs Curated KG    | 0.6565 | 0.6520 | +0.005 |   1.52  | 0.1279  |   0.96  |  No
```

## Summary of Significance

| Status | RQs | Count |
|--------|-----|-------|
| Significant (p<0.05) | RQ1, RQ3, RQ4, RQ4c | 4 |
| Not significant | RQ2, RQ4b, RQ6 | 3 |

---

# Sensitivity Analysis

## Option B: Report Both Analyses

Following methodological best practices (Bethard 2022, Dodge 2020), we report analyses with and without the s456 outlier.

### Impact of Outlier Exclusion

```
                          WITH s456 (n=5)    WITHOUT s456 (n=4)
----------------------------------------------------------------
RAW_GEN mean              0.5867 +/- 0.121   0.6469 +/- 0.014
RQ4b (GEN: CUR vs RAW)    p=0.954, ns        p<0.01, sig***
Conclusion                No difference      RAW > CUR
----------------------------------------------------------------
```

### Impact on Conclusions

| Analysis | Significant RQs | RAW > CUR Pattern |
|----------|-----------------|-------------------|
| With s456 | RQ1, RQ3, RQ4, RQ4c (4/7) | 2/3 conditions |
| Without s456 | RQ1, RQ3, RQ4, RQ4b, RQ4c (5/7) | 3/3 conditions |

### Interpretation

The primary conclusion (RAW > CUR) is robust:
- Confirmed in 2/3 conditions with all data
- Confirmed in 3/3 conditions without outlier
- The outlier affects only the GEN condition

---

# Discussion

## Why Does KG Help Sentiment but Not NER?

### Sentence-Level Tasks (Sentiment)
- Uses CLS token that aggregates entire sequence
- Additional tokens ENRICH global representation
- Model does NOT need precise boundaries
- More context = better classification

### Token-Level Tasks (NER)
- Each token needs individual classification
- Entity BOUNDARIES are critical
- Injected tokens SHIFT positions
- Even soft-position may not fully compensate

### Proposed Mechanism

```
Original:  [Madrid]  [es]  [la]  [capital]  [de]  [Espana]
Labels:    B-LOC     O     O     O          O     B-LOC

With KG:   [Madrid] [capital] [Espana] [ciudad] [es] [la] [capital] [de] [Espana]
                    ^injected ^injected ^injected
Problem:   Position shifts, boundary patterns disrupted
```

## Why Does Raw Data Outperform Curated Data?

### Hypothesis 1: Information Loss
- Curation removes "noise" that contains signal
- NER benefits from diverse linguistic patterns
- Aggressive cleaning removes edge cases

### Hypothesis 2: Distribution Shift (CONFIRMED)

**Quantitative Evidence:**

| Metric | TEST | CUR (train) | RAW (train) | Gap CUR | Gap RAW |
|--------|------|-------------|-------------|---------|---------|
| REDIRECCIÓN samples | 36.9% | 0.8% | 27.7% | **36.1%** | 9.2% |
| High density (>50%) | 50.8% | 23.1% | 62.1% | 27.7% | 11.3% |
| Average entity density | ~0.55 | 0.408 | 0.645 | High | Low |

**Mechanism:**
- Curation criterion ">90% entities" removed 3,755 samples from training pool
- These samples (REDIRECCIÓN, entity lists) constitute 37% of test set
- CUR model never learned patterns present in 37% of evaluation data
- RAW model maintained distribution alignment with test set

### Hypothesis 3: Boundary Signals
- "Noisy" tokens may mark entity boundaries
- Punctuation, special characters serve as delimiters
- Curation removes these natural markers

## Implications for Practitioners

1. **Do NOT assume KG helps all tasks**
   - Test on your specific task type
   - Sentence-level vs token-level matters

2. **Data curation is not always beneficial**
   - For NER, some "noise" may be signal
   - Evaluate curation impact empirically

3. **Quality over quantity for KG**
   - If using KG, curate it carefully
   - Generic KGs introduce more noise

---

# Conclusions

## Research Questions Answered

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | Does generic KG improve over baseline? | **NO** - Generic HURTS (-2.7%) |
| RQ2 | Does curated KG improve over baseline? | **NO** - Curated is NEUTRAL |
| RQ3 | Is curated KG > generic KG? | **YES** - Quality > Quantity |
| RQ4 | Does data curation help? | **NO** - Raw > Curated (+4.7%) |
| RQ5 | Is there Data x KG interaction? | **NO** - Effects are additive |
| RQ6 | Can KG compensate bad data? | **NO** - KG does not help RAW |
| RQ7 | Why different from Paper 1? | Task-dependency (token vs sentence) |

## Main Contributions

### Contribution 1: Task-Dependency Framework
- KG effectiveness depends on task granularity
- Sentence-level: KG helps
- Token-level: KG neutral/hurts

### Contribution 2: Curation Trade-off Discovery
- Data curation can HURT NER performance
- "Noise" may contain boundary signals
- Challenges assumption that clean = better

### Contribution 3: Knowledge Noise Confirmation
- Generic KGs introduce harmful noise
- Liu et al. (2020) finding confirmed for Spanish NER
- Quality dramatically outweighs quantity

### Contribution 4: Methodological Recommendations
- Always test KG on specific task type
- Evaluate curation impact empirically
- Use multiple seeds to detect instability

---

# Recommendations

## For Researchers

1. **When applying KG-enhanced models:**
   - Test on your specific task granularity
   - Do NOT assume benefits generalize
   - Compare against NoKG baseline

2. **When curating data for NER:**
   - Evaluate impact of each cleaning step
   - Some "noise" may be informative
   - Test raw vs curated empirically

3. **When selecting Knowledge Graphs:**
   - Prefer domain-specific curated KGs
   - Generic KGs often hurt more than help
   - Smaller + relevant > larger + generic

4. **When reporting results:**
   - Use multiple random seeds (>=5)
   - Report sensitivity analysis for outliers
   - Provide both with/without outlier results

## For Practitioners

```
Decision tree for KG usage:

1. What is your task type?
   - Sentence-level (classification, sentiment) -> KG may help
   - Token-level (NER, POS tagging) -> KG likely neutral/hurts

2. Do you have a curated KG?
   - Yes -> May provide modest benefit
   - No (generic only) -> Likely to hurt, skip KG

3. Is your data "clean"?
   - Very clean -> May have removed useful patterns
   - Some noise -> May actually help for NER

4. Recommendation:
   - Always compare KG vs NoKG on YOUR task
   - Always compare Curated vs Raw on YOUR task
   - Do NOT trust cross-task generalizations
```

---

# Appendix: Experimental Details

## Hardware Configuration

```
Device:          Jetson Orin NX 16GB
GPU:             Ampere architecture
RAM:             16GB shared CPU/GPU
Storage:         NVMe SSD
OS:              Ubuntu 20.04 (JetPack 5.x)
```

## Software Configuration

```
Python:          3.8
PyTorch:         1.13.0
Transformers:    4.x (BETO model)
CUDA:            11.4
```

## Training Time per Experiment

```
Average:         ~230 minutes (3.8 hours)
Total (30 exp):  ~115 hours
Batch size:      8
Epochs:          10
```

## Files Generated

- `ablation_analysis_d4000/all_results.csv`: Complete results
- `ablation_analysis_d4000/summary_stats.csv`: Summary statistics
- `ablation_analysis_d4000/pairwise_comparisons.csv`: Statistical tests
- `ablation_analysis_d4000/ablation_report.md`: Auto-generated report

---

# References

1. Liu, W., et al. (2020). K-BERT: Enabling Language Representation with Knowledge Graph. AAAI 2020.

2. Bethard, S., et al. (2022). Random Seed Selection and Reporting Practices in NLP. ACL 2022.

3. Dodge, J., et al. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv.

4. Mosbach, M., et al. (2021). On the Stability of Fine-tuning BERT. EACL 2021.

5. Wang, X., et al. (2019). KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation. TACL 2019.

---

*Document generated: 2026-01-22*
*Status: Experiments completed, analysis finalized*
*Next steps: Paper draft, figures generation, director review*
