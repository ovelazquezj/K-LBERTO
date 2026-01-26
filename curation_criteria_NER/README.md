# Paper 3: Curation Criteria Impact on NER

## A Systematic Study of Data Curation Effects on Named Entity Recognition

**Author:** Omar Francisco Velázquez Juárez
**Program:** PhD in Information and Knowledge Engineering - UAH
**Directors:** Dr. García Cabot Antonio, Dra. García López Eva
**Status:** Proposal - Pending director approval

---

## Overview

This research proposes a systematic study of how different data curation criteria affect NER performance. Building on findings from Paper 2 (where aggressive curation hurt performance), this study aims to:

1. Evaluate individual impact of 10 curation criteria
2. Identify beneficial vs harmful criteria for NER
3. Develop a distribution-aware curation framework
4. Provide task-specific curation guidelines

---

## Background

Paper 2 discovered that curated data underperformed raw data by 4.7% in Spanish NER. Root cause analysis revealed:

- Curation criterion ">90% entity density" removed patterns present in 37% of test data
- This created a train-test distribution shift
- The model never learned patterns it would encounter during evaluation

**Key Question:** Which curation criteria help NER, and which hurt it?

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | What is the individual impact of each curation criterion on NER? |
| RQ2 | Are there interactions between curation criteria? |
| RQ3 | Does impact vary by entity type (PER/LOC/ORG)? |
| RQ4 | How does train-test alignment affect each criterion? |
| RQ5 | Can we predict optimal criteria based on dataset characteristics? |

---

## Experimental Design

### Curation Criteria to Evaluate

| ID | Criterion | Description |
|----|-----------|-------------|
| C1 | Min tokens | Remove samples with <3 tokens |
| C2 | No entities | Remove samples without entities |
| C3 | BIO inconsistent | Remove invalid I- tags |
| C4 | Duplicates | Remove repeated samples |
| C5 | High density (>90%) | Remove almost-pure entity samples |
| C6 | High density (>80%) | More conservative threshold |
| C7 | High density (>70%) | Even more conservative |
| C8 | REDIRECCIÓN | Remove redirect samples |
| C9 | Max length | Remove very long samples (>50 tokens) |
| C10 | Entity balance | Balance PER/LOC/ORG distribution |

### Experiment Matrix

| Phase | Experiments | Compute Time |
|-------|-------------|--------------|
| Individual ablation | 50 | ~8 days |
| Combinations | 20 | ~3 days |
| Multi-dataset | 30 | ~5 days |
| **Total** | **100** | **~16 days** |

---

## Connection to Thesis

This paper provides empirical foundation for the thesis proposal:

```
Adaptive Edge NLP Architecture
├── Data Analyzer (detect distribution)
├── Curation Selector (Paper 3 framework)
├── KG Selection (Papers 1-2 findings)
└── K-LBERTO (adapted model)
```

---

## Directory Structure

```
curation_criteria_NER/
├── README.md                      # This file
├── EstrategiaPaper3.md           # Detailed research strategy
├── email_director_seguimiento.md  # Email draft for director
│
├── # Future directories (post-approval)
├── scripts/                       # Curation scripts
├── experiments/                   # Experiment configs
├── results/                       # Experiment results
└── analysis/                      # Analysis outputs
```

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Proposal & Approval | 1-2 weeks | **Current** |
| Preparation | 1 week | Pending |
| Experiments | 2.5 weeks | Pending |
| Multi-dataset | 1 week | Pending |
| Analysis & Writing | 2 weeks | Pending |

---

## Files

| File | Purpose |
|------|---------|
| `EstrategiaPaper3.md` | Complete research strategy in Spanish |
| `email_director_seguimiento.md` | Draft email for director communication |

---

*Created: 2026-01-25*
*Status: PROPOSAL - Awaiting director approval*
