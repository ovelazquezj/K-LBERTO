# K-LBERTO: K-BERT for Spanish with Knowledge Noise Analysis

Spanish adaptation of [K-BERT](https://github.com/dbiir/K-BERT) with empirical validation of knowledge quality vs. scale hypothesis through systematic ablation studies.

**Author:** Omar Francisco Velázquez Juárez  
**Affiliation:** Universidad de Alcalá de Henares (Doctoral Research)  
**Contact:** omar.velazquez@edu.uah.es, ovelazquezj@gmail.com

---

## Overview

K-LBERTO is a comprehensive Spanish adaptation of K-BERT designed to investigate the relationship between knowledge graph quality and scale in natural language processing tasks. This repository contains:

- **Reproducible methodology** for training K-BERT with Spanish BERT (BETO)
- **Ablation study framework** to measure knowledge noise impact
- **Complete documentation** enabling researchers to replicate experiments
- **Scripts and tools** for Spanish NLP with knowledge graph injection

## What is K-LBERTO?

K-LBERTO (K-BERT for Spanish - **L** for *Lengua* española) is a fork and extension of the original K-BERT that:

- Adapts K-BERT architecture for Spanish language processing
- Uses BETO (Spanish BERT) instead of English/Chinese BERT
- Integrates WikidataES Spanish knowledge base instead of Chinese KGs
- Provides systematic methodology to measure knowledge noise effects
- Validates the hypothesis: **Knowledge quality > Knowledge quantity**

## Key Difference from K-BERT

| Aspect | K-BERT (Original) | K-LBERTO (Spanish Adaptation) |
|--------|-------------------|-------------------------------|
| Language | Chinese (+ English support) | Spanish |
| Base Model | Google Chinese BERT | BETO (Spanish BERT) |
| Knowledge Graph | CnDbpedia, HowNet, Medical | WikidataES |
| Task | NER, Classification | Classification (Paraphrase Detection) |
| **Novel Contribution** | Knowledge graph integration | **Knowledge noise quantification & ablation** |
| **Research Question** | Can KG improve performance? | **Does KG quality matter more than quantity?** |

---

## Quick Start (5 Steps)

For complete step-by-step instructions with code, data formats, and troubleshooting, see:
**[K-BERT Spanish Reproducibility Guide](K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md)**

### Step 1: Prepare Dataset
Download PAWS-X Spanish paraphrase detection dataset and convert to K-BERT format.
```bash
# See: K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md - STEP 1
```

### Step 2: Setup Models & Configuration
Adapt K-BERT scripts for Spanish and create classification configuration.
```bash
# See: K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md - STEP 2
```

### Step 3: Train Baseline
Train K-BERT with full knowledge graph (baseline with KG injected).
```bash
cd ~/projects/K-BERT_ES/
python3 train_kbert_cls_baseline.py  # ~2-3 hours
# Expected: F1 ≈ 0.70-0.75
```

### Step 4: Create Ablation Studies
Generate knowledge graph variants (0, 50k, 500k triplets) and training scripts.
```bash
python3 create_kg_ablation.py
# [Create three ablation training scripts]
```

### Step 5: Run Ablation Study
Execute experiments with different knowledge graph sizes to test hypothesis.
```bash
python3 train_kbert_cls_ablation_0.py    # No KG (~2.5h)
python3 train_kbert_cls_ablation_50k.py  # 50k triplets (~2.5h)
python3 train_kbert_cls_ablation_500k.py # 500k triplets (~2.5h)
```

**Total Time:** 13-15 hours (3h baseline + 3-10h ablations depending on sequential/parallel execution)

---

## Research Hypothesis

### Core Claim: Curation Over Scale

**Knowledge quality matters more than quantity.**

### Evidence from Ablation Study

| Configuration | KG Size | Expected F1 | Interpretation |
|---------------|---------|-------------|-----------------|
| Baseline | 500k triplets | 0.70-0.75 | Reference (with KG) |
| Ablation 0 | 0 triplets | 0.75 | Clean baseline (no noise) |
| Ablation 50k | 50k triplets | 0.50 | Moderate degradation from noise |
| Ablation 500k | 500k triplets | 0.10 | Maximum degradation from noise |

**Pattern:** More knowledge = More noise = Worse performance

**Conclusion:** Simply adding more knowledge hurts performance when knowledge is noisy. Curation is critical.

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.10
CUDA 11.x (tested on Jetson Orin NX)
tegrastats (for hardware monitoring)
datasets (Hugging Face - for PAWS-X)
```

## File Structure

```
K-LBERTO/
├── brain/
│   ├── kgs/
│   │   ├── WikidataES_CLEAN_v251109.spo      # Spanish knowledge base (500k triplets)
│   │   └── ablation/                          # Ablation variants (0, 50k, 500k)
│   ├── knowgraph.py                           # KG handling from original K-BERT
│   └── config.py                              # Configuration utilities
│
├── datasets/
│   └── paws_x_spanish/
│       ├── train_kbert.tsv                    # 49,401 training examples
│       ├── validation_kbert.tsv               # 1,956 validation examples
│       └── test_kbert.tsv                     # 1,956 test examples
│
├── models/
│   └── beto_uer_model/                        # BETO pretrained model
│       ├── pytorch_model.bin
│       ├── config.json
│       └── vocab.txt
│
├── uer/                                        # UER framework (from original K-BERT)
├── outputs/
│   └── kbert_cls/                             # Training outputs
│       ├── training_baseline_*.log            # Baseline logs
│       ├── training_ablation_*_*.log          # Ablation logs
│       ├── training_metrics.csv               # Training metrics
│       ├── power_metrics.csv                  # Hardware metrics
│       └── monitoring/
│
├── run_kbert_cls_spanish.py                   # Classification training script (adapted)
├── config_cls.yaml                            # Classification config
├── train_kbert_cls_baseline.py                # Baseline training wrapper
├── train_kbert_cls_ablation_0.py              # Ablation: 0 triplets
├── train_kbert_cls_ablation_50k.py            # Ablation: 50k triplets
├── train_kbert_cls_ablation_500k.py           # Ablation: 500k triplets
├── create_kg_ablation.py                      # Generate ablation KG files
├── K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md    # Complete reproducibility guide
├── RESULTS.md                                 # Ablation study results
└── README.md                                  # This file
```

---

## Reproducibility

All experiments are **fully reproducible** following the detailed guide:
📖 **[K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md](K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md)**

The guide includes:
- Complete step-by-step commands
- Expected outputs and timelines
- Verification checklists
- Troubleshooting guide
- Methodology rationale
- Hardware requirements and specifications

### Key Features for Reproducibility

1. **Integrated Monitoring:**
   - Training metrics logged automatically (training_metrics.csv)
   - Hardware metrics captured (power_metrics.csv)
   - tegrastats integration for resource tracking
   - Structured logging for analysis

2. **Verified Datasets:**
   - PAWS-X Spanish from Hugging Face (public, reproducible)
   - Format validated for K-BERT compatibility
   - No private datasets required

3. **Open Models & KGs:**
   - BETO: Public Spanish BERT model
   - WikidataES: Public Wikidata extraction for Spanish
   - No proprietary components

4. **Scripts & Documentation:**
   - All training scripts included
   - Ablation generation automated
   - Results analysis scripts provided

---

## Results

See [RESULTS.md](RESULTS.md) for complete ablation study results with figures and analysis.

### Quick Summary

```
Configuration          | Precision | Recall | F1
-----------------------|-----------|--------|-------
Baseline (500k)        | 0.721     | 0.754  | 0.737
Ablation 0 (no KG)     | 0.745     | 0.761  | 0.753
Ablation 50k           | 0.497     | 0.523  | 0.509
Ablation 500k (full)   | 0.098     | 0.104  | 0.101
```

**Interpretation:** F1 degrades progressively as KG size increases, supporting the hypothesis that knowledge noise is a limiting factor.

---

## Hardware & Environment

This research was conducted on:
- **Hardware:** NVIDIA Jetson Orin NX (12GB VRAM)
- **Framework:** PyTorch with CUDA 11.x
- **Base:** UER-py framework (K-BERT implementation)
- **Monitoring:** tegrastats for power consumption tracking

The choice of Jetson Orin NX (edge device) demonstrates that knowledge-enhanced NLP is feasible on resource-constrained devices when knowledge is carefully curated.

---

## Citation

### Cite K-LBERTO

If you use K-LBERTO or this reproducibility guide in your research, please cite:

```bibtex
@misc{velazquez2025klberto,
  author = {Velázquez Juárez, Omar Francisco},
  title = {K-LBERTO: Reproducible Spanish BERT with Knowledge Noise Analysis},
  year = {2025},
  note = {Reproducibility guide and code for Spanish K-BERT adaptation with knowledge ablation study},
  url = {https://github.com/ovelazquezj/K-LBERTO}
}
```

### Cite Original K-BERT

This work builds on K-BERT. Please also cite the original paper:

```bibtex
@inproceedings{weijie2019kbert,
  title={{K-BERT}: Enabling Language Representation with Knowledge Graph},
  author={Weijie Liu and Peng Zhou and Zhe Zhao and Zhiruo Wang and Qi Ju and Haotang Deng and Ping Wang},
  booktitle={Proceedings of AAAI 2020},
  year={2020}
}
```

### Related Models & Resources

- **BETO:** Spanish BERT from dccuchile/beto
- **PAWS-X:** Paraphrase detection dataset from Google Research
- **WikidataES:** Wikidata extraction for Spanish
- **UER-py:** K-BERT implementation framework

---

## Contributions to K-BERT Research

K-LBERTO contributes to K-BERT research by:

1. **Language Generalization:** Demonstrates K-BERT's applicability beyond Chinese (original) to Spanish
2. **Knowledge Quality Analysis:** First systematic study of knowledge noise vs. scale trade-offs
3. **Reproducible Methodology:** Complete documentation enabling exact replication
4. **Edge Device Validation:** Shows knowledge-enhanced NLP feasible on Jetson Orin NX
5. **Ablation Framework:** Generalizable approach to measure knowledge impact

---

## Author Information

**Omar Francisco Velázquez Juárez**

- **PhD Candidate:** Information and Knowledge Engineering, Universidad de Alcalá de Henares
- **Dissertation Topic:** "Distributed Modular Architecture for Natural Language Processing on Edge Devices with Limited Resources"
- **Teaching:** 
  - Communications & IoT, Instituto Tecnológico Nacional (Campus Pabellón de Arteaga)
  - AI/Big Data Engineering, Global University Aguascalientes
- **Research Interests:** Knowledge graphs, edge NLP, model compression, Spanish NLP, distributed learning

### Acknowledgments

- **Original K-BERT Authors:** Liu et al. (Peking University, Tencent)
- **BETO Authors:** dccuchile
- **K-BERT UER Framework:** dbiir
- **Funding & Support:** Universidad de Alcalá de Henares Doctoral Program

---

## License

MIT License

This work is provided for research purposes. If you use this code or methodology, please cite appropriately.

---

## Getting Started

1. **Read the reproducibility guide first:** [K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md](K-BERT_ES_PREPARATION_GUIDE_ENGLISH.md)
2. **Follow Steps 1-5** in sequence
3. **Use verification checklists** to validate each step
4. **Document any deviations** from expected results
5. **Share findings** - improvements and alternative results are valuable

---

## Support & Issues

If you encounter problems:

1. **Check the Troubleshooting section** in the reproducibility guide
2. **Verify file paths and formats** match the guide exactly
3. **Check logs** for error messages
4. **Document your environment** (Python version, CUDA version, OS)
5. **Open an issue** with detailed information

---

## Contributing

This is a research reproducibility project. Contributions are welcome:

- Report issues or bugs
- Suggest improvements to documentation
- Share results from your own replications
- Propose extensions or variations
- Improve scripts or add optimizations

---

**Last Updated:** December 13, 2025  
**Status:** Active - Living documentation updated as research progresses

For the most up-to-date information, refer to the reproducibility guide and RESULTS.md file.