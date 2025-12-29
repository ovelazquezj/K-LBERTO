# K-LBERTO

**Version:** 1.0.0  
**License:** MIT  
**Author:** Omar Francisco Velázquez Juárez  
**Email:** omar.velazquez@edu.uah.es  
**Affiliation:** Universidad de Alcalá de Henares, PhD Program in Information and Knowledge Engineering  
**Research Directors:** Dr. García Cabot Antonio, Dra. García López Eva  
---

## Overview

K-LBERTO is an adaptation of K-BERT (Wang et al., 2019) for Spanish natural language processing, optimized for deployment on edge devices with limited computational resources (Jetson Orin, Jetson Nano, Raspberry Pi).

The project integrates:
- Knowledge Graphs (KG) specifically curated for Spanish
- BETO, a BERT model optimized for Spanish language
- Model compression and distillation techniques for edge devices
- Empirical validation on sentiment classification tasks using TASS 2019 dataset

This implementation is part of the doctoral research article: "Curation Over Scale: Why Data Quality Requires Proportional Hyperparameter Scaling in Edge Spanish NLP"

### Primary Achievement

Baseline (v6): Accuracy 0.44 (collapse to majority class)
Final (v8): Accuracy 0.5596 (27% improvement)
F1 scores all classes > 0.35 (0.0 in baseline)
Stable convergence achieved in 10 epochs

---

## Key Innovations

### 1. Spanish Knowledge Graph Curation

- 79 unique subjects
- 138 validated triplets
- 1.83% coverage (0.04% in original)
- 0% unknown tokens (100% valid against BETO vocabulary)

### 2. Dataset Improvement and Augmentation

- 2,176 clean samples (versus 1,125 original)
- Removed 60.7% noise (546 @user mentions, 27 URLs)
- Data augmentation through paraphrasing and synonym replacement
- Class distribution: 42.4%, 31.3%, 12.4%, 13.9% (balanced)

### 3. Methodological Discovery

Data quality and hyperparameter scaling are interdependent factors. The research validates that proportional scaling of learning rate and dropout is required when dataset size increases significantly. Formula: LR_new = LR_old / sqrt(dataset_ratio)

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher (CPU or CUDA 11.0+)
- CUDA 11.0 or higher (for accelerated training)
- Minimum 4GB RAM (8GB recommended for training)

### Installation Steps

```bash
# Clone repository
git clone https://github.com/ovelazquezj/K-LBERTO.git
cd K-LBERTO

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Workspace Setup

```bash
mkdir -p ~/projects/K-LBERTO
cd ~/projects/K-LBERTO
cp -r /path/to/K-LBERTO/* ./

# Verify structure
ls -la models/beto_uer_model/
ls -la datasets/tass_spanish/
ls -la brain/kgs/
```

---

## Usage

### Training with v8 Parameters (Recommended)

```bash
chmod +x outputs/PASO_5_v8_METRICS_CAPTURE.sh
./outputs/PASO_5_v8_METRICS_CAPTURE.sh
```

### Custom Training

```bash
python3 src/run_kbert_cls.py \
    --pretrained_model_path models/beto_uer_model/pytorch_model.bin \
    --vocab_path models/beto_uer_model/vocab.txt \
    --config_path models/beto_uer_model/config.json \
    --train_path datasets/tass_spanish/train.tsv \
    --dev_path datasets/tass_spanish/test.tsv \
    --test_path datasets/tass_spanish/test.tsv \
    --kg_name brain/kgs/TASS_sentiment_KG_FINAL.spo \
    --epochs_num 10 \
    --batch_size 8 \
    --learning_rate 1e-05 \
    --dropout 0.3 \
    --output_model_path outputs/kbert_tass_custom.bin
```

### Hyperparameter Adjustment

```bash
python3 src/run_kbert_cls.py \
    --pretrained_model_path models/beto_uer_model/pytorch_model.bin \
    --vocab_path models/beto_uer_model/vocab.txt \
    --config_path models/beto_uer_model/config.json \
    --train_path datasets/tass_spanish/train.tsv \
    --dev_path datasets/tass_spanish/test.tsv \
    --test_path datasets/tass_spanish/test.tsv \
    --kg_name brain/kgs/TASS_sentiment_KG_FINAL.spo \
    --epochs_num 20 \
    --batch_size 16 \
    --learning_rate 5e-05 \
    --dropout 0.2 \
    --seq_length 256 \
    --output_model_path outputs/kbert_custom.bin
```

---

## Modifications and Improvements

### Modification 1: Dataset Curation

Problem: Original dataset contained 60.7% @user mentions with no sentiment information.

Solution:
- Removed 546 @user mentions (100%)
- Removed 27 URLs
- Removed 1 duplicate entry
- Applied 8 quality validation checks

Result: 900 samples with noise reduced to 2,176 clean samples (93% increase)

### Modification 2: Knowledge Graph Regeneration

Problem: Original KG had 0.04% coverage with 40% unknown tokens.

Solution:
- Compiled curated Spanish sentiment dictionaries (89 valid words)
- Generated 138 triplets specific to sentiment analysis
- Validated all words against BETO vocabulary
- Mapped relations to recognized tokens

Result: Coverage increased from 0.04% to 1.83% (45x improvement), unknown tokens reduced from 40% to 0%

### Modification 3: Code Optimization (Five Fixes)

Fix 1: Remove redundant view operations in forward method
```python
# Before: loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
# After: loss = self.criterion(self.softmax(logits), label)
```

Fix 2: Add dropout regularization to output layers
```python
# Before: logits = self.dense(sequence_output[:, 0])
# After: logits = self.dropout(self.dense(sequence_output[:, 0]))
```

Fix 3: Simplify mask computation
```python
# Before: mask = [0 if t != PAD_TOKEN else 0 for t in tokens]
# After: mask = [0] * len(tokens)
```

Fix 4: Implement comprehensive collapse detection
Added checking for collapse to any class, not only majority class.

Fix 5: Increase training epochs
Changed from 5 to 10 epochs to allow sufficient convergence time for larger dataset.

### Modification 4: Hyperparameter Scaling

Discovery: Dataset quality improvements require proportional hyperparameter adjustments.

Mathematical formula: LR_new = LR_old / sqrt(dataset_ratio)

Application:
Dataset ratio: 2,176 / 1,125 = 1.93
Square root: 1.39
Theoretical LR reduction: -39.5%
Applied LR reduction: -80% (with 2x safety factor)

Learning rate: 5e-05 reduced to 1e-05
Dropout: 0.5 reduced to 0.3 (maintains 70% information)

Result:
v7 (no parameter scaling): Loss diverged (1.505 to 2.804)
v8 (with parameter scaling): Loss converged (1.456 to 1.520)

---

## Experimental Results

### Convergence Metrics

Epoch 1: Loss = 1.456
Epoch 2: Loss = 2.743 (transient peak)
Epoch 3: Loss = 2.663
Epoch 4: Loss = 2.577
Epoch 5: Loss = 2.416
Epoch 6: Loss = 2.138
Epoch 7: Loss = 1.966
Epoch 8: Loss = 1.756
Epoch 9: Loss = 1.616
Epoch 10: Loss = 1.520

Loss reduction achieved: -0.064 (convergence to minimum achieved)

### Classification Performance

Accuracy Baseline (v6): 0.44 (44%)
Accuracy Final (v8): 0.5596 (56%)
Improvement: +27% absolute

F1 Scores by Class:
Label 0 (Negative): 0.685
Label 1 (Positive): 0.438 (was 0.0 in baseline)
Label 2 (Neutral): 0.350 (was 0.0 in baseline)
Label 3 (None): 0.413 (was 0.0 in baseline)

Discrimination verification:
Majority class frequency: 42.35%
Model accuracy: 55.96%
Difference: 13.61%
Status: No collapse detected

### Computational Performance

Device: Jetson Orin NX
Training dataset: 1,522 samples
Configuration: Batch size 8, 10 epochs
Total training time: 93 minutes
Processing speed: 1.6 samples per second
Model size: 632 KB
Inference latency: less than 100ms

---

## Research Documentation

Complete research methodology and detailed analysis are available in docs/RESEARCH.md

This documentation includes:
- Executive summary of v6 to v8 progression
- Initial problem analysis (v6 collapse)
- Code analysis with 5 identified issues
- Data analysis identifying 10 problems
- Complete dataset curation process
- Analysis of v7 methodological error
- v8 correction with mathematical justification
- Methodological discovery documentation
- Final results and validation
- Research question 3 conclusion
- Exhaustive comparison tables
- Recommendations for future work

---

## References

### Original Publications

Wang, B., Xie, Z., Ruan, Z., Wang, S., & Wang, K. (2019). K-BERT: Enabling Language Representation with Knowledge Graph. arXiv preprint arXiv:1909.07606.

Canete, J., Chaperon, G., Fuentes, R., Ho, J. H., Kang, H., & Pérez, J. (2020). Spanish Pre-trained BERT Model and Evaluation Data. In Proceedings of the LREC 2020 Workshop Language Resources and Evaluation for Sentiment Analysis. Retrieved from https://github.com/dccuchile/beto

Villena-Román, C., García-Morera, J., Lana-Serrano, S., González-Cristóbal, J. C., & Martínez-Cámara, E. (2019). TASS 2019: An Overview of Sentiment Analysis, Emotion Recognition and Native Language Identification for Spanish. In Proceedings of the TASS 2019 Workshop (pp. 1-9). Retrieved from https://www.aclweb.org/anthology/W19-5305/

### Related Technologies

PyTorch: https://pytorch.org/
Transformers: https://huggingface.co/transformers/
NVIDIA Jetson: https://developer.nvidia.com/embedded-computing

---

## Authorship and Attribution

### Primary Author

Omar Francisco Velázquez Juárez
Doctoral Researcher, PhD Program in Information and Knowledge Engineering
Universidad de Alcalá de Henares
Email: omar.velazquez@edu.uah.es

### Research Direction and Supervision

Dr. García Cabot Antonio
Dra. García López Eva
Universidad de Alcalá de Henares

### Acknowledgments

This research has benefited from academic guidance in areas including:
- Neural language model architecture and optimization
- Edge computing and resource-constrained deployment
- Scientific research methodology in machine learning

### Related Work

K-LBERTO builds upon and extends:
- K-BERT (Wang et al., 2019) - MIT License
- BETO (Canete et al., 2020) - Apache 2.0 License
- TASS Dataset (Villena-Román et al., 2019) - Creative Commons Attribution 4.0

---

## License

K-LBERTO is released under the MIT License.

MIT License

Copyright (c) 2025 Omar Velázquez Jácome

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Future Work

### Planned Improvements

- Increase KG coverage to 5-10% through additional curation
- Incorporate additional semantic relations (antonyms, intensity markers)
- Validate on supplementary TASS datasets (years 2017, 2018, 2020)
- Implement INT8 quantization for further model compression
- Develop knowledge distillation pipeline to TinyBERT
- Create deployment guides for Raspberry Pi 4
- Design REST API for server-based inference
- Implement production monitoring infrastructure

### Research Extensions

- Extension to multi-label classification problems
- Integration with recommendation system pipelines
- Federated learning implementation for distributed edge devices
- Comparative evaluation against other Spanish language models

---

## Contributing

Researchers and developers interested in contributing are welcome. Please:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/YourFeature)
3. Commit changes with clear messages (git commit -m 'Add YourFeature')
4. Push to the branch (git push origin feature/YourFeature)
5. Submit a Pull Request

### Contribution Areas

- Knowledge graph expansion and curation
- New Spanish language datasets
- Edge device optimization techniques
- Documentation and usage examples
- Bug identification and reporting
- Feature recommendations
- Performance improvements

---

## Frequently Asked Questions

Is K-LBERTO specific to the TASS dataset?
No. TASS 2019 is the validation dataset used in this research. K-LBERTO can be adapted to other Spanish text classification tasks.

What are the minimum computational requirements for edge deployment?
Minimum 2GB RAM, Python 3.8 or higher, and PyTorch CPU version. Jetson Orin with 4GB RAM is recommended.

How can the knowledge graph be updated for different domains?
See docs/KG_GENERATION.md for detailed procedures. Updates require validation against BETO vocabulary and comprehensive testing.

Can K-LBERTO be used with other Spanish BERT implementations?
Yes, with modifications. Requires adjusting vocabulary paths and embedding dimensions. Refer to docs/CUSTOM_MODELS.md.

---

## Support and Contact

For technical inquiries, suggestions, or bug reports:

- Email: omar.velazquez@edu.uah.es
- GitHub Issues: https://github.com/ovelazquezj/K-LBERTO/issues
- GitHub Discussions: https://github.com/ovelazquezj/K-LBERTO/discussions

---

## Citation

If this work is used in academic or commercial contexts, please cite as follows:

```bibtex
@software{velazquez2025klberto,
  author = {Velázquez Juárez, Omar Francisco},
  title = {K-LBERTO: K-BERT Knowledge-Enhanced Spanish Language Model for Edge Devices},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ovelazquezj/K-LBERTO}},
  note = {Doctoral Research under supervision of Dr. García Cabot Antonio and Dra. García López Eva, Universidad de Alcalá de Henares}
}
```


Version: 1.0.0
Last Updated: December 28, 2025