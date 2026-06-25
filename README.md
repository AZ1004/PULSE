# PULSE 🫀
### Pretraining Ultrasound representations via Label-free Self-supervised Encoders

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-yellow.svg)](https://wandb.ai)

> **How far does contrastive self-supervision actually get you on breast ultrasound?**

This repository contains the code, configurations, and results for a label-efficiency study of SimCLR self-supervised pretraining on breast ultrasound classification, benchmarked against random-initialization and ImageNet-supervised baselines. The accompanying paper, *"How Far Does Contrastive Self-Supervision Get You on Breast Ultrasound? A Label-Efficiency Study on BUSI,"* was submitted to the MICCAI 2026 Deep-Breath workshop.

---

## Key finding

Self-supervised learning is often assumed to provide its largest advantage when labeled data is scarce. **We find the opposite on BUSI.** Both random-initialization and ImageNet-supervised baselines are essentially flat across the entire labeled-data range we test (10%–100%) — they reach their performance ceiling almost immediately, with very little data. **SimCLR is the only method that shows a conventional rising label-efficiency curve**, meaning it is the *most*, not least, label-hungry method in our comparison.

| Method | AUC @ 10% labels | AUC @ 100% labels | Shape |
|---|---|---|---|
| ImageNet supervised | 0.962 | 0.958 | flat (saturates immediately) |
| Random init | 0.863 | 0.876 | flat (saturates immediately) |
| SimCLR (fine-tuned) | 0.747 | 0.880 | **rises steadily** |
| SimCLR (linear probe) | 0.705 | 0.803 | **rises steadily** |

We argue this reflects a property of the BUSI dataset itself — strong, low-level, class-discriminative signal that conventional supervised learning exploits immediately, even with very little data — rather than a failure of contrastive pretraining. See the paper for full discussion and a proposed mechanism.

![Label efficiency curve](PULSE/efficiency_curve.png)

---

## Repository structure

```
PULSE/
├── data/
│   └── dataset.py            # BUSIDataset, BUSIDatasetSSL — handles multi-mask structure
├── augmentations/
│   └── ssl_transforms.py     # SimCLRAugment — dual-view augmentation pipeline
├── models/
│   └── simclr.py             # SimCLR architecture (encoder + projection head)
├── losses/
│   └── nt_xent.py            # NT-Xent contrastive loss, unit tested
├── scripts/
│   ├── train_ssl_v2.py       # SimCLR pretraining (AMP, grad accumulation, LARS, warm-started probe)
│   ├── train_baseline.py     # Random-init and ImageNet-supervised baselines, label-fraction sweep
│   ├── evaluate.py           # 5-fold downstream evaluation: linear probe + fine-tuning, all metrics
│   └── train_mae.py          # MAE pretraining (implemented, scoped out of current paper — see below)
├── figures/
│   └── efficiency_curve.png  # Main results figure
├── notebooks/
│   └── 02_augmentation_vis.ipynb
├── checkpoints/               # Not tracked in git — see Reproducing Results
└── README.md
```

---

## Results

All results are mean ± std over 5-fold stratified cross-validation, computed at the final training epoch of each fold (no checkpoint selection by validation performance, to avoid optimistic bias).

### Main comparison (100% labels)

| Method | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Random init | 0.876 ± 0.022 | 0.744 ± 0.026 | 0.687 ± 0.033 | 0.847 ± 0.011 |
| SimCLR (linear probe) | 0.803 ± 0.036 | 0.665 ± 0.030 | 0.574 ± 0.038 | 0.788 ± 0.016 |
| SimCLR (fine-tuned) | 0.880 ± 0.033 | 0.744 ± 0.069 | 0.746 ± 0.047 | 0.864 ± 0.029 |
| **ImageNet supervised** | **0.958 ± 0.010** | **0.860 ± 0.023** | **0.837 ± 0.037** | **0.914 ± 0.016** |

### Full label-efficiency grid (AUC)

| Method | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Random init | 0.863 ± 0.029 | 0.874 ± 0.031 | 0.868 ± 0.023 | 0.876 ± 0.022 |
| SimCLR (linear probe) | 0.705 ± 0.030 | 0.761 ± 0.033 | 0.788 ± 0.033 | 0.803 ± 0.036 |
| SimCLR (fine-tuned) | 0.747 ± 0.038 | 0.788 ± 0.038 | 0.868 ± 0.021 | 0.880 ± 0.033 |
| ImageNet supervised | 0.962 ± 0.007 | 0.961 ± 0.013 | 0.962 ± 0.008 | 0.958 ± 0.010 |

---

## Dataset

We use the **[BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)** (Breast Ultrasound Images), a publicly available benchmark containing 780 images across three classes:

| Class | Count |
|---|---|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |

No PhysioNet account or data use agreement is required. `BUSIDataset` in `data/dataset.py` handles BUSI's multi-mask file structure (some lesions have multiple annotated mask regions, e.g. `benign (100)_mask.png` and `benign (100)_mask_1.png`) by merging masks with a logical OR.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/AZ1004/PULSE.git
cd PULSE
pip install -r requirements.txt
```

### 2. Download BUSI

Download from Kaggle (link above) and place under `data/BUSI/Dataset_BUSI_with_GT/`, or pass a custom path via `--data_root` / `--data_dir` to any script below.

### 3. Pretrain SimCLR

```bash
python scripts/train_ssl_v2.py --data_dir data/BUSI --epochs 200 --batch_size 32 --accum_steps 8
```

### 4. Run baselines

```bash
python -m scripts.train_baseline --no-pretrained --label_frac 1.0   # random init
python -m scripts.train_baseline --label_frac 1.0                    # ImageNet supervised
```

Both support `--label_frac` (e.g. `0.1`, `0.25`, `0.5`, `1.0`) to reproduce the full efficiency curve.

### 5. Evaluate SimCLR (linear probe + fine-tuning)

```bash
python scripts/evaluate.py --checkpoint checkpoints/simclr/simclr_best.pt --mode linear_probe
python scripts/evaluate.py --checkpoint checkpoints/simclr/simclr_best.pt --mode fine_tune
```

Use `--label_fracs` to control which fractions are evaluated (default: `0.1 0.25 0.5 1.0`).

All experiments are tracked with [Weights & Biases](https://wandb.ai) under the **PULSE-SSL** project. Set your API key with `wandb login` before training.

---

## Why MAE is in the repo but not the paper

A full Masked Autoencoder (MAE) implementation is included in `scripts/train_mae.py` — ViT-Small encoder, lightweight decoder, configurable masking ratio — and was originally planned as a second self-supervised paradigm to compare against SimCLR. Given the workshop deadline and the fact that SimCLR alone revealed a clear, well-supported, and somewhat unexpected finding, we made the deliberate choice to scope the paper to a single, rigorously evaluated method rather than spread evaluation across two paradigms under time pressure. MAE on BUSI remains future work — see the Discussion section of the paper.

---

## Key hyperparameters (SimCLR)

| Parameter | Value |
|---|---|
| Encoder | ResNet-50 |
| Projection head | 2048 → 2048 → 128, BN on output (SimCLR v2) |
| Epochs | 200 |
| Effective batch size | 256 (via gradient accumulation) |
| Optimizer | LARS |
| LR schedule | Cosine + 10-epoch warmup |
| Temperature τ | 0.1 |
| Augmentations | Random resized crop (0.5–1.0), horizontal flip, color jitter, Gaussian blur — no hue jitter (not meaningful for grayscale ultrasound) |

Full configuration in `scripts/train_ssl_v2.py`.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@misc{zizaan2026pulse,
  author       = {Asma Zizaan},
  title        = {How Far Does Contrastive Self-Supervision Get You on Breast Ultrasound? A Label-Efficiency Study on BUSI},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/AZ1004/PULSE}
}
```

---

## Author

**Asma Zizaan** — PhD Candidate in AI, Mohammed VI Polytechnic University
[ORCID](https://orcid.org/0000-0002-9591-6550) · [Google Scholar](https://scholar.google.com) · asma.zizaan@um6p.ma

*This project is part of ongoing doctoral research on deep learning for breast cancer screening, supervised by Prof. Ali Idri.*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
