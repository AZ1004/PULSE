# PULSE 🫀
### Pretraining Ultrasound representations via Label-free Self-supervised Encoders

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-yellow.svg)](https://wandb.ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/demo.ipynb)

> **Can self-supervised pretraining close the performance gap of fully supervised models on breast ultrasound — without a single label during pretraining?**

This repository provides a rigorous, reproducible comparison of two self-supervised learning paradigms — **contrastive** (SimCLR) and **generative** (Masked Autoencoders / MAE) — for learning representations from breast ultrasound images. We evaluate downstream classification performance under varying label budgets and analyze what each method actually learns using UMAP and Grad-CAM.

---

## Why this matters

Annotating medical images is expensive, time-consuming, and requires scarce clinical expertise. Self-supervised learning offers a path to high-quality representations from *unlabeled* data — but most SSL research is validated on natural images. This project asks whether those gains transfer to ultrasound, and which pretraining paradigm is best suited to the clinical setting.

---

## Results

> *Results will be filled in as experiments complete. Table shows format.*

| Method | Pretraining | Linear Probe AUC | Fine-tuned AUC | AUC @ 10% labels |
|---|---|---|---|---|
| Random init | — | — | — | — |
| ImageNet supervised | Supervised | — | — | — |
| **SimCLR** (ours) | Contrastive | — | — | — |
| **MAE** (ours) | Generative | — | — | — |

All results are mean ± std over 3 seeds and 5-fold cross-validation. Statistical significance tested with Wilcoxon signed-rank test (α = 0.05).

---

## Method overview

```
┌─────────────────────────────────────────────────────────────┐
│                     PULSE pipeline                          │
│                                                             │
│  Unlabeled BUSI images                                      │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │         │                                              │
│  SimCLR     MAE                                             │
│  (ResNet-50) (ViT-Small)                                    │
│  NT-Xent    Masked patch                                    │
│  loss       reconstruction                                  │
│    │         │                                              │
│    └────┬────┘                                              │
│         │                                                   │
│    Pretrained encoder                                       │
│         │                                                   │
│    ┌────┴──────────┐                                        │
│    │               │                                        │
│  Linear probe   Fine-tune                                   │
│  (frozen enc.)  (full enc.)                                 │
│    │               │                                        │
│   AUC, sensitivity, specificity                             │
└─────────────────────────────────────────────────────────────┘
```

**SimCLR** learns by maximizing agreement between two augmented views of the same image via NT-Xent contrastive loss.

**MAE** learns by masking 75% of image patches and training a ViT encoder + lightweight decoder to reconstruct the missing pixels.

Both encoders are then evaluated with (a) a frozen linear probe and (b) full fine-tuning, across multiple label fractions.

---

## Dataset

We use the **[BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)** (Breast Ultrasound Images), a publicly available benchmark containing 780 images across three classes:

| Class | Count |
|---|---|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |

No PhysioNet account or data use agreement is required. Download instructions in [`data/README.md`](data/README.md).

---

## Repository structure

```
PULSE/
├── configs/                  # Hydra config files
│   ├── simclr.yaml
│   ├── mae.yaml
│   └── downstream.yaml
├── data/
│   ├── README.md             # Download instructions
│   └── dataset.py            # PyTorch Dataset class
├── models/
│   ├── simclr.py             # SimCLR encoder + projection head
│   ├── mae.py                # MAE encoder + decoder
│   └── downstream.py         # Linear probe + fine-tune heads
├── losses/
│   ├── nt_xent.py            # NT-Xent contrastive loss
│   └── mae_loss.py           # Masked patch reconstruction loss
├── augmentations/
│   └── ssl_transforms.py     # Dual-view augmenter for SimCLR, MAE masking
├── experiments/
│   ├── baseline_supervised/
│   ├── simclr/
│   └── mae/
├── notebooks/
│   ├── 01_eda.ipynb          # Dataset exploration
│   ├── 02_augmentation_vis.ipynb
│   └── demo.ipynb            # ← Colab-ready inference demo
├── scripts/
│   ├── train_simclr.py
│   ├── train_mae.py
│   └── evaluate.py
├── REPRODUCE.md              # Exact commands to reproduce all results
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/AZ1004/PULSE.git
cd PULSE
conda env create -f environment.yml
conda activate pulse
```

### 2. Download data

```bash
# Instructions and script in data/README.md
python data/download.py
```

### 3. Train SimCLR

```bash
python scripts/train_simclr.py --config-name simclr
```

### 4. Train MAE

```bash
python scripts/train_mae.py --config-name mae
```

### 5. Evaluate (linear probe + fine-tuning)

```bash
python scripts/evaluate.py --checkpoint experiments/simclr/best.ckpt --mode linear_probe
python scripts/evaluate.py --checkpoint experiments/simclr/best.ckpt --mode fine_tune
```

All experiments are tracked with [Weights & Biases](https://wandb.ai). Set your API key with `wandb login` before training.

---

## Reproducing results

See [`REPRODUCE.md`](REPRODUCE.md) for step-by-step commands to go from raw data to the final results table, including exact hyperparameters and random seeds.

---

## Key hyperparameters

| Parameter | SimCLR | MAE |
|---|---|---|
| Encoder | ResNet-50 | ViT-Small |
| Epochs | 200 | 200 |
| Batch size | 256 | 256 |
| Optimizer | LARS | AdamW |
| LR schedule | Cosine + warmup | Cosine + warmup |
| Temperature τ | 0.07 | — |
| Masking ratio | — | 0.75 |
| Projection head | 2048 → 512 → 128 | — |

All configs live in `configs/` and are fully reproducible via Hydra.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@misc{zizaan2025pulse,
  author       = {Asma Zizaan},
  title        = {PULSE: Pretraining Ultrasound Representations via Label-free Self-supervised Encoders},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/AZ1004/PULSE}
}
```

---

## Author

**Asma Zizaan** — PhD Candidate in AI, Mohammed VI Polytechnic University
[ORCID](https://orcid.org/0000-0002-9591-6550) · [Google Scholar](https://scholar.google.com) · asma.zizaan@um6p.ma

*This project is part of ongoing doctoral research on deep learning for breast cancer screening.*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
