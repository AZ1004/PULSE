# scripts/evaluate.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision import models
import wandb

from data.dataset import BUSIDataset, CLASS_TO_IDX
from models.simclr import SimCLR


# ── Config ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to SimCLR checkpoint (.pth)")
    parser.add_argument("--data_root",   type=str,
                        default=r"C:\Users\asmaa\Desktop\BUSI\Dataset_BUSI_with_GT")
    parser.add_argument("--mode",        type=str, default="linear_probe",
                        choices=["linear_probe", "fine_tune"])
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n_folds",     type=int, default=5)
    parser.add_argument("--label_fracs", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 1.0],
                        help="Label fractions for efficiency curve")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="PULSE-SSL")
    return parser.parse_args()


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    """
    Compute AUC, accuracy, sensitivity, specificity for 3-class BUSI.
    Sensitivity & specificity are computed per-class (OvR) then macro-averaged.
    """
    acc = (preds == labels).mean()

    # AUC — one-vs-rest macro average
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen if a class is absent in a small fold

    # Per-class sensitivity (recall) and specificity
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    sensitivities, specificities = [], []
    for i in range(len(CLASS_TO_IDX)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn + 1e-8))
        specificities.append(tn / (tn + fp + 1e-8))

    return {
        "acc":         acc,
        "auc":         auc,
        "sensitivity": np.mean(sensitivities),
        "specificity": np.mean(specificities),
    }


# ── Model loading ──────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, device: str) -> nn.Module:
    from scripts.train_ssl_v2 import SimCLREncoder
    """Load SimCLR encoder (SimCLREncoder from train_ssl_v2.py), strip projector."""
    model = SimCLREncoder(backbone="resnet50", proj_hidden=2048, proj_dim=128)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model", ckpt)

    # strict=False: we only need encoder weights, not the projector's
    # BN running stats (and projector isn't used downstream anyway)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_encoder] Missing keys (expected if projector-only): {missing}")
    if unexpected:
        print(f"[load_encoder] Unexpected keys: {unexpected}")

    encoder = model.encoder.to(device)
    return encoder


def build_classifier(encoder: nn.Module, mode: str,
                     n_classes: int, device: str) -> nn.Module:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat_dim = encoder(dummy).flatten(1).shape[-1]

    head = nn.Linear(feat_dim, n_classes).to(device)

    if mode == "linear_probe":
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    return nn.Sequential(encoder, nn.Flatten(), head)


# ── Training loop (downstream) ─────────────────────────────────────────────────

def train_downstream(model: nn.Module, loader: DataLoader,
                     optimizer, criterion, device: str, freeze: bool):
    model.train()
    if freeze:
        # Keep encoder in eval mode even during "training" of the head
        model[0].eval()

    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_downstream(model: nn.Module, loader: DataLoader,
                        device: str) -> tuple:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.vstack(all_probs))


# ── K-fold evaluation ──────────────────────────────────────────────────────────

def run_kfold(args, dataset: BUSIDataset,
              label_frac: float = 1.0) -> dict:
    """
    Run 5-fold CV for a given label fraction.
    Returns mean ± std of all metrics across folds.
    """
    labels_all  = [s["label"] for s in dataset.samples]
    skf         = StratifiedKFold(n_splits=args.n_folds,
                                  shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels_all)), labels_all)):

        # ── Label fraction subsampling ─────────────────────────────────────
        if label_frac < 1.0:
            rng = np.random.default_rng(42 + fold)
            n_keep = max(int(len(train_idx) * label_frac),
                         len(CLASS_TO_IDX))  # at least 1 per class
            train_idx = rng.choice(train_idx, size=n_keep, replace=False)

        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        # Re-attach transform per fold (dataset was built with transform=None)
        train_set = Subset(dataset, train_idx)
        val_set   = Subset(dataset, val_idx)

        # Apply transform via a wrapper
        class TransformSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset    = subset
                self.transform = transform
            def __len__(self):  return len(self.subset)
            def __getitem__(self, i):
                img, label = self.subset[i]
                return self.transform(img), label

        train_loader = DataLoader(TransformSubset(train_set, tf),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(TransformSubset(val_set, tf),
                                  batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # Fresh encoder + head for each fold
        encoder    = load_encoder(args.checkpoint, args.device)
        model      = build_classifier(encoder, args.mode,
                                      len(CLASS_TO_IDX), args.device)
        freeze     = args.mode == "linear_probe"
        params     = filter(lambda p: p.requires_grad, model.parameters())
        optimizer  = torch.optim.Adam(params, lr=args.lr)
        criterion  = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_downstream(model, train_loader, optimizer,
                             criterion, args.device, freeze)

        labels, preds, probs = evaluate_downstream(
            model, val_loader, args.device)
        metrics = compute_metrics(labels, preds, probs)
        fold_metrics.append(metrics)

        print(f"  Fold {fold+1}/{args.n_folds} | "
              f"AUC: {metrics['auc']:.4f} | "
              f"Acc: {metrics['acc']:.4f} | "
              f"Sens: {metrics['sensitivity']:.4f} | "
              f"Spec: {metrics['specificity']:.4f}")

    # Aggregate across folds
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = np.mean(vals)
        summary[f"{key}_std"]  = np.std(vals)

    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    wandb.init(
        project=args.wandb_project,
        name=f"eval_{args.mode}",
        config=vars(args),
    )

    # Build base dataset (no transform — applied per fold inside run_kfold)
    dataset = BUSIDataset(root=args.data_root, image_size=args.image_size,
                          transform=None)
    print(dataset)

    # ── Label efficiency curve ─────────────────────────────────────────────────
    efficiency_table = wandb.Table(
        columns=["label_frac", "auc_mean", "auc_std",
                 "acc_mean", "acc_std",
                 "sensitivity_mean", "sensitivity_std",
                 "specificity_mean", "specificity_std"])

    for frac in args.label_fracs:
        print(f"\n── Label fraction: {int(frac*100)}% ──────────────────────")
        summary = run_kfold(args, dataset, label_frac=frac)

        print(f"  AUC  : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"  Acc  : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
        print(f"  Sens : {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
        print(f"  Spec : {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")

        wandb.log({f"{args.mode}/frac_{int(frac*100)}/{k}": v
                   for k, v in summary.items()})

        efficiency_table.add_data(
            frac,
            summary["auc_mean"],  summary["auc_std"],
            summary["acc_mean"],  summary["acc_std"],
            summary["sensitivity_mean"], summary["sensitivity_std"],
            summary["specificity_mean"], summary["specificity_std"],
        )

    wandb.log({f"{args.mode}/efficiency_table": efficiency_table})
    wandb.finish()
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()