# scripts/train_baseline.py
# Supervised ImageNet ResNet-50 fine-tuned on BUSI — your ceiling baseline.

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix

from data.dataset import BUSIDataset, CLASS_TO_IDX


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str,
                        default=r"C:\Users\asmaa\Desktop\BUSI\Dataset_BUSI_with_GT")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="PULSE-SSL")
    parser.add_argument("--pretrained", action="store_true", default=True,
                    help="Use ImageNet weights (default). Pass --no-pretrained for random init.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--label_frac", type=float, default=1.0,
                    help="Fraction of training labels to use (e.g. 0.1, 0.25, 1.0)")
    return parser.parse_args()

def compute_full_metrics(labels: np.ndarray, preds: np.ndarray,
                         probs: np.ndarray, n_classes: int) -> dict:
    """Same protocol as evaluate.py's compute_metrics — AUC, accuracy,
    sensitivity, and specificity, macro-averaged across classes (OvR)."""
    acc = (preds == labels).mean()

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    sensitivities, specificities = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn + 1e-8))
        specificities.append(tn / (tn + fp + 1e-8))

    return {
        "auc": auc,
        "acc": acc,
        "sensitivity": np.mean(sensitivities),
        "specificity": np.mean(specificities),
    }

def main():
    args = get_args()
    wandb.init(project=args.wandb_project,
               name=f"baseline_random_init{int(args.label_frac*100)}" if not args.pretrained else f"baseline_supervised_imagenet{int(args.label_frac*100)}",
               config=vars(args))

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])

    dataset    = BUSIDataset(root=args.data_root, image_size=args.image_size)
    labels_all = [s["label"] for s in dataset.samples]
    skf        = StratifiedKFold(n_splits=args.n_folds,
                                 shuffle=True, random_state=42)

    from sklearn.metrics import roc_auc_score, confusion_matrix
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels_all)), labels_all)):

        class TransformSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset; self.transform = transform
            def __len__(self): return len(self.subset)
            def __getitem__(self, i):
                img, label = self.subset[i]
                return self.transform(img), label

        train_loader = DataLoader(
            TransformSubset(Subset(dataset, train_idx), tf),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            TransformSubset(Subset(dataset, val_idx), tf),
            batch_size=args.batch_size, shuffle=False, num_workers=0)
        if args.label_frac < 1.0:
            rng = np.random.default_rng(42 + fold)
            n_keep = max(int(len(train_idx) * args.label_frac), len(CLASS_TO_IDX))
            train_idx = rng.choice(train_idx, size=n_keep, replace=False)

        # ImageNet pretrained ResNet-50, replace head
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.resnet50(weights=weights)
        print(f"[baseline] {'ImageNet-pretrained' if args.pretrained else 'Random init (scratch)'}")
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_IDX))
        model    = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_auc = 0.0
        for epoch in range(args.epochs):
            # Train
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    probs = torch.softmax(
                        model(imgs.to(args.device)), dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.numpy())

            all_labels = np.concatenate(all_labels)
            all_probs  = np.vstack(all_probs)
            all_preds  = all_probs.argmax(axis=1)


            metrics = compute_full_metrics(all_labels, all_preds, all_probs, len(CLASS_TO_IDX))
            best_auc = max(best_auc, metrics["auc"])
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_TO_IDX))))
            

            wandb.log({f"baseline/fold{fold+1}/auc":         metrics["auc"],
                       f"baseline/fold{fold+1}/acc":         metrics["acc"],
                       f"baseline/fold{fold+1}/sensitivity": metrics["sensitivity"],
                       f"baseline/fold{fold+1}/specificity": metrics["specificity"],
                       "epoch": epoch})

        fold_metrics.append(metrics)
        print(f"  Confusion matrix (rows=true, cols=pred) [benign, malignant, normal]:")
        print(f"  {cm}")
        print(f"  Class distribution in this val fold: "
                f"benign={sum(all_labels==0)}, malignant={sum(all_labels==1)}, normal={sum(all_labels==2)}")
        print(f"Fold {fold+1} | AUC: {metrics['auc']:.4f} | Acc: {metrics['acc']:.4f} "
              f"| Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f}")

    # Aggregate across folds — mean ± std for all four metrics
    summary = {}
    for key in ["auc", "acc", "sensitivity", "specificity"]:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = np.mean(vals)
        summary[f"{key}_std"]  = np.std(vals)

    print(f"\nBaseline results ({'ImageNet' if args.pretrained else 'Random init'}, "
          f"{int(args.label_frac*100)}% labels):")
    print(f"  AUC  : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  Acc  : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"  Sens : {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
    print(f"  Spec : {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")

    wandb.log({f"baseline/{k}": v for k, v in summary.items()})
    wandb.finish()


if __name__ == "__main__":
    main()