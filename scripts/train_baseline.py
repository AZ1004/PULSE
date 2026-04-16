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
    return parser.parse_args()


def main():
    args = get_args()
    wandb.init(project=args.wandb_project,
               name="baseline_supervised_imagenet",
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

        # ImageNet pretrained ResNet-50, replace head
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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

            try:
                auc = roc_auc_score(all_labels, all_probs,
                                    multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")

            acc = (all_preds == all_labels).mean()
            best_auc = max(best_auc, auc)

            wandb.log({f"baseline/fold{fold+1}/auc": auc,
                       f"baseline/fold{fold+1}/acc": acc,
                       "epoch": epoch})

        fold_metrics.append({"auc": best_auc, "acc": acc})
        print(f"Fold {fold+1} | Best AUC: {best_auc:.4f} | Final Acc: {acc:.4f}")

    aucs = [m["auc"] for m in fold_metrics]
    print(f"\nBaseline AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    wandb.log({"baseline/auc_mean": np.mean(aucs),
               "baseline/auc_std":  np.std(aucs)})
    wandb.finish()


if __name__ == "__main__":
    main()