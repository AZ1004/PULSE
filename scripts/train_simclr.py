"""
train_simclr.py — PULSE Project
================================
SimCLR self-supervised pretraining on breast ultrasound images (BUSI dataset).

Usage
-----
  # Full pretraining
  python train_simclr.py --data_dir data/BUSI --epochs 200 --batch_size 256

  # Quick smoke test (1 epoch, tiny batch)
  python train_simclr.py --data_dir data/BUSI --epochs 1 --batch_size 32 --smoke_test

  # Resume from checkpoint
  python train_simclr.py --data_dir data/BUSI --resume checkpoints/simclr_epoch_100.pt

  # Linear evaluation only (freeze encoder)
  python train_simclr.py --data_dir data/BUSI --eval_only --ckpt checkpoints/simclr_best.pt

Architecture
------------
  Encoder  : ResNet-50 (backbone, torchvision pretrained weights optional)
  Projector: 2-layer MLP  →  128-d unit-hypersphere
  Loss     : NT-Xent (temperature-scaled cross-entropy)
  Eval     : Linear probe trained on frozen encoder every `eval_every` epochs
"""

import argparse
import os
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models

import wandb

# ── Local PULSE modules ────────────────────────────────────────────────────────
from data.dataset import BUSIDataset          # returns (view1, view2, label)

from losses.nt_xent import NTXentLoss          # your existing NT-Xent implementation


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PULSE — SimCLR pretraining")

    # Paths
    p.add_argument("--data_dir",    type=str, default="data/BUSI")
    p.add_argument("--ckpt_dir",    type=str, default="checkpoints/simclr")
    p.add_argument("--resume",      type=str, default=None,  help="Path to checkpoint .pt")
    p.add_argument("--ckpt",        type=str, default=None,  help="Encoder ckpt for eval_only")

    # Model
    p.add_argument("--backbone",    type=str, default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--proj_dim",    type=int, default=128,   help="Projector output dim")
    p.add_argument("--proj_hidden", type=int, default=2048,  help="Projector hidden dim")

    # SSL training
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=0.6,  help="Base LR (scaled by batch/32)")
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.5,  help="NT-Xent temperature τ")
    p.add_argument("--warmup_epochs",type=int,  default=10)
    p.add_argument("--min_lr",      type=float, default=0.0)

    # Linear eval
    p.add_argument("--eval_every",  type=int, default=20,    help="Run linear probe every N epochs")
    p.add_argument("--eval_epochs", type=int, default=100,   help="Epochs for linear probe training")
    p.add_argument("--eval_lr",     type=float, default=0.1)
    p.add_argument("--eval_only",   action="store_true")

    # Infra
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--smoke_test",  action="store_true", help="1-epoch sanity check")
    p.add_argument("--wandb_project", type=str, default="PULSE")
    p.add_argument("--wandb_run",   type=str, default=None)
    p.add_argument("--no_wandb",    action="store_true")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model
# ══════════════════════════════════════════════════════════════════════════════

class SimCLREncoder(nn.Module):
    """
    ResNet backbone + 2-layer MLP projector.

    Forward returns (h, z):
      h : [B, feat_dim]  — encoder representations (used for linear probe)
      z : [B, proj_dim]  — L2-normalised projector outputs (used for NT-Xent)
    """

    def __init__(self, backbone: str = "resnet50", proj_hidden: int = 2048,
                 proj_dim: int = 128):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        base = getattr(models, backbone)(weights=None)
        self.feat_dim = base.fc.in_features

        # Strip the classification head; keep everything up to the avg-pool
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # → [B, feat_dim, 1, 1]

        # ── Projector (SimCLR v1 style: Linear → BN → ReLU → Linear) ────────
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)          # [B, feat_dim]
        z = self.projector(h)                   # [B, proj_dim]
        z = F.normalize(z, dim=1)               # unit hypersphere
        return h, z


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimiser & LR schedule
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """LARS is ideal for SimCLR; fall back to SGD if not installed."""
    scaled_lr = args.lr * args.batch_size / 64.0

    try:
        from torch.optim import SGD
        from torch_optimizer import LARS           # pip install torch-optimizer
        optimizer = LARS(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
        print(f"[optim] LARS  lr={scaled_lr:.4f}")
    except ImportError:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        print(f"[optim] SGD (LARS not found)  lr={scaled_lr:.4f}")

    return optimizer


def cosine_lr_with_warmup(optimizer: torch.optim.Optimizer, epoch: int,
                           args: argparse.Namespace):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`.
    Called once per epoch (not per step).
    """
    scaled_lr = args.lr * args.batch_size / 64.0

    if epoch < args.warmup_epochs:
        lr = scaled_lr * (epoch + 1) / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (scaled_lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════════════════════
# 4. Training loop — one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: SimCLREncoder,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: NTXentLoss,
                    device: torch.device,
                    epoch: int,
                    args: argparse.Namespace,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> dict:

    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        # BUSIDataset returns (view1, view2, label) — labels unused here
        view1, view2, _ = batch
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss  = criterion(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss  = criterion(z1, z2)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0:
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss.item():.4f}")

    avg_loss  = total_loss / len(loader)
    elapsed   = time.time() - t0
    throughput = len(loader.dataset) / elapsed

    return {
        "train/loss":       avg_loss,
        "train/epoch":      epoch,
        "train/throughput": throughput,   # samples/sec
        "train/time_s":     elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Linear evaluation (frozen encoder)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(encoder: SimCLREncoder, loader: DataLoader,
                     device: torch.device):
    """Extract h (pre-projector) representations for the whole split."""
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        if  len(batch) == 3:
            imgs, _, ys = batch        # view1 only; augmentation off in eval mode
        else:
            imgs, ys = batch    
        h, _ = encoder(imgs.to(device))
        feats.append(h.cpu())
        labels.append(ys)
    return torch.cat(feats), torch.cat(labels)


def linear_eval(encoder: SimCLREncoder,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                args: argparse.Namespace,
                epoch: int) -> dict:
    """
    Train a single linear layer on top of frozen encoder features.
    Returns top-1 accuracy on val split.
    """
    print("[linear eval] Extracting features…")
    X_tr, y_tr = extract_features(encoder, train_loader, device)
    X_val, y_val = extract_features(encoder, val_loader, device)

    num_classes = len(y_tr.unique())
    head = nn.Linear(encoder.feat_dim, num_classes).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=args.eval_lr, weight_decay=1e-4)

    X_tr  = X_tr.to(device)
    y_tr  = y_tr.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    head.train()
    for ep in range(args.eval_epochs):
        logits = head(X_tr)
        loss   = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        preds = head(X_val).argmax(dim=1)
        acc   = (preds == y_val).float().mean().item()

    print(f"[linear eval @ epoch {epoch}]  val_acc={acc:.4f}")
    return {"eval/linear_acc": acc, "eval/epoch": epoch}


# ══════════════════════════════════════════════════════════════════════════════
# 6. Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: SimCLREncoder, optimizer, scaler, epoch: int,
                    best_loss: float, ckpt_dir: str, tag: str = "latest"):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"simclr_{tag}.pt")
    torch.save({
        "epoch":      epoch,
        "best_loss":  best_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scaler":     scaler.state_dict() if scaler else None,
    }, path)
    print(f"[ckpt] saved → {path}")
    return path


def load_checkpoint(path: str, model: SimCLREncoder, optimizer=None,
                    scaler=None, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    epoch     = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"[ckpt] loaded ← {path}  (epoch {epoch}, best_loss {best_loss:.4f})")
    return epoch, best_loss


# ══════════════════════════════════════════════════════════════════════════════
# 7. Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def build_loaders(args: argparse.Namespace):
    """
    Split BUSI into 80 / 10 / 10 train / val / test.
    BUSIDataset is expected to return (view1, view2, label).
    For linear eval we pass ssl=False so it returns (img, img, label)
    with standard resize/normalise only — no augmentation.
    """
    # Use the SSL wrapper for the pretraining loader
    from data.dataset import BUSIDataset, BUSIDatasetSSL
    from torchvision import transforms
    from augmentations.ssl_transforms import SimCLRAugment # Ensure correct imports
    # 1. Create the base dataset (this one takes the root path)
    # We pass transform=None here because BUSIDatasetSSL will handle the augmentation
    base_dataset = BUSIDataset(root=args.data_dir, image_size=224)

    # 2. Create the SSL augmenter
    ssl_transform = SimCLRAugment(size=224)

    # 3. Wrap the base dataset in the SSL version
    # Note: Use 'busi_dataset' as the argument name to match your dataset.py
    full_ssl = BUSIDatasetSSL(busi_dataset=base_dataset, ssl_transform=ssl_transform)
    
    #3. Eval Loader (The "Lens" for Linear Probing)
    # We use a simple transform here so it returns Tensors, not PIL Images
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We create a separate instance for eval so it has its own transform
    full_eval = BUSIDataset(root=args.data_dir, image_size=224, transform=eval_transform)    # This is for linear probe

    n        = len(full_ssl)
    n_train  = int(0.8 * n)
    n_val    = int(0.1 * n)
    n_test   = n - n_train - n_val

    gen = torch.Generator().manual_seed(args.seed)
    tr_idx, val_idx, te_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=gen)

    from torch.utils.data import Subset
    ssl_train = Subset(full_ssl,  list(tr_idx))
    ev_train  = Subset(full_eval, list(tr_idx))
    ev_val    = Subset(full_eval, list(val_idx))
    ev_test   = Subset(full_eval, list(te_idx))

    # Keep drop_last=True for training to avoid small batches destabilizing BatchNorm
    ssl_kw = dict(num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Keep drop_last=False for eval to ensure we see the whole dataset
    eval_kw = dict(num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    ssl_loader   = DataLoader(ssl_train, batch_size=args.batch_size, shuffle=True,  **ssl_kw)
    eval_tr_ldr  = DataLoader(ev_train,  batch_size=64,             shuffle=False, **eval_kw)
    eval_val_ldr = DataLoader(ev_val,    batch_size=64,             shuffle=False, **eval_kw)

    print(f"[data]  SSL train={len(ssl_train)} | "
          f"eval train={len(ev_train)} | val={len(ev_val)} | test={len(ev_test)}")
    return ssl_loader, eval_tr_ldr, eval_val_ldr


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"simclr_{args.backbone}_bs{args.batch_size}",
            config=vars(args),
            tags=["simclr", "ssl", "busi", args.backbone],
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    ssl_loader, eval_tr_ldr, eval_val_ldr = build_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SimCLREncoder(
        backbone=args.backbone,
        proj_hidden=args.proj_hidden,
        proj_dim=args.proj_dim,
    ).to(device)

    print(f"[model] {args.backbone}  feat_dim={model.feat_dim}  "
          f"proj={args.proj_hidden}→{args.proj_dim}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {total_params:.1f}M parameters")

    # ── Loss / Optimiser / Scaler ─────────────────────────────────────────────
    criterion = NTXentLoss(temperature=args.temperature, device=device)
    optimizer = build_optimizer(model, args)
    scaler    = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.ckpt:
            raise ValueError("--ckpt required with --eval_only")
        load_checkpoint(args.ckpt, model, device=device)
        metrics = linear_eval(model, eval_tr_ldr, eval_val_ldr, device, args, epoch=0)
        print(metrics)
        if not args.no_wandb:
            wandb.log(metrics)
            wandb.finish()
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device)
        start_epoch += 1  # continue from next epoch

    if args.smoke_test:
        print("[smoke_test] Overriding epochs → 1")
        args.epochs = 1

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):

        lr = cosine_lr_with_warmup(optimizer, epoch, args)

        train_metrics = train_one_epoch(
            model, ssl_loader, optimizer, criterion, device, epoch, args, scaler)
        train_metrics["train/lr"] = lr

        # ── Logging ───────────────────────────────────────────────────────────
        print(f"[epoch {epoch:03d}]  loss={train_metrics['train/loss']:.4f}"
              f"  lr={lr:.5f}  {train_metrics['train/throughput']:.0f} img/s")
        if not args.no_wandb:
            wandb.log(train_metrics, step=epoch)

        # ── Checkpointing ─────────────────────────────────────────────────────
        is_best = train_metrics["train/loss"] < best_loss
        if is_best:
            best_loss = train_metrics["train/loss"]
            save_checkpoint(model, optimizer, scaler, epoch, best_loss,
                            args.ckpt_dir, tag="best")

        # Periodic checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, scaler, epoch, best_loss,
                            args.ckpt_dir, tag=f"epoch_{epoch+1:03d}")

        # Always keep "latest" for resume
        save_checkpoint(model, optimizer, scaler, epoch, best_loss,
                        args.ckpt_dir, tag="latest")

        # ── Linear evaluation ─────────────────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_metrics = linear_eval(
                model, eval_tr_ldr, eval_val_ldr, device, args, epoch)
            if not args.no_wandb:
                wandb.log(eval_metrics, step=epoch)

    print(f"\n[done]  best_loss={best_loss:.4f}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()