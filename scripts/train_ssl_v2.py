"""
train_simclr.py — PULSE Project
================================
SimCLR self-supervised pretraining on breast ultrasound images (BUSI dataset).

Usage
-----
  # Local run (RTX 4050, effective batch = 64×4 = 256)
  python train_simclr.py --data_dir data/BUSI --epochs 400 --batch_size 64 --accum_steps 4

  # Toubkal run (multi-GPU, effective batch = 256×4 = 1024)
  python train_simclr.py --data_dir data/BUSI --epochs 1000 --batch_size 256 --accum_steps 4 --compile

  # Quick smoke test
  python train_simclr.py --data_dir data/BUSI --epochs 1 --batch_size 32 --smoke_test

  # Resume from checkpoint
  python train_simclr.py --data_dir data/BUSI --resume checkpoints/simclr/simclr_latest.pt

  # Linear evaluation only (frozen encoder)
  python train_simclr.py --data_dir data/BUSI --eval_only --ckpt checkpoints/simclr/simclr_best_acc.pt

  # Low-label evaluation (10% of training labels — MICCAI low-budget experiment)
  python train_simclr.py --data_dir data/BUSI --eval_only --ckpt checkpoints/simclr/simclr_best_acc.pt --low_label 0.1

Architecture
------------
  Encoder  : ResNet-50 (randomly initialised)
  Projector: 2-layer MLP + BN on output -> 128-d unit-hypersphere  (SimCLR v2)
  Loss     : NT-Xent (temperature-scaled cross-entropy)
  Eval     : Warm-started linear probe on frozen encoder

Key features
------------
  - Gradient accumulation  : effective_bs = batch_size × accum_steps
  - Ultrasound augmentations: tuned for grayscale medical images (no hue jitter)
  - Grad norm clipping+logging: detects silent gradient explosions
  - EMA loss curve         : clean W&B signal even at small batch sizes
  - Early stopping         : patience on eval/linear_acc (Toubkal self-termination)
  - Low-label eval         : --low_label fraction for low-budget downstream experiment
  - Best-acc checkpoint fix: only saves when acc strictly improves
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
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models

import wandb

# ── Local PULSE modules ────────────────────────────────────────────────────────
from data.dataset import BUSIDataset, BUSIDatasetSSL          # returns (view1, view2, label)

from losses.nt_xent import NTXentLoss       # NT-Xent implementation


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PULSE -- SimCLR pretraining")

    # Paths
    p.add_argument("--data_dir",     type=str,   default="data/BUSI")
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints/simclr")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Resume training from a .pt checkpoint")
    p.add_argument("--ckpt",         type=str,   default=None,
                   help="Encoder checkpoint for --eval_only")

    # Model
    p.add_argument("--backbone",     type=str,   default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--proj_dim",     type=int,   default=128,
                   help="Projector output dimension")
    p.add_argument("--proj_hidden",  type=int,   default=2048,
                   help="Projector hidden dimension")

    # SSL training
    p.add_argument("--epochs",       type=int,   default=400)
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="Per-step batch size (before accumulation)")
    p.add_argument("--accum_steps",  type=int,   default=4,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size x accum_steps")
    p.add_argument("--lr",           type=float, default=0.6,
                   help="Base LR; scaled by effective_batch / 256")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature",  type=float, default=0.5,
                   help="NT-Xent temperature tau")
    p.add_argument("--warmup_epochs",type=int,   default=10)
    p.add_argument("--min_lr",       type=float, default=0.0)
    p.add_argument("--loss_ema",     type=float, default=0.98,
                   help="EMA decay for smoothed loss logging")

    # Linear eval
    p.add_argument("--eval_every",   type=int,   default=20,
                   help="Run linear probe every N epochs")
    p.add_argument("--eval_epochs",  type=int,   default=100,
                   help="Epochs to train the linear head")
    p.add_argument("--eval_lr",      type=float, default=0.1)
    p.add_argument("--eval_only",    action="store_true")
    p.add_argument("--low_label",    type=float, default=1.0,
                   help="Fraction of training labels to use in linear probe "
                        "(e.g. 0.1 = 10%%). Set <1.0 for low-budget experiments.")

    # Early stopping (keyed on eval/linear_acc)
    p.add_argument("--patience",     type=int,   default=0,
                   help="Stop if eval/linear_acc does not improve for N evals. "
                        "0 = disabled. Recommended: 10 for Toubkal runs.")

    # Gradient clipping
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max gradient norm. 0 = disabled.")

    # Infra
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--compile",      action="store_true",
                   help="torch.compile the model (PyTorch >= 2.0, Toubkal)")
    p.add_argument("--smoke_test",   action="store_true",
                   help="Override to 1 epoch for a quick sanity check")
    p.add_argument("--wandb_project",type=str,   default="PULSE")
    p.add_argument("--wandb_run",    type=str,   default=None)
    p.add_argument("--no_wandb",     action="store_true")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ultrasound-tuned augmentation pipeline
# ══════════════════════════════════════════════════════════════════════════════

class BUSIAugmentation(nn.Module):
    """
    Dual-view augmentation for breast ultrasound images.

    Key differences from standard SimCLR (ImageNet) augmentations:
      - No hue jitter     : ultrasound is grayscale; hue shift is meaningless
                            and can destroy speckle texture cues.
      - Reduced saturation: same reason.
      - Conservative crop : RandomResizedCrop(scale=(0.5, 1.0)) instead of
                            (0.08, 1.0) — lesions are small; aggressive crops
                            frequently exclude the ROI entirely.
      - Gaussian blur     : kept; models real transducer focus variation.
      - Horizontal flip   : valid for breast US; vertical flip excluded
                            (probe orientation convention).
      - Grayscale always  : input_channels=1 is common for BUSI; converts
                            3-channel duplicated tensors back to meaningful space.

    Usage:
        aug = BUSIAugmentation(image_size=224)
        view1, view2 = aug(img), aug(img)   # two stochastic views
    """

    def __init__(self, image_size: int = 224):
        super().__init__()
        from torchvision import transforms as T

        # Colour jitter params tuned for ultrasound (no hue)
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.1, hue=0.0)

        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * image_size) | 1,
                           sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats work fine
                        std=[0.229, 0.224, 0.225]),    # for transfer init
        ])

    def forward(self, x):
        return self.transform(x)



class SimCLREncoder(nn.Module):
    """
    ResNet backbone + 2-layer MLP projector (SimCLR v2 style).

    Forward returns (h, z):
      h : [B, feat_dim]  -- pre-projector representations  -> linear probe
      z : [B, proj_dim]  -- L2-normalised projector output -> NT-Xent loss

    Projector uses BN on the output layer (SimCLR v2 over v1).
    Consistently improves linear probe accuracy by 1-3% on small datasets
    by preventing representation collapse on the unit hypersphere.
    """

    def __init__(self, backbone: str = "resnet50",
                 proj_hidden: int = 2048, proj_dim: int = 128):
        super().__init__()

        base = getattr(models, backbone)(weights=None)
        self.feat_dim = base.fc.in_features

        # Drop the classification head, keep everything through avg-pool
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # [B, D, 1, 1]

        # SimCLR v2 projector: Linear -> BN -> ReLU -> Linear -> BN
        # BN on output (affine=False) acts as implicit L2 regulariser
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),   # <-- v2 addition
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)             # [B, feat_dim]
        z = F.normalize(self.projector(h), dim=1)  # [B, proj_dim]
        return h, z


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimiser & LR schedule
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    LARS is recommended for SimCLR at large batch sizes.
    Falls back to SGD if torch-optimizer is not installed.
    LR is scaled by effective_batch_size / 256.
    """
    effective_bs = args.batch_size * args.accum_steps
    scaled_lr    = args.lr * effective_bs / 256.0

    try:
        from torch_optimizer import LARS
        optimizer = LARS(model.parameters(), lr=scaled_lr,
                         weight_decay=args.weight_decay, momentum=0.9)
        print(f"[optim] LARS  lr={scaled_lr:.4f}  effective_bs={effective_bs}")
    except ImportError:
        optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        print(f"[optim] SGD (LARS not found -- pip install torch-optimizer)  "
              f"lr={scaled_lr:.4f}  effective_bs={effective_bs}")

    return optimizer


def cosine_lr_with_warmup(optimizer: torch.optim.Optimizer,
                           epoch: int, args: argparse.Namespace) -> float:
    """Linear warmup -> cosine decay. Called once per epoch."""
    effective_bs = args.batch_size * args.accum_steps
    scaled_lr    = args.lr * effective_bs / 256.0

    if epoch < args.warmup_epochs:
        lr = scaled_lr * (epoch + 1) / args.warmup_epochs
    else:
        t  = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (scaled_lr - args.min_lr) * (1 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════════════════════
# 4. Training loop -- one epoch (with gradient accumulation)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: SimCLREncoder,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: NTXentLoss,
                    device: torch.device,
                    epoch: int,
                    args: argparse.Namespace,
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    loss_ema: float) -> tuple:
    """
    Returns (metrics_dict, updated_loss_ema).

    Gradient accumulation: gradients are accumulated over `accum_steps`
    micro-batches before each optimiser step, making the effective batch
    size = batch_size x accum_steps without increasing VRAM usage.
    NT-Xent benefits directly: more negatives per update step.
    """
    model.train()
    total_loss      = 0.0
    total_grad_norm = 0.0
    optimizer_steps = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    
    for step, (view1, view2, _) in enumerate(loader):
        # BUSIDatasetSSL returns (view1, view2, label)
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        # ── Forward + scaled loss ─────────────────────────────────────────────
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                _, z1 = model(view1)
                _, z2 = model(view2)
                # Divide by accum_steps so gradients average, not sum
                loss  = criterion(z1, z2) / args.accum_steps
            scaler.scale(loss).backward()
        else:
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss  = criterion(z1, z2) / args.accum_steps
            loss.backward()

        # Accumulate unscaled loss for logging
        total_loss += loss.item() * args.accum_steps

        # ── Optimiser step every accum_steps micro-batches ────────────────────
        if (step + 1) % args.accum_steps == 0:
            if scaler is not None:
                # Unscale before clipping so the norm is in real gradient space
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
                total_grad_norm += grad_norm.item()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if optimizer_steps % 10 == 0:
                print(f"  [epoch {epoch:03d} | opt_step {optimizer_steps:04d}]"
                      f"  loss={total_loss / (step + 1):.4f}"
                      f"  grad_norm={total_grad_norm / optimizer_steps:.3f}")

    # Handle any remainder micro-batches at the end of the epoch
    if len(loader) % args.accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss   = total_loss / len(loader)
    elapsed    = time.time() - t0
    throughput = len(loader.dataset) / elapsed

    # EMA-smoothed loss: much cleaner W&B curve than raw per-epoch loss
    loss_ema = (args.loss_ema * loss_ema + (1 - args.loss_ema) * avg_loss
                if loss_ema > 0 else avg_loss)

    metrics = {
        "train/loss":         avg_loss,
        "train/loss_smooth":  loss_ema,
        "train/grad_norm":    total_grad_norm / max(1, optimizer_steps),
        "train/epoch":        epoch,
        "train/throughput":   throughput,
        "train/time_s":       elapsed,
        "train/effective_bs": args.batch_size * args.accum_steps,
    }
    return metrics, loss_ema


# ══════════════════════════════════════════════════════════════════════════════
# 5. Linear evaluation (frozen encoder, warm-started head)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(encoder: SimCLREncoder, loader: DataLoader,
                     device: torch.device):
    """Extract pre-projector h representations for the entire split."""
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        imgs = batch[0]
        ys   = batch[1] if len(batch) == 2 else batch[2]
        h, _ = encoder(imgs.to(device))
        feats.append(h.cpu())
        labels.append(ys)
    return torch.cat(feats), torch.cat(labels)


def linear_eval(encoder: SimCLREncoder,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                args: argparse.Namespace,
                epoch: int,
                probe_head: Optional[nn.Linear],
                best_linear_acc: float) -> tuple:
    """
    Train a linear probe on frozen encoder features.

    The head is warm-started from the previous eval's weights instead of
    being re-initialised cold. This eliminates the sawtooth pattern in
    eval/linear_acc: the head only adapts to encoder drift since last eval
    rather than re-learning from scratch, giving a monotonically improving
    signal in W&B.

    --low_label < 1.0 subsamples the training features for low-budget
    downstream evaluation (MICCAI low-label experiment).

    Returns (metrics_dict, updated_probe_head, updated_best_acc).
    """
    print("[linear eval] Extracting features...")
    X_tr, y_tr   = extract_features(encoder, train_loader, device)
    X_val, y_val = extract_features(encoder, val_loader,   device)

    # ── Low-label subsampling ─────────────────────────────────────────────────
    if args.low_label < 1.0:
        n_keep = max(1, int(len(X_tr) * args.low_label))
        perm   = torch.randperm(len(X_tr))[:n_keep]
        X_tr, y_tr = X_tr[perm], y_tr[perm]
        print(f"[linear eval] Low-label mode: using {n_keep}/{len(X_tr)} "
              f"samples ({args.low_label*100:.0f}%)")

    num_classes = int(y_tr.max().item()) + 1

    if probe_head is None or probe_head.out_features != num_classes:
        probe_head = nn.Linear(encoder.feat_dim, num_classes).to(device)
        print("[linear eval] Head initialised fresh")
    else:
        probe_head = probe_head.to(device)
        print("[linear eval] Head warm-started from previous probe")

    opt = torch.optim.Adam(probe_head.parameters(),
                           lr=args.eval_lr, weight_decay=1e-4)

    X_tr  = X_tr.to(device);  y_tr  = y_tr.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)

    probe_head.train()
    for _ in range(args.eval_epochs):
        logits = probe_head(X_tr)
        loss   = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    probe_head.eval()
    with torch.no_grad():
        preds = probe_head(X_val).argmax(dim=1)
        acc   = (preds == y_val).float().mean().item()

    best_linear_acc = max(best_linear_acc, acc)
    print(f"[linear eval @ epoch {epoch}]  "
          f"val_acc={acc:.4f}  best={best_linear_acc:.4f}")

    metrics = {
        "eval/linear_acc":      acc,
        "eval/best_linear_acc": best_linear_acc,  # monotonically non-decreasing
        "eval/epoch":           epoch,
    }
    # Return head on CPU to avoid holding VRAM between evals
    return metrics, probe_head.cpu(), best_linear_acc


# ══════════════════════════════════════════════════════════════════════════════
# 6. Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: SimCLREncoder, optimizer, scaler,
                    probe_head: Optional[nn.Linear],
                    epoch: int, best_loss: float, best_acc: float,
                    ckpt_dir: str, tag: str = "latest") -> str:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"simclr_{tag}.pt")
    torch.save({
        "epoch":          epoch,
        "best_loss":      best_loss,
        "best_acc":       best_acc,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict() if scaler else None,
        "probe_head":     probe_head.state_dict() if probe_head else None,
        "probe_feat_dim": model.feat_dim,
    }, path)
    print(f"[ckpt] saved -> {path}")
    return path


def load_checkpoint(path: str, model: SimCLREncoder,
                    optimizer=None, scaler=None,
                    device: torch.device = torch.device("cpu")):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    epoch     = ckpt.get("epoch",     0)
    best_loss = ckpt.get("best_loss", float("inf"))
    best_acc  = ckpt.get("best_acc",  0.0)

    probe_head = None
    if ckpt.get("probe_head") is not None:
        num_classes = next(iter(ckpt["probe_head"].values())).shape[0]
        probe_head  = nn.Linear(ckpt["probe_feat_dim"], num_classes)
        probe_head.load_state_dict(ckpt["probe_head"])

    print(f"[ckpt] loaded <- {path}  "
          f"(epoch {epoch}, best_loss {best_loss:.4f}, best_acc {best_acc:.4f})")
    return epoch, best_loss, best_acc, probe_head


# ══════════════════════════════════════════════════════════════════════════════
# 7. Data loaders
# ══════════════════════════════════════════════════════════════════════════════
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        # Applies the same transform pipeline N times to the same image
        return [self.base_transforms(x) for _ in range(self.n_views)]
    # This class is unused and can be removed.
    pass
    
def build_loaders(args: argparse.Namespace):
    """
    Deterministic 80 / 10 / 10 split (seeded).
    BUSIDataset(ssl=True)  -> (view1, view2, label)  -- augmented dual-view
    BUSIDataset(ssl=False) -> (img,   img,   label)  -- clean resize/normalise
    """
    # Helper for BUSIDatasetSSL to generate dual views
    class _DualView:
        """
        Thin wrapper so BUSIDatasetSSL gets a callable that returns (view1, view2).
        BUSIDatasetSSL expects: view1, view2 = ssl_transform(image)
        """
        def __init__(self, aug: BUSIAugmentation):
            self.aug = aug
     
        def __call__(self, img):
            return self.aug(img), self.aug(img)
    
    import torchvision.transforms as T

    eval_transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Base dataset with NO transform (BUSIDatasetSSL applies augmentation internally)
    full_ssl = BUSIDataset(args.data_dir, transform=None)
    # 1. Create the base dataset (this one takes the root path)
    base_ssl_dataset = BUSIDataset(args.data_dir, transform=None)
    # 2. Create the SSL augmenter and wrap it for dual views
    ssl_augmenter = BUSIAugmentation(image_size=224)
    dual_view_transform = _DualView(ssl_augmenter)
    # 3. Wrap the base dataset in the SSL version
    full_ssl = BUSIDatasetSSL(busi_dataset=base_ssl_dataset, ssl_transform=dual_view_transform)
    full_eval = BUSIDataset(args.data_dir, transform=eval_transform)

    n       = len(full_ssl)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    gen = torch.Generator().manual_seed(args.seed)
    tr_idx, val_idx, te_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=gen)

    ssl_train = Subset(full_ssl,  list(tr_idx))
    ev_train  = Subset(full_eval, list(tr_idx))
    ev_val    = Subset(full_eval, list(val_idx))
    ev_test   = Subset(full_eval, list(te_idx))

    kw = dict(num_workers=args.num_workers, pin_memory=True, drop_last=True)

    ssl_loader   = DataLoader(ssl_train, batch_size=args.batch_size,
                              shuffle=True,  **kw)
    eval_tr_ldr  = DataLoader(ev_train,  batch_size=512, shuffle=False, **kw)
    eval_val_ldr = DataLoader(ev_val,    batch_size=512, shuffle=False, **kw)

    eff_bs = args.batch_size * args.accum_steps
    print(f"[data]  SSL train={len(ssl_train)} | eval train={len(ev_train)} "
          f"| val={len(ev_val)} | test={len(ev_test)}")
    print(f"[data]  micro_bs={args.batch_size}  "
          f"accum_steps={args.accum_steps}  effective_bs={eff_bs}")
    return ssl_loader, eval_tr_ldr, eval_val_ldr


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[device] {device}  ({gpu_name})")

    # ── W&B ───────────────────────────────────────────────────────────────────
    eff_bs = args.batch_size * args.accum_steps
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run or
                  f"simclr_{args.backbone}_effbs{eff_bs}_e{args.epochs}"),
            config={**vars(args), "effective_batch_size": eff_bs,
                    "gpu": gpu_name},
            tags=["simclr", "ssl", "busi", args.backbone, f"effbs{eff_bs}"],
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    ssl_loader, eval_tr_ldr, eval_val_ldr = build_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SimCLREncoder(backbone=args.backbone,
                          proj_hidden=args.proj_hidden,
                          proj_dim=args.proj_dim).to(device)

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("[model] torch.compile enabled")
        else:
            print("[model] torch.compile not available (requires PyTorch >= 2.0)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.backbone}  feat_dim={model.feat_dim}  "
          f"proj={args.proj_hidden}->{args.proj_dim}  {n_params:.1f}M params")

    # ── Loss / Optimiser / AMP scaler ─────────────────────────────────────────
    criterion = NTXentLoss(temperature=args.temperature, device=device)
    optimizer = build_optimizer(model, args)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Persistent state ──────────────────────────────────────────────────────
    start_epoch      = 0
    best_loss        = float("inf")
    best_linear_acc  = 0.0
    probe_head       = None   # persisted across evals for warm-starting
    loss_ema         = 0.0
    patience_counter = 0      # evals without improvement (early stopping)

    # ── Eval-only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.ckpt:
            raise ValueError("--ckpt is required with --eval_only")
        load_checkpoint(args.ckpt, model, device=device)
        metrics, _, _ = linear_eval(model, eval_tr_ldr, eval_val_ldr,
                                    device, args, epoch=0,
                                    probe_head=None, best_linear_acc=0.0)
        print(metrics)
        if not args.no_wandb:
            wandb.log(metrics); wandb.finish()
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        start_epoch, best_loss, best_linear_acc, probe_head = load_checkpoint(
            args.resume, model, optimizer, scaler, device)
        start_epoch += 1

    if args.smoke_test:
        print("[smoke_test] Overriding epochs -> 1")
        args.epochs = 1

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs):

        lr = cosine_lr_with_warmup(optimizer, epoch, args)

        train_metrics, loss_ema = train_one_epoch(
            model, ssl_loader, optimizer, criterion,
            device, epoch, args, scaler, loss_ema)
        train_metrics["train/lr"] = lr

        print(f"[epoch {epoch:03d}/{args.epochs}]  "
              f"loss={train_metrics['train/loss']:.4f}  "
              f"loss_smooth={train_metrics['train/loss_smooth']:.4f}  "
              f"grad_norm={train_metrics['train/grad_norm']:.3f}  "
              f"lr={lr:.5f}  "
              f"{train_metrics['train/throughput']:.0f} img/s")

        if not args.no_wandb:
            wandb.log(train_metrics, step=epoch)

        # ── Linear evaluation ─────────────────────────────────────────────────
        run_eval = ((epoch + 1) % args.eval_every == 0
                    or epoch == args.epochs - 1)
        if run_eval:
            eval_metrics, probe_head, best_linear_acc = linear_eval(
                model, eval_tr_ldr, eval_val_ldr, device, args,
                epoch, probe_head, best_linear_acc)
            if not args.no_wandb:
                wandb.log(eval_metrics, step=epoch)

            # BUG FIX: only save best_acc checkpoint when acc strictly improves
            current_acc = eval_metrics["eval/linear_acc"]
            if current_acc > best_linear_acc:
                best_linear_acc = current_acc
                patience_counter = 0
                save_checkpoint(model, optimizer, scaler, probe_head,
                                epoch, best_loss, best_linear_acc,
                                args.ckpt_dir, tag="best_acc")
            else:
                patience_counter += 1
                print(f"[early stop] no improvement for {patience_counter} eval(s)"
                      + (f" / {args.patience}" if args.patience > 0 else ""))

            # ── Early stopping ────────────────────────────────────────────────
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"[early stop] patience {args.patience} reached at epoch {epoch}. "
                      f"Best linear acc: {best_linear_acc:.4f}. Stopping.")
                save_checkpoint(model, optimizer, scaler, probe_head,
                                epoch, best_loss, best_linear_acc,
                                args.ckpt_dir, tag="latest")
                break

        # ── Secondary: best NT-Xent loss ─────────────────────────────────────
        if train_metrics["train/loss"] < best_loss:
            best_loss = train_metrics["train/loss"]
            save_checkpoint(model, optimizer, scaler, probe_head,
                            epoch, best_loss, best_linear_acc,
                            args.ckpt_dir, tag="best_loss")

        # Periodic snapshot every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, scaler, probe_head,
                            epoch, best_loss, best_linear_acc,
                            args.ckpt_dir, tag=f"epoch_{epoch+1:04d}")

        # Latest for resume
        save_checkpoint(model, optimizer, scaler, probe_head,
                        epoch, best_loss, best_linear_acc,
                        args.ckpt_dir, tag="latest")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n[done]  best_loss={best_loss:.4f}  "
          f"best_linear_acc={best_linear_acc:.4f}")
    if not args.no_wandb:
        wandb.summary["best_linear_acc"] = best_linear_acc
        wandb.summary["best_loss"]       = best_loss
        wandb.finish()


if __name__ == "__main__":
    main()
