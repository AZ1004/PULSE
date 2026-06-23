"""
train_mae.py — PULSE Project
=============================
Masked Autoencoder (MAE) self-supervised pretraining on breast ultrasound
images (BUSI dataset). Companion script to train_ssl_v2.py (SimCLR).

Usage
-----
  # Local run (RTX 4050, effective batch = 64x4 = 256)
  python train_mae.py --data_dir data/BUSI --epochs 400 --batch_size 64 --accum_steps 4

  # Toubkal run (multi-GPU, effective batch = 256x4 = 1024)
  python train_mae.py --data_dir data/BUSI --epochs 1000 --batch_size 256 --accum_steps 4 --compile

  # Quick smoke test
  python train_mae.py --data_dir data/BUSI --epochs 1 --batch_size 32 --smoke_test

  # Resume from checkpoint
  python train_mae.py --data_dir data/BUSI --resume checkpoints/mae/mae_latest.pt

  # Masking ratio ablation (run 3x for the paper's ablation table)
  python train_mae.py --data_dir data/BUSI --mask_ratio 0.50 --wandb_run mae_mask50
  python train_mae.py --data_dir data/BUSI --mask_ratio 0.75 --wandb_run mae_mask75
  python train_mae.py --data_dir data/BUSI --mask_ratio 0.85 --wandb_run mae_mask85

  # Linear evaluation only (frozen encoder)
  python train_mae.py --data_dir data/BUSI --eval_only --ckpt checkpoints/mae/mae_best_acc.pt

  # Low-label evaluation (10% of training labels — MICCAI low-budget experiment)
  python train_mae.py --data_dir data/BUSI --eval_only --ckpt checkpoints/mae/mae_best_acc.pt --low_label 0.1

Architecture
------------
  Encoder  : ViT-Small (patch 16, randomly initialised)
  Decoder  : lightweight ViT decoder (shallow, narrow embed dim)
  Loss     : MSE reconstruction on masked patches only
  Eval     : Warm-started linear probe on frozen encoder (CLS token)

Key features
------------
  - Gradient accumulation   : effective_bs = batch_size x accum_steps
  - Patch masking           : random masking, ratio configurable (default 0.75)
  - Sin-cos positional embeddings (fixed, not learned) — standard MAE choice
  - Grad norm clipping+logging
  - EMA loss curve for clean W&B signal
  - Early stopping on eval/linear_acc
  - Low-label eval          : --low_label fraction for low-budget downstream experiment
  - Mirrors train_ssl_v2.py conventions so evaluate.py / checkpoints are consistent
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

import wandb

# ── Local PULSE modules ────────────────────────────────────────────────────────
from data.dataset import BUSIDataset, BUSIDatasetSSL          # returns (view1, view2, label)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PULSE -- MAE pretraining")

    # Paths
    p.add_argument("--data_dir",     type=str,   default="data/BUSI")
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints/mae")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Resume training from a .pt checkpoint")
    p.add_argument("--ckpt",         type=str,   default=None,
                   help="Encoder checkpoint for --eval_only")

    # Model — ViT-Small encoder, lightweight decoder (standard MAE recipe)
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--patch_size",   type=int,   default=16)
    p.add_argument("--encoder_dim",  type=int,   default=384,   # ViT-Small
                   help="Encoder embedding dimension")
    p.add_argument("--encoder_depth",type=int,   default=12)
    p.add_argument("--encoder_heads",type=int,   default=6)
    p.add_argument("--decoder_dim",  type=int,   default=192,   # lightweight, per MAE paper
                   help="Decoder embedding dimension (narrower than encoder)")
    p.add_argument("--decoder_depth",type=int,   default=4)
    p.add_argument("--decoder_heads",type=int,   default=6)
    p.add_argument("--mask_ratio",   type=float, default=0.75,
                   help="Fraction of patches masked. Ablate: 0.50 / 0.75 / 0.85")

    # SSL training
    p.add_argument("--epochs",       type=int,   default=400)
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="Per-step batch size (before accumulation)")
    p.add_argument("--accum_steps",  type=int,   default=4,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size x accum_steps")
    p.add_argument("--lr",           type=float, default=1.5e-4,
                   help="Base LR; scaled by effective_batch / 256 (MAE uses AdamW, lower LR than SimCLR)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=20,
                   help="MAE typically needs a longer warmup than SimCLR")
    p.add_argument("--min_lr",       type=float, default=0.0)
    p.add_argument("--loss_ema",     type=float, default=0.98)

    # Linear eval
    p.add_argument("--eval_every",   type=int,   default=20)
    p.add_argument("--eval_epochs",  type=int,   default=100)
    p.add_argument("--eval_lr",      type=float, default=0.1)
    p.add_argument("--eval_only",    action="store_true")
    p.add_argument("--low_label",    type=float, default=1.0,
                   help="Fraction of training labels used in linear probe")

    # Early stopping
    p.add_argument("--patience",     type=int,   default=0,
                   help="Stop if eval/linear_acc does not improve for N evals. 0 = disabled.")

    # Gradient clipping
    p.add_argument("--grad_clip",    type=float, default=1.0)

    # Infra
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--compile",      action="store_true")
    p.add_argument("--smoke_test",   action="store_true")
    p.add_argument("--wandb_project",type=str,   default="PULSE")
    p.add_argument("--wandb_run",    type=str,   default=None)
    p.add_argument("--no_wandb",     action="store_true")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Patch embedding + sin-cos positional encoding
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Splits image into patches and linearly projects each to embed_dim."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


def sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Fixed 2D sin-cos positional embedding (not learned).
    Standard MAE choice — generalizes better than learned embeddings
    at small dataset sizes like BUSI (780 images).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid   = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid   = torch.stack(grid, dim=0).reshape(2, -1)

    def _1d_sincos(dim, pos):
        omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)))
        out   = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = _1d_sincos(embed_dim // 2, grid[0])
    emb_w = _1d_sincos(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)  # [grid_size**2, embed_dim]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAE model — encoder + lightweight decoder
# ══════════════════════════════════════════════════════════════════════════════

class MAEEncoder(nn.Module):
    """
    ViT encoder that processes ONLY visible (unmasked) patches.
    This is the key efficiency trick from the MAE paper — the encoder
    never sees masked tokens, making pretraining ~3-4x cheaper than
    naively running a full ViT and masking the loss only.
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=384,
                depth=12, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size   = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = sincos_pos_embed(embed_dim, self.grid_size)
        # Prepend a zero position for the CLS token; buffer = not trained
        self.register_buffer(
            "pos_embed",
            torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0).unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm   = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=0.02)
        self.embed_dim = embed_dim

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Randomly mask patches by shuffling and keeping only the first
        (1 - mask_ratio) fraction. Returns visible tokens + bookkeeping
        indices needed by the decoder to restore original patch order.
        """
        B, N, D = x.shape
        n_keep  = int(N * (1 - mask_ratio))

        noise        = torch.rand(B, N, device=x.device)
        shuffle_idx  = torch.argsort(noise, dim=1)
        restore_idx  = torch.argsort(shuffle_idx, dim=1)

        keep_idx = shuffle_idx[:, :n_keep]
        x_masked = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask for loss computation: 1 = masked (removed), 0 = kept
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, restore_idx)

        return x_masked, mask, restore_idx

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        """
        Returns:
            latent      : [B, n_visible+1, D]  encoded visible patches + CLS
            mask        : [B, N]               1 = masked, 0 = visible
            restore_idx : [B, N]               to unshuffle in decoder
        """
        x = self.patch_embed(x)                       # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]               # patch pos embed (skip CLS slot)

        x, mask, restore_idx = self.random_masking(x, mask_ratio)

        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)               # prepend CLS

        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, restore_idx

    def forward_no_mask(self, x: torch.Tensor):
        """Full forward pass with NO masking — used for downstream eval
        (linear probe / fine-tuning), where we want the CLS token from
        the complete image, not a masked reconstruction setup."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]   # CLS token only, [B, D]


class MAEDecoder(nn.Module):
    """
    Lightweight decoder: reconstructs pixel values for masked patches.
    Intentionally narrow/shallow per the MAE paper — the decoder's job
    ends at pretraining (discarded for downstream tasks), so it doesn't
    need encoder-level capacity.
    """

    def __init__(self, num_patches, patch_size=16, encoder_dim=384,
                decoder_dim=192, depth=4, num_heads=6):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token     = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        grid_size = int(num_patches ** 0.5)
        pos_embed = sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer(
            "decoder_pos_embed",
            torch.cat([torch.zeros(1, decoder_dim), pos_embed], dim=0).unsqueeze(0))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=decoder_dim * 4, activation="gelu",
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm   = nn.LayerNorm(decoder_dim)
        self.pred   = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        nn.init.normal_(self.mask_token, std=0.02)
        self.num_patches = num_patches

    def forward(self, latent: torch.Tensor, restore_idx: torch.Tensor):
        """
        Args:
            latent      : [B, n_visible+1, encoder_dim] from encoder (CLS + visible)
            restore_idx : [B, N] to restore original patch order

        Returns:
            pred : [B, N, patch_size*patch_size*3] reconstructed pixels per patch
        """
        x = self.decoder_embed(latent)                  # [B, n_visible+1, decoder_dim]

        B, N = restore_idx.shape
        n_visible = x.shape[1] - 1  # minus CLS

        # Append mask tokens for all masked positions
        mask_tokens = self.mask_token.repeat(B, N - n_visible, 1)
        x_no_cls    = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # [B, N, decoder_dim]

        # Unshuffle back to original patch order
        x_no_cls = torch.gather(
            x_no_cls, 1,
            restore_idx.unsqueeze(-1).expand(-1, -1, x_no_cls.shape[-1]))

        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)    # re-prepend CLS
        x = x + self.decoder_pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        x = self.pred(x)

        return x[:, 1:, :]  # drop CLS, return only patch predictions


class MAE(nn.Module):
    """Full MAE: encoder + decoder + patchify/unpatchify utilities."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.patch_size = args.patch_size
        self.encoder = MAEEncoder(
            img_size=args.img_size, patch_size=args.patch_size,
            embed_dim=args.encoder_dim, depth=args.encoder_depth,
            num_heads=args.encoder_heads)
        self.decoder = MAEDecoder(
            num_patches=self.encoder.num_patches, patch_size=args.patch_size,
            encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,
            depth=args.decoder_depth, num_heads=args.decoder_heads)
        self.feat_dim = args.encoder_dim   # for evaluate.py compatibility

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, num_patches, patch_size^2 * 3]"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)        # [B, h, w, p, p, C]
        x = x.reshape(B, h * w, p * p * C)
        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, restore_idx = self.encoder(imgs, mask_ratio)
        pred    = self.decoder(latent, restore_idx)         # [B, N, p*p*3]
        target  = self.patchify(imgs)                        # [B, N, p*p*3]

        # Reconstruction loss on MASKED patches only (mask == 1)
        loss_per_patch = (pred - target) ** 2
        loss_per_patch = loss_per_patch.mean(dim=-1)          # [B, N]
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        return loss, pred, mask


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optimiser & LR schedule (AdamW — standard for MAE, unlike SimCLR's LARS)
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """AdamW with LR scaled by effective_batch_size / 256, per MAE paper convention."""
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name.lower() or "bias" in name.lower() or "pos_embed" in name.lower():
            params.append({"params": [param], "weight_decay": 0.0})
        else:
            params.append({"params": [param]})

    effective_bs = args.batch_size * args.accum_steps
    scaled_lr    = args.lr * effective_bs / 256.0

    optimizer = torch.optim.AdamW(params, lr=scaled_lr,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    print(f"[optim] AdamW  lr={scaled_lr:.6f}  effective_bs={effective_bs}")
    return optimizer


def cosine_lr_with_warmup(optimizer: torch.optim.Optimizer,
                          epoch: int, args: argparse.Namespace) -> float:
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
# 5. Training loop — one epoch (gradient accumulation, mirrors train_ssl_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: MAE, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args: argparse.Namespace,
                    scaler: Optional[torch.amp.GradScaler],
                    loss_ema: float) -> tuple:
    model.train()
    total_loss      = 0.0
    total_grad_norm = 0.0
    optimizer_steps = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    # NOTE: BUSIDatasetSSL returns (view1, view2, label) for SimCLR's dual-view
    # contract. MAE only needs ONE view per image (masking IS the augmentation),
    # so we simply use view1 and discard view2 here.
    for step, (view1, _view2, _label) in enumerate(loader):
        view1 = view1.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, _, _ = model(view1, mask_ratio=args.mask_ratio)
                loss = loss / args.accum_steps
            scaler.scale(loss).backward()
        else:
            loss, _, _ = model(view1, mask_ratio=args.mask_ratio)
            loss = loss / args.accum_steps
            loss.backward()

        total_loss += loss.item() * args.accum_steps

        if (step + 1) % args.accum_steps == 0:
            if scaler is not None:
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

    loss_ema = avg_loss if loss_ema == 0 else (
        args.loss_ema * loss_ema + (1 - args.loss_ema) * avg_loss)

    metrics = {
        "train/loss":          avg_loss,
        "train/loss_smooth":   loss_ema,
        "train/grad_norm":     total_grad_norm / max(1, optimizer_steps),
        "train/throughput":    throughput,
        "train/effective_bs":  args.batch_size * args.accum_steps,
        "train/epoch":         epoch,
        "train/mask_ratio":    args.mask_ratio,
    }
    return metrics, loss_ema


# ══════════════════════════════════════════════════════════════════════════════
# 6. Linear evaluation (mirrors train_ssl_v2.py — warm-started head)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model: MAE, loader: DataLoader, device: torch.device):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        # Use forward_no_mask for downstream eval — full image, CLS token
        cls_feat = model.encoder.forward_no_mask(imgs)
        feats.append(cls_feat.cpu())
        labels.append(lbls)
    if not feats:
        raise ValueError("No features extracted! Check if your loader is empty.")
    return torch.cat(feats), torch.cat(labels)


def linear_eval(model: MAE, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, args: argparse.Namespace, epoch: int,
                probe_head: Optional[nn.Linear], best_linear_acc: float) -> tuple:
    """Identical protocol to train_ssl_v2.py's linear_eval for fair comparison."""
    print("[linear eval] Extracting features...")
    X_tr, y_tr   = extract_features(model, train_loader, device)
    X_val, y_val = extract_features(model, val_loader,   device)

    if args.low_label < 1.0:
        n_keep = max(1, int(len(X_tr) * args.low_label))
        perm   = torch.randperm(len(X_tr))[:n_keep]
        X_tr, y_tr = X_tr[perm], y_tr[perm]
        print(f"[linear eval] Low-label mode: using {n_keep}/{len(X_tr)} "
              f"samples ({args.low_label*100:.0f}%)")

    num_classes = int(y_tr.max().item()) + 1

    if probe_head is None or probe_head.out_features != num_classes:
        probe_head = nn.Linear(model.feat_dim, num_classes).to(device)
        print("[linear eval] Head initialised fresh")
    else:
        probe_head = probe_head.to(device)
        print("[linear eval] Head warm-started from previous probe")

    opt = torch.optim.Adam(probe_head.parameters(),
                           lr=args.eval_lr, weight_decay=1e-4)

    X_tr, y_tr   = X_tr.to(device),  y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

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
    print(f"[linear eval @ epoch {epoch}]  val_acc={acc:.4f}  best={best_linear_acc:.4f}")

    metrics = {
        "eval/linear_acc":      acc,
        "eval/best_linear_acc": best_linear_acc,
        "eval/epoch":           epoch,
    }
    return metrics, probe_head.cpu(), best_linear_acc


# ══════════════════════════════════════════════════════════════════════════════
# 7. Checkpoint helpers (same schema as train_ssl_v2.py for evaluate.py reuse)
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: MAE, optimizer, scaler, probe_head: Optional[nn.Linear],
                    epoch: int, best_loss: float, best_acc: float,
                    ckpt_dir: str, tag: str = "latest") -> str:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"mae_{tag}.pt")
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


def load_checkpoint(path: str, model: MAE, optimizer=None, scaler=None,
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
# 8. Data loaders — reuses BUSIDatasetSSL; MAE only consumes view1 (see above)
# ══════════════════════════════════════════════════════════════════════════════

class _SingleViewWrapper:
    """
    BUSIDatasetSSL expects ssl_transform(img) -> (view1, view2).
    MAE needs only one randomly-resized-cropped view (masking does the
    rest of the "augmentation" work) — we still return a tuple of two
    IDENTICAL light augmentations so BUSIDatasetSSL's contract is unchanged
    and train_one_epoch above simply discards the second.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        return self.base_transform(img), self.base_transform(img)


def build_mae_augment(img_size: int = 224):
    """
    Lighter augmentation than SimCLR — MAE's masking IS the primary
    self-supervisory signal, so heavy color/geometric augmentation is
    unnecessary and can even hurt reconstruction-based pretraining.
    Per the MAE paper: random resized crop + flip is sufficient.
    """
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_loaders(args: argparse.Namespace):
    """Identical 80/10/10 deterministic split as train_ssl_v2.py (same seed)
    so SimCLR and MAE are evaluated on the exact same data partition."""
    import torchvision.transforms as T

    eval_transform = T.Compose([
        T.Resize([args.img_size, args.img_size]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset    = BUSIDataset(args.data_dir, transform=None)
    mae_augmenter   = build_mae_augment(img_size=args.img_size)
    single_view_tf  = _SingleViewWrapper(mae_augmenter)
    full_ssl  = BUSIDatasetSSL(busi_dataset=base_dataset, ssl_transform=single_view_tf)
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

    ssl_kw  = dict(num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_kw = dict(num_workers=args.num_workers, pin_memory=True, drop_last=False)

    ssl_loader   = DataLoader(ssl_train, batch_size=args.batch_size, shuffle=True,  **ssl_kw)
    eval_tr_ldr  = DataLoader(ev_train,  batch_size=512, shuffle=False, **eval_kw)
    eval_val_ldr = DataLoader(ev_val,    batch_size=512, shuffle=False, **eval_kw)

    eff_bs = args.batch_size * args.accum_steps
    print(f"[data]  SSL train={len(ssl_train)} | eval train={len(ev_train)} "
          f"| val={len(ev_val)} | test={len(ev_test)}")
    print(f"[data]  micro_bs={args.batch_size}  accum_steps={args.accum_steps}  "
          f"effective_bs={eff_bs}  mask_ratio={args.mask_ratio}")
    return ssl_loader, eval_tr_ldr, eval_val_ldr


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[device] {device}  ({gpu_name})")

    eff_bs = args.batch_size * args.accum_steps
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run or
                  f"mae_vits_mask{int(args.mask_ratio*100)}_effbs{eff_bs}_e{args.epochs}"),
            config={**vars(args), "effective_batch_size": eff_bs, "gpu": gpu_name},
            tags=["mae", "ssl", "busi", "vit-small", f"mask{int(args.mask_ratio*100)}"],
        )

    ssl_loader, eval_tr_ldr, eval_val_ldr = build_loaders(args)

    model = MAE(args).to(device)

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("[model] torch.compile enabled")
        else:
            print("[model] torch.compile not available (requires PyTorch >= 2.0)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] ViT-Small encoder + lightweight decoder  "
          f"mask_ratio={args.mask_ratio}  {n_params:.1f}M params")

    optimizer = build_optimizer(model, args)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch      = 0
    best_loss        = float("inf")
    best_linear_acc  = 0.0
    probe_head       = None
    loss_ema         = 0.0
    patience_counter = 0

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

    if args.resume:
        start_epoch, best_loss, best_linear_acc, probe_head = load_checkpoint(
            args.resume, model, optimizer, scaler, device)
        start_epoch += 1

    if args.smoke_test:
        print("[smoke_test] Overriding epochs -> 1")
        args.epochs = 1

    for epoch in range(start_epoch, args.epochs):

        lr = cosine_lr_with_warmup(optimizer, epoch, args)

        train_metrics, loss_ema = train_one_epoch(
            model, ssl_loader, optimizer, device, epoch, args, scaler, loss_ema)
        train_metrics["train/lr"] = lr

        print(f"[epoch {epoch:03d}/{args.epochs}]  "
              f"loss={train_metrics['train/loss']:.4f}  "
              f"loss_smooth={train_metrics['train/loss_smooth']:.4f}  "
              f"grad_norm={train_metrics['train/grad_norm']:.3f}  "
              f"lr={lr:.6f}  "
              f"{train_metrics['train/throughput']:.0f} img/s")

        if not args.no_wandb:
            wandb.log(train_metrics, step=epoch)

        run_eval = ((epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1)
        if run_eval:
            eval_metrics, probe_head, best_linear_acc = linear_eval(
                model, eval_tr_ldr, eval_val_ldr, device, args,
                epoch, probe_head, best_linear_acc)
            if not args.no_wandb:
                wandb.log(eval_metrics, step=epoch)

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

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"[early stop] patience {args.patience} reached at epoch {epoch}. "
                      f"Best linear acc: {best_linear_acc:.4f}. Stopping.")
                save_checkpoint(model, optimizer, scaler, probe_head,
                                epoch, best_loss, best_linear_acc,
                                args.ckpt_dir, tag="latest")
                break

        if train_metrics["train/loss"] < best_loss:
            best_loss = train_metrics["train/loss"]
            save_checkpoint(model, optimizer, scaler, probe_head,
                            epoch, best_loss, best_linear_acc,
                            args.ckpt_dir, tag="best_loss")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, scaler, probe_head,
                            epoch, best_loss, best_linear_acc,
                            args.ckpt_dir, tag=f"epoch_{epoch+1:04d}")

        save_checkpoint(model, optimizer, scaler, probe_head,
                        epoch, best_loss, best_linear_acc,
                        args.ckpt_dir, tag="latest")

    print(f"\n[done]  best_loss={best_loss:.4f}  best_linear_acc={best_linear_acc:.4f}")
    if not args.no_wandb:
        wandb.summary["best_linear_acc"] = best_linear_acc
        wandb.summary["best_loss"]       = best_loss
        wandb.finish()


if __name__ == "__main__":
    main()
