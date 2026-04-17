# data/dataset.py

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ── Label mapping ──────────────────────────────────────────────────────────────
CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def _collect_samples(root: Path) -> list[dict]:
    """
    Walk the BUSI folder and collect one record per image (not per mask).

    Returns a list of dicts:
        {
            "image_path": Path,
            "mask_paths": list[Path],   # [] for normal class (often no mask)
            "label":      int,
            "class_name": str,
            "stem":       str,          # e.g. "benign (1)"
        }
    """
    samples = []

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")

        # Collect every file that is NOT a mask
        image_files = sorted([
            p for p in class_dir.iterdir()
            if p.suffix.lower() == ".png" and "_mask" not in p.name
        ])

        for img_path in image_files:
            stem = img_path.stem  # e.g. "benign (1)"

            # Find all masks for this image (handles _mask, _mask_1, _mask_2 …)
            mask_paths = sorted(class_dir.glob(f"{stem}_mask*.png"))

            samples.append({
                "image_path": img_path,
                "mask_paths": list(mask_paths),
                "label":      label,
                "class_name": class_name,
                "stem":       stem,
            })

    return samples


class BUSIDataset(Dataset):
    """
    BUSI dataset for supervised learning (classification).

    Returns (image_tensor, label) by default.
    Set return_mask=True to also return the merged binary mask.

    Args:
        root        : Path to Dataset_BUSI_with_GT folder.
        transform   : torchvision transform applied to the PIL image.
        image_size  : Resize target before applying transform (default 224).
        return_mask : If True, __getitem__ returns (image, mask, label).
        classes     : Subset of classes to include, e.g. ["benign","malignant"].
                      None = all three classes.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        image_size: int = 224,
        return_mask: bool = False,
        classes: list[str] | None = None,
    ):
        self.root        = Path(root)
        self.transform   = transform
        self.image_size  = image_size
        self.return_mask = return_mask

        all_samples = _collect_samples(self.root)

        # Optional class filtering
        if classes is not None:
            unknown = set(classes) - set(CLASS_TO_IDX)
            if unknown:
                raise ValueError(f"Unknown class(es): {unknown}. "
                                 f"Valid: {list(CLASS_TO_IDX)}")
            all_samples = [s for s in all_samples if s["class_name"] in classes]

        self.samples = all_samples

        # Class counts — useful for weighted sampling later
        self.class_counts = {c: 0 for c in CLASS_TO_IDX}
        for s in self.samples:
            self.class_counts[s["class_name"]] += 1

    # ── helpers ────────────────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> Image.Image:
        """Load as RGB (BUSI PNGs are grayscale — we replicate to 3 channels
        so ImageNet-pretrained backbones work without modification)."""
        return Image.open(path).convert("RGB")

    def _load_mask(self, mask_paths: list[Path]) -> Image.Image | None:
        """
        Merge multiple masks with a logical OR into a single binary mask.
        Returns None if no masks exist (e.g. some normal images).
        """
        if not mask_paths:
            return None

        import numpy as np
        merged = None
        for mp in mask_paths:
            m = Image.open(mp).convert("L")  # grayscale
            arr = (TF.to_tensor(m) > 0).squeeze(0).numpy()
            merged = arr if merged is None else (merged | arr)

        # Return as PIL for consistent transform handling
        return Image.fromarray((merged * 255).astype("uint8"), mode="L")

    # ── core interface ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = self._load_image(sample["image_path"])
        label = sample["label"]

        # Resize before any transform
        image = TF.resize(image, [self.image_size, self.image_size])

        if self.transform is not None:
            image = self.transform(image)

        if self.return_mask:
            mask = self._load_mask(sample["mask_paths"])
            if mask is not None:
                mask = TF.resize(mask, [self.image_size, self.image_size])
                mask = TF.to_tensor(mask)  # [1, H, W], values in {0, 1}
            else:
                # Normal class — return empty mask
                mask = torch.zeros(1, self.image_size, self.image_size)
            return image, mask, label

        return image, label

    # ── convenience ────────────────────────────────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """
        Returns inverse-frequency weights per class — useful for
        WeightedRandomSampler to handle class imbalance.

        Usage:
            weights = dataset.class_weights()
            sample_weights = [weights[s["label"]] for s in dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(dataset))
        """
        total = len(self.samples)
        weights = []
        for label in range(len(CLASS_TO_IDX)):
            class_name = IDX_TO_CLASS[label]
            count = self.class_counts[class_name]
            weights.append(total / (len(CLASS_TO_IDX) * count))
        return torch.tensor(weights, dtype=torch.float)

    def __repr__(self) -> str:
        lines = [
            f"BUSIDataset | root: {self.root}",
            f"  Total samples : {len(self.samples)}",
        ]
        for c, n in self.class_counts.items():
            lines.append(f"  {c:12s} : {n}")
        return "\n".join(lines)

class BUSIDatasetSSL(Dataset):
    """
    Wraps BUSIDataset for self-supervised pretraining.

    __getitem__ returns (view1, view2, label).
    Label is kept for monitoring purposes — NOT used in SSL loss.

    Args:
        busi_dataset  : An instantiated BUSIDataset (transform=None).
        ssl_transform : Your dual-view augmenter from ssl_transforms.py.
    """

    def __init__(self, busi_dataset: BUSIDataset, ssl_transform):
        self.dataset       = busi_dataset
        self.ssl_transform = ssl_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Get raw PIL image (no transform applied at dataset level)
        image, label = self.dataset[idx]

        # ssl_transform returns two differently augmented views
        view1, view2 = self.ssl_transform(image)

        return view1, view2, label