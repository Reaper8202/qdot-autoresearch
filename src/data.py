from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class QDotDataset(Dataset):
    """Synthetic quantum-dot dataset: gaussian blobs on noisy background."""

    def __init__(self, cfg: dict, split: str = "train", seed: int | None = None):
        self.image_size = int(cfg["image_size"])
        self.length = int(cfg["train_size"] if split == "train" else cfg["val_size"])

        s = cfg["synthetic"]
        self.min_dots = int(s["min_dots"])
        self.max_dots = int(s["max_dots"])
        self.radius_min = float(s["dot_radius_min"])
        self.radius_max = float(s["dot_radius_max"])
        self.noise_std = float(s.get("noise_std", 0.05))
        self.background = float(s.get("background", 0.05))

        base_seed = seed if seed is not None else 0
        rng = np.random.default_rng(base_seed)
        self.seeds = rng.integers(0, 2**31, size=self.length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.seeds[idx]))
        H = W = self.image_size

        image = np.full((H, W), self.background, dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.float32)

        n_dots = rng.integers(self.min_dots, self.max_dots + 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        for _ in range(n_dots):
            cx = float(rng.uniform(0, W))
            cy = float(rng.uniform(0, H))
            r = float(rng.uniform(self.radius_min, self.radius_max))
            sigma = r / 2.0
            intensity = float(rng.uniform(0.6, 1.0))

            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            gauss = intensity * np.exp(-d2 / (2.0 * sigma ** 2))
            image += gauss
            # mask covers pixels where dot contribution >= half-max
            mask = np.maximum(mask, (gauss >= 0.5 * intensity).astype(np.float32))

        # Gaussian noise
        image += rng.normal(0.0, self.noise_std, (H, W)).astype(np.float32)
        image = np.clip(image, 0.0, None)
        image /= image.max() + 1e-8

        image_t = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        mask_t = torch.from_numpy(mask).long()           # [H, W], values in {0, 1}
        return image_t, mask_t


def build_dataset(cfg: dict, *, split: str, seed: int | None = None) -> QDotDataset:
    return QDotDataset(cfg, split=split, seed=seed)
