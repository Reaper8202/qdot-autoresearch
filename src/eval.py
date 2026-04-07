from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_dataset
from src.metrics import segmentation_metrics
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate qdot segmentation checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    val_ds = build_dataset(cfg, split="val", seed=int(cfg.get("seed", 0)) + 1)
    loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    all_metrics = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_metrics.append(segmentation_metrics(logits.cpu(), labels))

    out = {
        "val_pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in all_metrics])),
        "val_iou": float(np.mean([m["iou"] for m in all_metrics])),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
