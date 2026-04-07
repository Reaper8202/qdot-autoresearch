from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_dataset
from src.metrics import segmentation_metrics
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses, all_metrics = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            all_metrics.append(segmentation_metrics(logits.cpu(), labels.cpu()))

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in all_metrics])) if all_metrics else float("nan"),
        "val_iou": float(np.mean([m["iou"] for m in all_metrics])) if all_metrics else float("nan"),
    }


def train(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(cfg, split="train", seed=seed)
    val_ds = build_dataset(cfg, split="val", seed=seed + 1)
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=num_workers)

    device = _select_device()
    model = build_model(cfg).to(device)

    class_weights = cfg.get("class_weights")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    history = []
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        batch_losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            **val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "final": history[-1] if history else {}}, f, indent=2)

    return history[-1] if history else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train qdot segmentation model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
