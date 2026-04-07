from __future__ import annotations

import torch


def segmentation_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """
    Args:
        logits: [B, 2, H, W] raw model output
        labels: [B, H, W] long tensor with class indices {0, 1}
    """
    pred = torch.argmax(logits, dim=1)  # [B, H, W]
    true = labels                        # [B, H, W]

    pixel_accuracy = (pred == true).float().mean().item()

    pred_fg = pred == 1
    true_fg = true == 1
    intersection = (pred_fg & true_fg).sum().item()
    union = (pred_fg | true_fg).sum().item()
    iou = 1.0 if union == 0 else intersection / union

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "iou": float(iou),
    }
