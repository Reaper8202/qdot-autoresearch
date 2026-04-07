from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.model import build_model


def load_image(path: Path, image_size: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {'.npy'}:
        arr = np.load(path)
    else:
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError('Pillow is required to load standard image files.') from e
        img = Image.open(path).convert('L').resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    return arr


def connected_components(mask: np.ndarray) -> tuple[int, list[list[float]]]:
    try:
        from scipy import ndimage
    except ImportError as e:
        raise RuntimeError('scipy is required for connected component counting.') from e
    labeled, num = ndimage.label(mask.astype(np.uint8))
    centroids: list[list[float]] = []
    for label_idx in range(1, num + 1):
        ys, xs = np.where(labeled == label_idx)
        if len(xs) == 0:
            continue
        centroids.append([float(ys.mean()), float(xs.mean())])
    return len(centroids), centroids


def save_visuals(image: np.ndarray, pred_mask: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'original.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pred_mask, cmap='gray')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'pred_mask.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap='autumn', alpha=0.45)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'overlay.png', dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run qdot model inference on a single image')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = int(cfg['image_size'])
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    image = load_image(input_path, image_size)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor.to(device))
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    dot_count, centroids = connected_components(pred_mask)
    save_visuals(image, pred_mask, output_dir)

    result = {
        'input_path': str(input_path),
        'image_size': image_size,
        'dot_count': dot_count,
        'centroids': centroids,
        'artifacts': {
            'original': str(output_dir / 'original.png'),
            'pred_mask': str(output_dir / 'pred_mask.png'),
            'overlay': str(output_dir / 'overlay.png'),
        },
    }
    (output_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
