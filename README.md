# qdot-autoresearch

Plain PyTorch segmentation baseline for synthetic quantum-dot images. Designed for autonomous experimentation in Google Colab or locally.

## Goal

Achieve >90% IoU on synthetic segmentation of gaussian-blob quantum-dot images. No real-world data, no framework dependencies beyond PyTorch.

## Setup

```bash
uv sync
```

## GUI (generate & predict)

```bash
streamlit run src/gui.py
```

Opens a local web app at `http://localhost:8501`. Click **Generate & Predict** to:
- Generate a random synthetic quantum-dot image
- Run the trained U-Net segmentation model on it
- View the original image, true mask, predicted mask, overlay, and dot counts side-by-side

Requires a trained checkpoint at `runs/baseline/model.pt` (see **Train** below).

## Train

```bash
python -m src.train --config configs/baseline.yaml
```

Per-epoch JSON is printed to stdout. Final checkpoint and `metrics.json` are saved to `runs/baseline/`.

## Eval

```bash
python -m src.eval \
  --config configs/baseline.yaml \
  --checkpoint runs/baseline/model.pt \
  --output runs/baseline/eval.json
```

## Predict on one image

```bash
python -m src.predict \
  --config configs/baseline.yaml \
  --checkpoint runs/baseline/model.pt \
  --input path/to/image.png \
  --output-dir runs/predictions/example
```

This writes:
- `original.png`
- `pred_mask.png`
- `overlay.png`
- `result.json` with the predicted dot count and centroids

## Config

Edit `configs/baseline.yaml` to tune:

| Key | Description |
|---|---|
| `seed` | Global RNG seed |
| `image_size` | Image dimensions (square) |
| `train_size` / `val_size` | Dataset size |
| `batch_size` | Batch size |
| `max_epochs` | Training epochs |
| `learning_rate` | Adam LR |
| `class_weights` | CrossEntropy weights `[bg, fg]` |
| `encoder_channels` | U-Net feature channels per level |
| `synthetic.*` | Dot count, radius, noise parameters |

## Architecture

- **Data**: Gaussian blobs on noisy background, numpy-generated, seeded per sample
- **Model**: U-Net with configurable depth (encoder_channels), BatchNorm, ReLU
- **Loss**: CrossEntropyLoss with class weights (dots are sparse)
- **Metrics**: `val_pixel_accuracy`, `val_iou`

## Colab

See `colab/README.md` for notebook workflow.
