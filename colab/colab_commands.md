# Google Colab T4 commands

## 1. Optional: mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clone the repo

```bash
%cd /content
!git clone https://github.com/Reaper8202/qdot-autoresearch.git
%cd /content/qdot-autoresearch
```

## 3. Install dependencies

```bash
!pip install -U pip
!pip install uv
!uv sync
```

If `uv sync` gives you trouble in Colab, use this fallback:

```bash
!pip install -e .
```

## 4. Recommended first experiment

Use the 0-200 dense regime before trying anything broader.

```bash
!./.venv/bin/python -m src.train --config configs/high_density_0_200_640_t4.yaml
```

## 5. Evaluate the trained model

```bash
!./.venv/bin/python -m src.eval \
  --config configs/high_density_0_200_640_t4.yaml \
  --checkpoint runs/high_density_0_200_640_t4/model.pt \
  --output runs/high_density_0_200_640_t4/eval.json
```

## 6. Optional second experiment

Only try this after checking the 0-200 results.

```bash
!./.venv/bin/python -m src.train --config configs/high_density_0_500_640_t4.yaml
```

## 7. Optional: copy outputs to Google Drive

```bash
!mkdir -p /content/drive/MyDrive/qdot_autoresearch_runs
!cp -r runs/high_density_0_200_640_t4 /content/drive/MyDrive/qdot_autoresearch_runs/
```

## Notes

- The earlier 0-999 at 640x640 regime was unstable locally and is not the main recommendation.
- Start with 0-200, inspect IoU, and only then consider 0-500.
- 95% IoU across 0-999 is not a realistic assumption without further redesign.
