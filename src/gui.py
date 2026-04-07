"""
Streamlit GUI for qdot segmentation.

Launch:
    streamlit run src/gui.py
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.config import load_config
from src.data import QDotDataset
from src.model import build_model
from src.predict import connected_components

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_CHECKPOINT_PATH = _PROJECT_ROOT / "runs" / "baseline" / "model.pt"


@st.cache_resource
def _load_model():
    cfg = load_config(_CONFIG_PATH)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = build_model(cfg)
    state = torch.load(_CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, device


def _generate_and_predict(model, cfg, device, seed: int):
    dataset = QDotDataset(cfg, split="val", seed=seed)
    image_t, mask_t = dataset[0]  # [1,H,W], [H,W]

    image_np = image_t.squeeze(0).numpy()       # [H,W] float32 in [0,1]
    true_mask_np = mask_t.numpy().astype(np.float32)  # [H,W] 0 or 1

    true_count, _ = connected_components(true_mask_np)

    tensor = image_t.unsqueeze(0).to(device)   # [1,1,H,W]
    with torch.no_grad():
        logits = model(tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    pred_count, centroids = connected_components(pred_mask)
    return image_np, true_mask_np, pred_mask, true_count, pred_count, centroids


def _overlay_figure(image_np: np.ndarray, pred_mask: np.ndarray):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_np, cmap="gray", vmin=0, vmax=1)
    ax.imshow(
        np.ma.masked_where(pred_mask == 0, pred_mask),
        cmap="autumn",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def main():
    st.set_page_config(page_title="QDot Segmentation", layout="wide")
    st.title("QDot Segmentation")
    st.caption(
        "Generates a synthetic quantum-dot image and runs the trained U-Net. "
        "Each click uses a fresh random seed."
    )

    if not _CHECKPOINT_PATH.exists():
        st.error(
            f"Checkpoint not found: `{_CHECKPOINT_PATH}`\n\n"
            "Train the model first:\n```\npython -m src.train --config configs/baseline.yaml\n```"
        )
        return

    model, cfg, device = _load_model()
    st.caption(f"Device: `{device}` | Config: `{_CONFIG_PATH.name}`")

    if st.button("Generate & Predict", type="primary"):
        seed = random.randint(0, 2**31 - 1)
        with st.spinner("Running inference…"):
            image_np, true_mask, pred_mask, true_count, pred_count, centroids = (
                _generate_and_predict(model, cfg, device, seed)
            )

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted dots", pred_count)
        m2.metric("True dots", true_count)
        m3.metric("Seed", seed)

        # Images — four columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Original**")
            st.image(image_np, clamp=True, use_container_width=True)
        with c2:
            st.markdown("**True mask**")
            st.image(true_mask, clamp=True, use_container_width=True)
        with c3:
            st.markdown("**Predicted mask**")
            st.image(pred_mask, clamp=True, use_container_width=True)
        with c4:
            st.markdown("**Overlay**")
            fig = _overlay_figure(image_np, pred_mask)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


if __name__ == "__main__":
    main()
