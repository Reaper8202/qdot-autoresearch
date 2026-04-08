# Colab workflow

Use Colab as a thin runner, not as the project home.

## Recommended setup

For a Google Colab T4, start with the dense but still reasonable regime:
- `configs/high_density_0_200_640_t4.yaml`

Then, if results are strong, try:
- `configs/high_density_0_500_640_t4.yaml`

Do **not** treat `configs/high_density_0_999_640.yaml` as the default plan. That regime is closer to a stress test than a sensible first target.

## Suggested notebook steps

1. Clone your GitHub repo.
2. Install dependencies.
3. Run training with a recommended T4 config.
4. Run evaluation.
5. Save `runs/` outputs and metrics to Drive if needed.

See `colab/colab_commands.md` for copy-paste commands.
