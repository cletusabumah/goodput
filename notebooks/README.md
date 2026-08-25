# Notebooks

Exploration and demos only. Training, checkpointing, and metrics live in `src/goodput/`. **Do not import from notebooks** in library code or CI.

| File | Purpose |
|------|---------|
| [`colab_gpu_demo.ipynb`](colab_gpu_demo.ipynb) | Ticket 4.2 — short GPU (or CPU fallback) train in Google Colab |

## Colab GPU demo (ticket 4.2)

1. Open [`colab_gpu_demo.ipynb`](colab_gpu_demo.ipynb) in [Colab](https://colab.research.google.com/) (upload, or **Open in Colab** from GitHub).
2. **Runtime → Change runtime type → T4 GPU**. If the GPU quota is exhausted, CPU still works — `train_from_settings` falls back when CUDA is missing.
3. Run all cells. Knobs match [`experiments/colab.yaml`](../experiments/colab.yaml): 12 steps, seed 42, naive ckpt every 4 steps.
4. You should see finite `final_loss`, `goodput` in `[0, 1]`, and `artifacts/reports/colab-gpu-demo/report.json`.

Local clone (no Colab):

```bash
goodput-run --config experiments/colab.yaml
```

That YAML asks for `device: cuda`; without a GPU the trainer uses CPU. CI never starts Colab; `tests/test_colab_demo.py` checks YAML ↔ notebook knobs and a CPU train of the same settings.
