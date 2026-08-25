"""Ticket 4.2 — Colab demo YAML + notebook knobs stay in lockstep; CPU-safe."""

from __future__ import annotations

import json
from pathlib import Path

from goodput.config import Settings
from goodput.experiments import load_experiment_yaml
from goodput.metrics import REQUIRED_REPORT_FIELDS, build_run_report
from goodput.providers import LocalFsCheckpointStore
from goodput.training import train_from_settings

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_gpu_demo.ipynb"
COLAB_YAML = ROOT / "experiments" / "colab.yaml"

# Documented demo knobs (notebook markdown table + Settings() cell).
DEMO_KNOBS = {
    "seed": 42,
    "steps": 12,
    "batch_size": 8,
    "input_size": 16,
    "hidden_size": 32,
    "learning_rate": 0.01,
    "ckpt_interval": 4,
    "ckpt_mode": "naive",
    "num_workers": 1,
}


def test_colab_yaml_matches_documented_knobs() -> None:
    spec = load_experiment_yaml(COLAB_YAML)
    assert spec.mode == "train"
    assert spec.name == "colab-gpu-demo"
    s = spec.settings
    for key, expected in DEMO_KNOBS.items():
        assert getattr(s, key) == expected
    assert s.device == "cuda"  # Colab T4 path; train_from_settings falls back to CPU
    assert s.num_workers == 1


def test_notebook_exists_and_documents_settings() -> None:
    assert NOTEBOOK.is_file()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    markdown = "\n".join(cells[0]["source"])
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "ticket 4.2" in markdown.lower() or "4.2" in markdown
    assert "experiments/colab.yaml" in markdown
    assert "T4" in markdown
    for key, value in DEMO_KNOBS.items():
        assert str(value) in markdown
        assert f"{key}=" in code or f"{key} =" in code
    assert "train_from_settings" in code
    assert "git+https://github.com/cletusabumah/goodput.git" in code
    # Must not reimplement the trainer in the notebook.
    assert "class ToyMLP" not in code


def test_colab_settings_train_on_cpu(tmp_path: Path) -> None:
    """CI has no GPU; train_from_settings must still complete with device=cpu."""
    settings = Settings(
        run_name="colab-ci",
        seed=DEMO_KNOBS["seed"],
        num_workers=1,
        steps=DEMO_KNOBS["steps"],
        batch_size=DEMO_KNOBS["batch_size"],
        input_size=DEMO_KNOBS["input_size"],
        hidden_size=DEMO_KNOBS["hidden_size"],
        learning_rate=DEMO_KNOBS["learning_rate"],
        ckpt_interval=DEMO_KNOBS["ckpt_interval"],
        ckpt_mode="naive",
        ckpt_dir=tmp_path / "ckpts",
        device="cpu",
        artifacts_dir=tmp_path / "artifacts",
        ci_mode=True,
    )
    store = LocalFsCheckpointStore(settings.ckpt_dir)
    result = train_from_settings(settings, checkpoint_store=store)
    assert result.ok
    assert result.steps_completed == 12
    assert result.device == "cpu"
    report = build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=result.steps_completed,
        ckpt_save_seconds=result.ckpt_save_seconds,
        ckpt_restore_seconds=result.ckpt_restore_seconds,
        final_loss=result.final_loss,
    )
    for key in REQUIRED_REPORT_FIELDS:
        assert key in report
    assert 0.0 <= report["goodput"] <= 1.0
