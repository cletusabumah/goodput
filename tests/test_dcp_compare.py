"""Ticket 4.4 — naive / incremental vs DCP latency note."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodput.evaluation.dcp_compare import (
    DCP_ROW_FIELDS,
    load_dcp_compare_yaml,
    render_dcp_table,
    run_dcp_compare,
)


def test_load_committed_dcp_compare_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_dcp_compare_yaml(root / "experiments" / "dcp-compare.yaml")
    assert spec.name == "dcp-compare"
    assert spec.repeats == 3
    assert spec.hidden_size == 256


def test_load_dcp_compare_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmode: dcp_compare\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown dcp-compare keys"):
        load_dcp_compare_yaml(path)


def test_run_dcp_compare_writes_json_and_markdown(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny-dcp.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: tiny-dcp",
                "mode: dcp_compare",
                "repeats: 1",
                "input_size: 16",
                "hidden_size: 32",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
                f"work_dir: {tmp_path / 'work'}",
                f"output_dir: {tmp_path / 'artifacts' / 'sweeps' / 'tiny-dcp'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_dcp_compare_yaml(cfg)
    result = run_dcp_compare(spec)
    assert result.json_path is not None and result.table_path is not None
    assert result.json_path.is_file()
    assert result.table_path.is_file()
    assert len(result.rows) == 3
    paths = [row["path"] for row in result.rows]
    assert paths == [
        "naive_torch_save",
        "incremental_model_only",
        "torch_distributed_checkpoint",
    ]
    naive, incremental, dcp = result.rows
    assert naive["available"] is True
    assert incremental["available"] is True
    assert naive["save_s"] is not None and naive["save_s"] > 0
    assert incremental["save_s"] is not None and incremental["save_s"] > 0
    assert naive["bytes"] > 0
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dcp_compare"
    assert "not a production dcp integration" in payload["writeup"].lower()
    markdown = result.table_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Naive vs incremental vs torch.distributed.checkpoint")
    for key in DCP_ROW_FIELDS:
        assert key in markdown
    assert "naive_torch_save" in markdown
    if dcp["available"]:
        assert dcp["save_s"] is not None
    else:
        notes = str(dcp["notes"]).lower()
        assert "skipped" in notes or "not importable" in notes


def test_render_dcp_table_includes_writeup() -> None:
    text = render_dcp_table(
        [
            {
                "path": "naive_torch_save",
                "available": True,
                "save_s": 0.01,
                "restore_s": 0.02,
                "bytes": 100,
                "notes": "pickle",
            }
        ]
    )
    assert "not a production dcp integration" in text.lower()
    assert "0.010000" in text


def test_cli_dcp_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "cli-dcp.yaml"
    out = tmp_path / "artifacts"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-dcp",
                "mode: dcp_compare",
                "repeats: 1",
                "input_size: 8",
                "hidden_size: 16",
                f"artifacts_dir: {out}",
                f"work_dir: {tmp_path / 'work'}",
                f"output_dir: {out / 'sweeps' / 'cli-dcp'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--dcp-compare", str(cfg)])
    assert code == 0
    table_dir = out / "sweeps" / "cli-dcp"
    assert (table_dir / "dcp_compare.json").is_file()
    assert (table_dir / "table.md").is_file()
