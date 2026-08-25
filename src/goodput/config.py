"""Environment-driven settings (no hardcoded paths or credentials)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOODPUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_workers: int = Field(default=2, ge=1)
    # Compose node id (ticket 2.4). Only rank 0 writes checkpoints so workers
    # sharing a volume do not clobber latest.pt. Local multi-process ignores this.
    rank: int = Field(default=0, ge=0)
    seed: int = 42

    steps: int = Field(default=100, ge=1)
    batch_size: int = Field(default=8, ge=1)
    input_size: int = Field(default=16, ge=1)
    hidden_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)

    ckpt_interval: int = Field(default=10, ge=0)
    ckpt_mode: Literal["naive", "incremental"] = "naive"
    # How often incremental mode writes a full model+optimizer base (ticket 2.1).
    ckpt_full_every: int = Field(default=4, ge=1)
    ckpt_dir: Path = Path("artifacts/checkpoints")

    fault_mode: Literal["none", "kill", "hang", "bitflip"] = "none"
    fault_at: str = "random"
    fault_mean_interval: int = Field(default=0, ge=0)
    # Parent polls rank-0 progress; stall at a checkpointed step ⇒ hang (ticket 3.1).
    health_check_timeout_s: float = Field(default=3.0, gt=0)
    # Cross-rank grad norm ratio detector after a simulated bit-flip (ticket 3.2).
    bitflip_detect: bool = True
    bitflip_grad_ratio_threshold: float = Field(default=8.0, gt=1.0)

    artifacts_dir: Path = Path("artifacts")
    run_name: str = "local-dev"
    report_format: Literal["json", "csv", "both"] = "json"

    checkpoint_provider: Literal["mock", "local_fs"] = "local_fs"
    fault_provider: Literal["mock", "process"] = "mock"
    metrics_provider: Literal["json_file", "stdout", "mock"] = "json_file"

    tracker: Literal["none", "mlflow", "wandb"] = "none"
    # Local file: URI by default — no MLflow cloud account required.
    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    wandb_project: str = "goodput"
    ci_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
