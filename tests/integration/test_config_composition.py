"""Verifies the Hydra config groups compose and fully resolve without error.

This is the load-bearing test for the config system itself: if someone adds
a new config group, renames a key, or breaks an interpolation, this test
fails fast in CI instead of surfacing three layers deep in a training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytestmark = pytest.mark.integration

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())


def test_default_config_composes_and_resolves() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
        # to_container(resolve=True) forces every ${...} interpolation to
        # evaluate — this is what catches broken/circular references.
        resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["project_name"] == "ecg-digitization"
    assert resolved["data"]["dataloader"]["batch_size"] == 16
    assert resolved["model"]["architecture"] is None
    assert resolved["training"]["optimizer"]["name"] == "adamw"
    assert resolved["logging"]["wandb"]["project"] == "ecg-digitization"


def test_experiment_override_composes_correctly() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=exp001_baseline"])
        resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["run_name"] == "exp001_baseline"
    assert resolved["training"]["epochs"] == 50
    assert resolved["training"]["optimizer"]["lr"] == 5.0e-4
    assert resolved["data"]["dataloader"]["batch_size"] == 32
    assert "baseline" in resolved["logging"]["wandb"]["tags"]


def test_cli_style_override_wins_over_defaults() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["data.dataloader.batch_size=64", "training.epochs=5"],
        )
    assert cfg.data.dataloader.batch_size == 64
    assert cfg.training.epochs == 5


def test_env_paths_resolve_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECG_DATA_ROOT", "/tmp/custom_data")
    monkeypatch.setenv("ECG_ARTIFACTS_ROOT", "/tmp/custom_artifacts")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.env.data_root == "/tmp/custom_data"
    assert cfg.data.root_dir == "/tmp/custom_data"
    assert cfg.training.checkpoint.dir.startswith("/tmp/custom_artifacts/checkpoints/")
