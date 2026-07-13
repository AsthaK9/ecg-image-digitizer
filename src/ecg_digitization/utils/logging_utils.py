"""Logging and experiment-tracking setup.

Centralizes two independent concerns:
  1. Application/system logging via the standard library `logging` module,
     with a Rich-formatted console handler and an optional rotating file handler.
  2. Experiment tracking backends (TensorBoard, Weights & Biases), initialized
     from the resolved Hydra config so training code never touches backend
     SDKs directly — it calls `ExperimentLogger.log_metrics(...)` and this
     module fans it out to whichever backends are enabled.

Keeping these fanned out through a single `ExperimentLogger` means swapping,
disabling, or adding a tracking backend later touches this file only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler


def setup_logging(cfg: DictConfig) -> None:
    """Configure root Python logging from the resolved `logging` config group.

    Must be called once, early in every entrypoint (train.py, predict.py,
    evaluate.py), before any module-level `logging.getLogger(__name__)`
    calls emit output.
    """
    log_level = getattr(logging, str(cfg.logging.log_level).upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    ]

    if cfg.logging.log_to_file:
        log_dir = Path(cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(logging.Formatter(cfg.logging.log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized at level %s", cfg.logging.log_level)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))


class ExperimentLogger:
    """Fans out scalar metrics, images, and config to enabled tracking backends.

    Usage:
        exp_logger = ExperimentLogger(cfg)
        exp_logger.log_metrics({"train/loss": 0.42}, step=global_step)
        exp_logger.log_image("val/sample_0_overlay", image_array, step=epoch)
        exp_logger.close()
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._logger = logging.getLogger(self.__class__.__name__)

        self._tb_writer = None
        self._wandb_run = None

        if cfg.logging.tensorboard.enabled:
            self._init_tensorboard()

        if cfg.logging.wandb.enabled:
            self._init_wandb()

    def _init_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(self.cfg.logging.tensorboard.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=self.cfg.logging.tensorboard.flush_secs,
        )
        self._logger.info("TensorBoard logging enabled -> %s", log_dir)

    def _init_wandb(self) -> None:
        import wandb

        wandb_cfg = self.cfg.logging.wandb
        self._wandb_run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.name,
            mode=wandb_cfg.mode,
            tags=list(wandb_cfg.tags),
            notes=wandb_cfg.notes,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        self._logger.info(
            "Weights & Biases logging enabled -> project=%s mode=%s",
            wandb_cfg.project,
            wandb_cfg.mode,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics to every enabled backend."""
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=step)

        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def log_image(self, tag: str, image: Any, step: int | None = None) -> None:
        """Log a single image (e.g. predicted-vs-true waveform overlay)."""
        if self._tb_writer is not None:
            # Expects CHW format, float in [0, 1] or uint8.
            self._tb_writer.add_image(tag, image, global_step=step, dataformats="HWC")

        if self._wandb_run is not None:
            import wandb

            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_hparams(params, {})

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so modules don't import `logging` directly everywhere."""
    return logging.getLogger(name)


def fail_fast_on_unhandled_exceptions() -> None:
    """Ensure uncaught exceptions are logged (not silently swallowed by some
    notebook/orchestration contexts) before the process exits.
    """

    def _handle(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        logging.getLogger("uncaught").critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = _handle
