"""Training entrypoint.

This is currently a scaffold: it proves out the full configuration,
logging, and seeding stack end-to-end (Hydra composition -> logging setup ->
seeding -> experiment tracker init) with NO model or training loop attached
yet. Once a model exists (see src/ecg_digitization/models/registry.py), the
actual training loop will be wired in here via
`ecg_digitization.training.trainer.Trainer`.

Usage:
    python scripts/train.py
    python scripts/train.py experiment=exp001_baseline
    python scripts/train.py data.dataloader.batch_size=64 training.epochs=10
    python scripts/train.py --multirun training.optimizer.lr=1e-3,5e-4,1e-4
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ecg_digitization.utils.logging_utils import (
    ExperimentLogger,
    fail_fast_on_unhandled_exceptions,
    setup_logging,
)
from ecg_digitization.utils.seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fail_fast_on_unhandled_exceptions()
    setup_logging(cfg)

    logger.info("Starting run '%s' (project=%s)", cfg.run_name, cfg.project_name)
    logger.info("Full resolved config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed, deterministic=True)

    exp_logger = ExperimentLogger(cfg)
    exp_logger.log_hyperparams(
        {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
         if isinstance(v, (int, float, str, bool))}
    )

    if cfg.model.architecture is None:
        logger.warning(
            "No model architecture configured (model.architecture=null). "
            "Config, logging, and seeding pipeline verified successfully; "
            "stopping here until a model is implemented (see "
            "src/ecg_digitization/models/registry.py)."
        )
        exp_logger.close()
        return

    # ------------------------------------------------------------------
    # Once a model exists, this is where the real flow attaches:
    #
    #   train_loader, val_loader = build_dataloaders(cfg)
    #   model = build_model(cfg.model)
    #   trainer = Trainer(cfg, model, exp_logger)
    #   trainer.fit(train_loader, val_loader)
    # ------------------------------------------------------------------
    raise NotImplementedError(
        "Model architecture is set but the training loop is not yet implemented."
    )


if __name__ == "__main__":
    main()
