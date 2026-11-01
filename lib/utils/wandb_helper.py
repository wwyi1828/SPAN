import json
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from lib.utils.logging import LoggerManager, LoggingConfig, WandBConfig

def _to_container(cfg: DictConfig) -> Dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(data, dict):
        data.pop("hydra", None)
        # Keep the payload JSON-compatible without modifying W&B internals.
        return json.loads(json.dumps(data))
    return {}


def create_logger(cfg: DictConfig) -> Optional[LoggerManager]:
    logging_cfg = cfg.get("logging")
    if logging_cfg is None:
        return None
    results_cfg = logging_cfg.get("results")
    wandb_cfg = logging_cfg.get("wandb")
    if results_cfg is None or wandb_cfg is None:
        return None

    results_dir = results_cfg.get("dir")
    if not results_dir:
        return None

    wandb_config = WandBConfig(
        enabled=bool(wandb_cfg.get("enabled", True)),
        project=str(wandb_cfg.get("project", "SPAN")),
        entity=wandb_cfg.get("entity"),
        tags=list(wandb_cfg.get("tags", []) or []),
        notes=str(wandb_cfg.get("notes", "")),
    )
    logger_config = LoggingConfig(
        mode=str(logging_cfg.get("mode", "best")),
        wandb=wandb_config,
        results_dir=str(results_dir),
    )
    training_config = _to_container(cfg)
    return LoggerManager(logger_config, training_config)
