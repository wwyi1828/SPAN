import json
import importlib.util
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from lib.utils.logging import LoggerManager, LoggingConfig, WandBConfig

HAS_WANDB = importlib.util.find_spec("wandb") is not None


def _patch_wandb_settings_serializer() -> None:
    """
    WandB's default Settings uses Sequence[str] fields backed by tuples, which
    triggers pydantic serializer warnings during model_dump(). Patch the method
    once to silence the warning and coerce those tuples to lists for logging.
    """
    try:
        from wandb.sdk.wandb_settings import Settings
    except Exception:
        return

    if getattr(Settings.model_dump, "_span_patched", False):
        return

    original_model_dump = Settings.model_dump

    def model_dump_no_warn(self, *args, **kwargs):
        kwargs.setdefault("warnings", False)
        data = original_model_dump(self, *args, **kwargs)
        for key in ("ignore_globs", "x_stats_disk_paths"):
            if isinstance(data.get(key), tuple):
                data[key] = list(data[key])
        return data

    setattr(model_dump_no_warn, "_span_patched", True)
    Settings.model_dump = model_dump_no_warn


if HAS_WANDB:
    _patch_wandb_settings_serializer()

def _to_container(cfg: DictConfig) -> Dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(data, dict):
        data.pop("hydra", None)
        # Convert tuples to lists via JSON serialization to fix Pydantic warnings
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
