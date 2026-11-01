import os
import json
import fcntl
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
import logging

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)

@dataclass
class WandBConfig:

    enabled: bool = True
    project: str = "SPAN"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class LoggingConfig:

    mode: str = "best"
    wandb: WandBConfig = field(default_factory=WandBConfig)
    results_dir: str = "results"

class LoggerManager:

    def __init__(self, cfg: LoggingConfig, training_config: Dict[str, Any]):

        self.log_mode = cfg.mode
        self.wandb_config = cfg.wandb
        self.results_dir = cfg.results_dir
        self.training_config = training_config

        self.run = None
        self.epoch_logs = []
        self.best_metrics = {}

        os.makedirs(self.results_dir, exist_ok=True)

        if HAS_WANDB and self.wandb_config.enabled:
            self._init_wandb()
        else:
            logger.warning("W&B not available or disabled")

    def _init_wandb(self):

        if not HAS_WANDB:
            logger.warning("wandb not installed")
            return

        try:
            if wandb.run is not None:
                logger.info(f"Using existing W&B run: {wandb.run.name}")
                self.run = wandb.run
            else:
                wandb_config = self._flatten_config(self.training_config)
                self.run = wandb.init(
                    project=self.wandb_config.project,
                    entity=self.wandb_config.entity,
                    config=wandb_config,
                    tags=self.wandb_config.tags,
                    notes=self.wandb_config.notes,
                    mode="offline" if self.log_mode == "disabled" else "online"
                )
                logger.info(f"W&B initialized: {self.wandb_config.project}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.run = None

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):

        log_data = {"epoch": epoch, **metrics}
        self.epoch_logs.append(log_data)

        # Send epoch metrics whenever W&B is enabled.
        if self.run:
            try:
                wandb.log(log_data)
            except Exception as e:
                logger.error(f"Failed to log epoch {epoch}: {e}")

    def log_best(self, metrics: Dict[str, Any], checkpoint_path: Optional[str] = None):

        self.best_metrics = metrics

        if self.run:
            try:

                best_log = {f"best_{k}": v for k, v in metrics.items()}
                wandb.log(best_log)

                if checkpoint_path and os.path.exists(checkpoint_path):
                    artifact = wandb.Artifact('best_model', type='model')
                    artifact.add_file(checkpoint_path)
                    self.run.log_artifact(artifact)

            except Exception as e:
                logger.error(f"Failed to log best metrics: {e}")

        logger.info(f"Best metrics logged: {metrics}")

    def get_logs_summary(self) -> Dict[str, Any]:

        return {
            'mode': self.log_mode,
            'num_epochs_logged': len(self.epoch_logs),
            'best_metrics': self.best_metrics,
            'wandb_run': self.run.name if self.run else None,
        }

    def finish(self):

        if self.run:
            try:
                wandb.finish()
                logger.info("W&B session finished")
            except Exception as e:
                logger.error(f"Error finishing W&B: {e}")

    @staticmethod
    def _flatten_config(cfg: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:

        items = []
        for k, v in cfg.items():
            if k.startswith('_'):
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(LoggerManager._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


# ========== Unified Results & Logs Management ==========

def get_results_path(results_dir: Union[str, Path], filename: str) -> Path:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    return results_path / filename


def save_results(data: Dict[str, Any], results_dir: Union[str, Path], filename: str) -> Path:
    result_path = get_results_path(results_dir, filename)
    tmp_path = result_path.with_name(f"{result_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)
    logger.info(f"Results saved to {result_path}")
    return result_path


def update_results_file(
    results_dir: Union[str, Path],
    filename: str,
    updater: Callable[[Dict[str, Any]], None],
) -> Path:
    result_path = get_results_path(results_dir, filename)
    lock_path = result_path.with_suffix(result_path.suffix + ".lock")

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if result_path.exists():
            with open(result_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        updater(results)
        save_results(results, results_dir, filename)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return result_path


def save_metrics(
    dataset: str,
    run_signature: str,
    seed: int,
    metrics: Dict[str, float],
    results_dir: Union[str, Path],
    filename: str,
    metric_names: Optional[List[str]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    num_seeds: int = 5
) -> None:
    slot_index = int(seed)
    if slot_index < 0:
        raise ValueError("seed must be non-negative")
    slot_count = max(num_seeds, slot_index + 1)

    def mapping(parent, key):
        if key not in parent:
            parent[key] = {}
        if not isinstance(parent[key], dict):
            raise ValueError(f"Results field {key!r} must be a mapping")
        return parent[key]

    def slots(parent, key):
        if key not in parent:
            parent[key] = [None] * slot_count
        values = parent[key]
        if not isinstance(values, list):
            raise ValueError(f"Results field {key!r} must be a list")
        values.extend([None] * max(0, slot_count - len(values)))
        return values

    def _update(results: Dict[str, Any]) -> None:
        if not isinstance(results, dict):
            raise ValueError("Results file must contain a mapping")
        signature_store = mapping(mapping(results, dataset), run_signature)
        metric_store = mapping(signature_store, "metrics")
        names = metric_names if metric_names is not None else list(metrics.keys())
        for name in names:
            slots(metric_store, name)[slot_index] = metrics[name]
        if extra_fields:
            for name, value in extra_fields.items():
                slots(signature_store, name)[slot_index] = value

    update_results_file(results_dir, filename, _update)
