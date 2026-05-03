import hydra
from omegaconf import DictConfig

from lib.utils.entrypoint import initialize_hydra_run
from tasks.vision.shared.data import DatasetBundle, prepare_segmentation_dataset
from .trainer import run_training, Metrics
from tasks.vision.shared.features import resolve_feature_dim
from lib.utils.model_utils import format_token_init_types
from lib.utils.logging import save_metrics

SEGMENTATION_METRICS = ["f1", "iou", "recall", "precision"]


def build_run_signature(cfg: DictConfig) -> str:
    token_init_str = format_token_init_types(cfg.model.get('token_init_types', ['max', 'mean', 'fix1e-4']))
    return (
        f"{cfg.get('features_variant', 'R50')}_"
        f"{cfg.model.econvs_type}_"
        f"{cfg.model.slide_configs}_"
        f"cf{cfg.model.channel_factor}_"
        f"{cfg.model.trans_type}_"
        f"{token_init_str}"
    )


def save_results(cfg: DictConfig, metrics: Metrics) -> None:
    """Save segmentation results using the unified logging interface."""
    save_path = cfg.logging.results.get("save_path")
    if save_path is None:
        return

    metrics_dict = {name: getattr(metrics, name) for name in SEGMENTATION_METRICS}

    save_metrics(
        dataset=cfg.dataset,
        run_signature=build_run_signature(cfg),
        seed=cfg.seed,
        metrics=metrics_dict,
        results_dir=cfg.logging.results.dir,
        filename=save_path,
        metric_names=SEGMENTATION_METRICS
    )


@hydra.main(version_base=None, config_path="../../../../configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    def _sync_features_dim(run_cfg: DictConfig) -> None:
        # Sync features_dim with features_variant (PLIP=768, V2=1280, CONCH=512, default=1024)
        run_cfg.features_dim = resolve_feature_dim(run_cfg.get('features_variant', 'R50'))

    cfg = initialize_hydra_run(
        cfg,
        "SPAN Patch Segmentation - Hydra Configuration",
        seed=cfg.seed,
        mutate_cfg=_sync_features_dim,
    )
    dataset_bundle: DatasetBundle = prepare_segmentation_dataset(cfg)
    best_metrics = run_training(cfg, dataset_bundle)
    save_results(cfg, best_metrics)
    print("\n" + "=" * 60)
    print("Best Test Metrics")
    print("=" * 60)
    print(f"F1: {best_metrics.f1:.4f}")
    print(f"IoU: {best_metrics.iou:.4f}")
    print(f"Recall: {best_metrics.recall:.4f}")
    print(f"Precision: {best_metrics.precision:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
