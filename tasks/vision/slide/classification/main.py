import hydra
import hashlib
import json
from omegaconf import DictConfig, OmegaConf

from lib.utils.entrypoint import initialize_hydra_run
from tasks.vision.shared.data import DatasetBundle, prepare_classification_dataset
from .trainer import run_training, Metrics
from lib.utils.model_utils import format_token_init_types
from lib.utils.logging import save_metrics

CLASSIFICATION_METRICS = [
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "avg_auc",
]


def build_run_signature(cfg: DictConfig) -> str:
    token_init_str = format_token_init_types(cfg.model.get('token_init_types', ['fix1e-3']))
    enc_act = cfg.model.get('enc_act', '')
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    cls_cfg = OmegaConf.to_container(cfg.classification, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    signature_payload = {
        "features_variant": cfg.features_variant,
        "model": model_cfg,
        "classification": cls_cfg,
        "training": {
            "lr": training_cfg.get("lr"),
            "weight_decay": training_cfg.get("weight_decay"),
            "early_stop_patience": training_cfg.get("early_stop_patience"),
        },
    }
    payload_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:10]
    return (
        f"{cfg.features_variant}_"
        f"{cfg.model.econvs_type}_"
        f"{enc_act}_"
        f"{cfg.model.slide_configs}_"
        f"{cfg.model.global_strategy}_"
        f"{cfg.classification.head_div}_"
        f"{token_init_str}_"
        f"h{signature_hash}"
    )

def save_results(cfg: DictConfig, metrics: Metrics) -> None:
    """Save classification results using the unified logging interface."""
    save_path = cfg.logging.results.get("save_path")
    if save_path is None:
        return

    metrics_dict = {name: getattr(metrics, name) for name in CLASSIFICATION_METRICS}

    save_metrics(
        dataset=cfg.dataset,
        run_signature=build_run_signature(cfg),
        seed=cfg.seed,
        metrics=metrics_dict,
        results_dir=cfg.logging.results.dir,
        filename=save_path,
        metric_names=CLASSIFICATION_METRICS,
        extra_fields={"aucs": list(metrics.aucs)}
    )


@hydra.main(version_base=None, config_path="../../../../configs", config_name="classification")
def main(cfg: DictConfig) -> None:
    cfg = initialize_hydra_run(
        cfg,
        "SPAN Slide Classification - Hydra Configuration",
        seed=cfg.seed,
    )
    dataset_bundle: DatasetBundle = prepare_classification_dataset(cfg)
    best_metrics = run_training(cfg, dataset_bundle)
    save_results(cfg, best_metrics)
    print("\n" + "=" * 60)
    print("Best Test Metrics")
    print("=" * 60)
    print(f"Accuracy: {best_metrics.accuracy:.4f}")
    print(f"Macro F1: {best_metrics.macro_f1:.4f}")
    print(f"Macro Precision: {best_metrics.macro_precision:.4f}")
    print(f"Macro Recall: {best_metrics.macro_recall:.4f}")
    print(f"AUCs: {best_metrics.aucs}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
