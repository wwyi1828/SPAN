import hydra
import hashlib
import json
from omegaconf import DictConfig, OmegaConf

from lib.utils.entrypoint import initialize_hydra_run
from tasks.vision.shared.data import prepare_survival_dataset
from .trainer import run_training, Metrics
from lib.utils.model_utils import format_token_init_types
from lib.utils.logging import save_metrics

SURVIVAL_METRICS = ["c_index"]


def build_run_signature(cfg: DictConfig) -> str:
    token_init_str = format_token_init_types(cfg.model.get('token_init_types', ['fix1e-3']))
    enc_act = cfg.model.get('enc_act', '')
    edge_mode = cfg.model.get("edge_mode", "none")
    conv_bias_flag = int(bool(cfg.model.get("conv_bias", False)))
    share_qkv_flag = int(bool(cfg.model.get("share_qkv", True)))
    ff_ratio = cfg.model.get("ff_ratio", "")
    if isinstance(ff_ratio, float) and ff_ratio.is_integer():
        ff_ratio_str = str(int(ff_ratio))
    else:
        ff_ratio_str = str(ff_ratio).replace(".", "p")
    input_dropout = float(cfg.model.get("input_projection", {}).get("dropout", 0.0))
    input_drop_tag = int(round(input_dropout * 100))
    arch_tag = (
        f"{cfg.model.econvs_type}_{edge_mode}_{cfg.model.global_strategy}_"
        f"cb{conv_bias_flag}_sq{share_qkv_flag}_ff{ff_ratio_str}_drop{input_drop_tag:03d}"
    )
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    survival_cfg = OmegaConf.to_container(cfg.survival, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    signature_payload = {
        "features_variant": cfg.features_variant,
        "model": model_cfg,
        "survival": survival_cfg,
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
        f"{arch_tag}_"
        f"{enc_act}_"
        f"{cfg.model.slide_configs}_"
        f"{token_init_str}_"
        f"h{signature_hash}"
    )


def save_results(cfg: DictConfig, metrics: Metrics) -> None:
    """Save survival prediction results using the unified logging interface."""
    save_path = cfg.logging.results.get("save_path")
    if save_path is None:
        return

    metrics_dict = {name: getattr(metrics, name) for name in SURVIVAL_METRICS}

    save_metrics(
        dataset=cfg.dataset,
        run_signature=build_run_signature(cfg),
        seed=cfg.seed,
        metrics=metrics_dict,
        results_dir=cfg.logging.results.dir,
        filename=save_path,
        metric_names=SURVIVAL_METRICS
    )


@hydra.main(version_base=None, config_path="../../../../configs", config_name="survival")
def main(cfg: DictConfig) -> None:
    cfg = initialize_hydra_run(
        cfg,
        "SPAN Survival Prediction - Hydra Configuration",
        seed=cfg.seed,
    )
    dataset_bundle = prepare_survival_dataset(cfg)
    best_metrics = run_training(cfg, dataset_bundle)
    save_results(cfg, best_metrics)
    print("\n" + "=" * 60)
    print("Best Test Metrics")
    print("=" * 60)
    print(f"C-Index: {best_metrics.c_index:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
