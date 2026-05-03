import torch.nn as nn
from omegaconf import DictConfig

from ..encoder import build_slide_encoder_bundle

from .aggregators import SurvivalAggregator, SurvivalMultiAggregator


class ModuleCompositor(nn.Module):
    def __init__(self, encoder, aggregator, multi_aggregator, classifier_1):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.multi_aggregator = multi_aggregator
        self.classifier_1 = classifier_1


def build_model(cfg: DictConfig):
    bundle = build_slide_encoder_bundle(
        cfg,
        num_outputs=cfg.n_groups,
        default_features_variant="UNI",
    )

    embed_dim = bundle.model_components["embed_dim"]
    num_layers = len(bundle.encoder_config)
    num_classes = cfg.n_groups
    cls_drop = cfg.survival.get('cls_drop', 0.05)

    classifier_1 = nn.Sequential(
        nn.Dropout(cls_drop),
        nn.Linear(num_layers * embed_dim, num_classes)
    )

    aggregator = SurvivalAggregator(
        input_dim=embed_dim,
        dropout=cls_drop,
        num_classes=num_classes,
    )

    multi_aggregator = SurvivalMultiAggregator(
        input_dim=embed_dim,
        dropout=cls_drop,
        num_layers=num_layers,
        num_classes=num_classes,
        lastnorm=False,
        weight=False
    )

    model = ModuleCompositor(
        encoder=bundle.encoder,
        aggregator=aggregator,
        multi_aggregator=multi_aggregator,
        classifier_1=classifier_1
    )

    return model, bundle.padder
