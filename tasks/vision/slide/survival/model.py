import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..encoder import build_slide_encoder_bundle
from ..attention_aggregators import MultiAggregator
from .aggregators import SurvivalAggregator


class SPANSurvival(nn.Module):
    def __init__(self, encoder, padder, head, aggregation):
        super().__init__()
        self.encoder = encoder
        self.padder = padder
        self.head = head
        self.aggregation = aggregation

    def forward(self, coords, features):
        coords, features, mask = self.padder(coords.int(), features)
        _, scales, globals_, _, state = self.encoder(coords, features, sentinel_mask=mask)
        if self.aggregation == "concat":
            inputs = torch.cat([features[0, None] for features in globals_], dim=1)
        elif self.aggregation == "addition":
            inputs = scales[-1][~state.sentinel_mask]
        else:
            inputs = globals_
        hazards = torch.sigmoid(self.head(inputs))
        survival = torch.cumprod(1 - hazards, dim=1)
        return hazards, survival


def build_model(cfg: DictConfig):
    aggregation = str(cfg.survival.aggregation).strip().lower()
    if aggregation not in ("concat", "addition", "multi_addition"):
        raise ValueError(f"Unknown survival aggregation: {aggregation!r}")
    bundle = build_slide_encoder_bundle(cfg, num_outputs=cfg.n_groups,
                                       default_features_variant="UNI", require_channel_factor=1)
    embed_dim = bundle.model_components["embed_dim"]
    num_layers = len(bundle.encoder_config)
    dropout = cfg.survival.get("cls_drop", 0.05)
    if aggregation == "concat":
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers * embed_dim, cfg.n_groups))
    elif aggregation == "addition":
        head = SurvivalAggregator(embed_dim, dropout, cfg.n_groups)
    else:
        head = MultiAggregator(input_dim=embed_dim, dropout=dropout,
                                       num_layers=num_layers, num_classes=cfg.n_groups,
                                       lastnorm=False, weight=False)
    model = SPANSurvival(bundle.encoder, bundle.padder, head, aggregation)
    return model, bundle.padder
