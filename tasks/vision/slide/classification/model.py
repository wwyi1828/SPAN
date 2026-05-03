from typing import Callable, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..encoder import build_slide_encoder_bundle

from ..attention_aggregators import (
    MultiAggregator,
    MultiAttnAggregator,
)


class _SimpleAggregator(nn.Module):
    def __init__(self, module: nn.Module, selector: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], object]) -> None:
        super().__init__()
        self.module = module
        self.selector = selector

    def forward(self, feats_list, global_feats):
        return self.module(self.selector(feats_list, global_feats))


def _pool_last_scale_addition(
    feats_list: Sequence[torch.Tensor], _: Sequence[torch.Tensor]
) -> torch.Tensor:
    last_scale = feats_list[-1]
    max_feat = torch.max(last_scale, dim=0, keepdim=True)[0]
    mean_feat = torch.mean(last_scale, dim=0, keepdim=True)
    return max_feat + mean_feat


def _create_aggregator(
    aggr_method: str,
    embed_dim: int,
    num_layers: int,
    num_classes: int,
    cls_drop: float,
    head_act: str,
    head_div: int,
    head_norm: bool,
) -> nn.Module:
    method = str(aggr_method).strip().lower()

    if method == "concat":
        return _SimpleAggregator(
            nn.Sequential(
                nn.LayerNorm(num_layers * embed_dim),
                nn.Dropout(cls_drop),
                nn.Linear(num_layers * embed_dim, num_classes),
            ),
            lambda _, gf: torch.cat([feat[0, None] for feat in gf], dim=1),
        )

    if method == "mean":
        return _SimpleAggregator(
            nn.Sequential(nn.Dropout(cls_drop), nn.Linear(embed_dim, num_classes)),
            lambda _, gf: torch.cat([feat[0, None] for feat in gf], dim=0).mean(dim=0, keepdim=True),
        )

    if method == "last":
        return _SimpleAggregator(
            nn.Sequential(nn.Dropout(cls_drop), nn.Linear(embed_dim, num_classes)),
            lambda _, gf: gf[-1][0, None],
        )

    if method == "addition":
        return _SimpleAggregator(
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(cls_drop),
                nn.Linear(embed_dim, num_classes),
            ),
            _pool_last_scale_addition,
        )

    if method == "multi_addition":
        return _SimpleAggregator(
            MultiAggregator(embed_dim, cls_drop, num_layers, num_classes, True, False, head_norm),
            lambda _, gf: gf,
        )

    if method == "multiattn":
        return MultiAttnAggregator(
            embed_dim, head_act, False, True, cls_drop, 0.0, num_layers, num_classes, head_div
        )

    raise ValueError(f"Unknown aggregation method: {method}")

class SPANClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_layers: int,
        num_classes: int,
        aggr_method: str,
        cls_drop: float,
        head_act: str,
        head_div: int,
        head_norm: bool,
        padder: nn.Module,
    ) -> None:

        super().__init__()
        self.encoder = encoder
        self.padder = padder
        self.aggregator = _create_aggregator(
            aggr_method=aggr_method,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            cls_drop=cls_drop,
            head_act=head_act,
            head_div=head_div,
            head_norm=head_norm,
        )

    def forward(self, ins_pos: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:

        ins_pos, feats = self.padder(ins_pos, feats)

        coords, feats_list, global_feats, _, _ = self.encoder(ins_pos, feats)
        return self.aggregate(feats_list, global_feats)

    def aggregate(self, feats_list, global_feats) -> torch.Tensor:
        return self.aggregator(feats_list, global_feats)


def build_model(cfg: DictConfig, num_classes: int) -> SPANClassifier:
    bundle = build_slide_encoder_bundle(
        cfg,
        num_outputs=num_classes,
        default_features_variant="R50",
        require_channel_factor=1,
    )

    model = SPANClassifier(
        encoder=bundle.encoder,
        embed_dim=bundle.model_components["embed_dim"],
        num_layers=len(bundle.encoder_config),
        num_classes=num_classes,
        aggr_method=cfg.classification.aggr_method,
        cls_drop=cfg.classification.cls_drop,
        head_act=cfg.classification.head_act,
        head_div=cfg.classification.head_div,
        head_norm=cfg.classification.get("head_norm", False),
        padder=bundle.padder,
    )
    return model
