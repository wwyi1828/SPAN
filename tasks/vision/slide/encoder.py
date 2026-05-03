from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from omegaconf import DictConfig

from src.span.builders import build_model_config
from src.span.model import SPAN_Encoder, create_block
from src.span.preprocessing import SPAN_Padder
from tasks.vision.shared.features import resolve_feature_dim


@dataclass(frozen=True)
class SlideEncoderBundle:
    encoder: SPAN_Encoder
    padder: SPAN_Padder
    encoder_config: Sequence[Any]
    model_components: dict
    in_channels: int


def build_slide_encoder_bundle(
    cfg: DictConfig,
    num_outputs: int,
    default_features_variant: str = "R50",
    require_channel_factor: Optional[int] = None,
) -> SlideEncoderBundle:
    fv = getattr(cfg, "features_variant", default_features_variant)
    in_channels = resolve_feature_dim(fv)

    encoder_config, _, model_components = build_model_config(
        cfg, num_outputs=num_outputs, in_channels=in_channels, enc_act=cfg.model.enc_act
    )

    if (
        require_channel_factor is not None
        and int(model_components["channel_factor"]) != int(require_channel_factor)
    ):
        raise ValueError(
            f"Expected channel_factor == {int(require_channel_factor)}, "
            f"got {int(model_components['channel_factor'])}."
        )

    encoder_blocks = [
        create_block(config, idx + 1, mask_padding=True)
        for idx, config in enumerate(encoder_config)
    ]
    encoder = SPAN_Encoder(
        encoder_blocks,
        embed_dim=in_channels,
        token_init_types=model_components["token_init_types"],
    )

    encoder_depth = len(model_components["encoder_config_string"])
    padder = SPAN_Padder(
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        dilation=cfg.model.dilation,
        n_layers=max(encoder_depth - 1, 0),
        pad_feats=None,
    )

    return SlideEncoderBundle(
        encoder=encoder,
        padder=padder,
        encoder_config=encoder_config,
        model_components=model_components,
        in_channels=in_channels,
    )
