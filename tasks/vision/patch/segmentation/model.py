from omegaconf import DictConfig
import torch.nn as nn

from src.span.builders import build_model_config
from src.span.model import create_block, SPAN_Encoder, SPAN_Decoder
from src.span.preprocessing import SPAN_Padder
from tasks.vision.shared.features import resolve_feature_dim


class SegmentationModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, padder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.padder = padder

    def forward(self, coords, features):
        coords, features, mask = self.padder(coords.int(), features)
        coords, scales, globals_, shape, state = self.encoder(coords, features, sentinel_mask=mask)
        _, decoded = self.decoder(coords[-1], scales, globals_, shape, state)
        return decoded[-1][~state.sentinel_mask]


def build_model(cfg: DictConfig):
    in_channels = resolve_feature_dim(cfg.get("features_variant", "R50"), cfg.get("features_dim"))
    encoder_config, decoder_config, model_components = build_model_config(
        cfg, num_outputs=1, in_channels=in_channels, enc_act=cfg.model.enc_act
    )

    padder = SPAN_Padder(
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        dilation=model_components['dilation'],
        n_layers=model_components['ds_layers'],
        pad_feats=None,
    )
    encoder_blocks = [create_block(config) for config in encoder_config]
    decoder_blocks = [create_block(config) for config in decoder_config]
    encoder = SPAN_Encoder(
        encoder_blocks,
        embed_dim=in_channels,
        token_init_types=model_components['token_init_types'],
    )
    decoder = SPAN_Decoder(decoder_blocks)
    model = SegmentationModel(encoder=encoder, decoder=decoder, padder=padder)
    return model, padder
