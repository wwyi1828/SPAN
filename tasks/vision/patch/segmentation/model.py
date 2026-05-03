from omegaconf import DictConfig
import torch.nn as nn

from src.span.builders import build_model_config
from src.span.model import create_block, SPAN_Encoder, SPAN_Decoder
from src.span.preprocessing import SPAN_Padder


class SegmentationModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def build_model(cfg: DictConfig):
    in_channels = cfg.features_dim
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
    encoder_blocks = [create_block(config, i + 1, mask_padding=True) for i, config in enumerate(encoder_config)]
    decoder_blocks = [create_block(config, len(decoder_config) - i) for i, config in enumerate(decoder_config)]
    encoder = SPAN_Encoder(
        encoder_blocks,
        embed_dim=in_channels,
        token_init_types=model_components['token_init_types'],
    )
    decoder = SPAN_Decoder(decoder_blocks)
    model = SegmentationModel(encoder=encoder, decoder=decoder)
    return model, padder
