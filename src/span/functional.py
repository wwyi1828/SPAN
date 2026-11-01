import torch.nn as nn

_ACTIVATION_ALIASES = {
    "relu": "ReLU",
    "gelu": "GELU",
    "hardswish": "Hardswish",
    "silu": "SiLU",
    "swish": "SiLU",
    "tanh": "Tanh",
    "leakyrelu": "LeakyReLU",
}


def create_activation(name: str, **kwargs):
    if not isinstance(name, str):
        raise ValueError(f"Activation name must be a string, got {name!r}")
    key = name.strip()
    activation_class = getattr(nn, _ACTIVATION_ALIASES.get(key.lower(), key), None)
    if not isinstance(activation_class, type) or not issubclass(activation_class, nn.Module):
        raise ValueError(f"Unknown activation: {name!r}")
    return activation_class(**kwargs)
