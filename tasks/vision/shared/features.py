from numbers import Integral
from typing import Optional

_FEATURE_DIM_MAP = {
    "R50": 1024,
    "UNI": 1024,
    "PLIP": 768,
    "V2": 1280,
    "CONCH": 512,
}


def resolve_feature_dim(features_variant: str, features_dim: Optional[int] = None) -> int:
    if features_dim is not None:
        if isinstance(features_dim, bool) or not isinstance(features_dim, Integral) or features_dim <= 0:
            raise ValueError(f"features_dim must be a positive integer, got {features_dim!r}")
        return int(features_dim)
    if not isinstance(features_variant, str) or features_variant.strip().upper() not in _FEATURE_DIM_MAP:
        raise ValueError(f"Unknown feature variant: {features_variant!r}; set features_dim explicitly for custom features")
    return _FEATURE_DIM_MAP[features_variant.strip().upper()]
