"""Utilities for working with precomputed feature variants."""
from __future__ import annotations

from typing import Any

# Known embedding dimensions for supported feature variants.
_FEATURE_DIM_MAP = {
    "PLIP": 768,
    "V2": 1280,
    "CONCH": 512,
}

_DEFAULT_DIM = 1024


def resolve_feature_dim(features_variant: Any, default: int = _DEFAULT_DIM) -> int:
    """Return the embedding dimension associated with a feature variant name."""
    if features_variant is None:
        return default
    variant_key = str(features_variant).upper()
    return _FEATURE_DIM_MAP.get(variant_key, default)
