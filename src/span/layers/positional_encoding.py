import torch
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from timm.layers import trunc_normal_
import math

# Utilities for 2D Rotary Position Embedding (RoPE)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension in pairs: (x1, x2) -> (-x2, x1).

    x: (..., D) where D must be even
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

def _build_inv_freq(dim: int, base: float, device=None, dtype=None) -> torch.Tensor:
    """Return inverse frequencies of length dim//2 using base (theta).

    Standard formula: inv_freq[i] = base^(-2i/dim)
    """
    assert dim % 2 == 0, "Rotary dimension must be even"
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (idx / dim))
    return inv_freq.to(dtype if dtype is not None else torch.float32)

def build_2d_rope_cos_sin(
    ins_pos: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position 2D rotary cos/sin for a head_dim.

    - Split head_dim into two halves for H and W (each must be even).
    - ins_pos: (N, 2) integer grid coords (h, w).
    - Returns cos, sin with shape (N, 1, head_dim) for broadcasting to (N, H, D).
    """
    assert head_dim % 4 == 0, "For 2D RoPE, head_dim must be divisible by 4"
    device = ins_pos.device if device is None else device
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    D = head_dim
    Dh = D // 2  # per-axis dim
    # Each axis itself rotates in pairs, so needs even
    assert Dh % 2 == 0

    h_idx, w_idx = ins_pos.unbind(dim=1)

    inv_freq_h = _build_inv_freq(Dh, base, device=device, dtype=torch.float32)
    # outer produces (N, Dh//2)
    freqs_h = torch.outer(h_idx.to(torch.float32), inv_freq_h)
    # duplicate per 2-dim pair so that rotate_half aligns
    freqs_h = torch.repeat_interleave(freqs_h, 2, dim=1)  # (N, Dh)

    inv_freq_w = inv_freq_h  # share same spectrum for simplicity
    freqs_w = torch.outer(w_idx.to(torch.float32), inv_freq_w)
    freqs_w = torch.repeat_interleave(freqs_w, 2, dim=1)  # (N, Dh)

    # concat [H-block, W-block] -> (N, D)
    freqs = torch.cat([freqs_h, freqs_w], dim=1)
    cos = freqs.cos().to(dtype=dtype).unsqueeze(1)  # (N, 1, D)
    sin = freqs.sin().to(dtype=dtype).unsqueeze(1)  # (N, 1, D)
    return cos, sin

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply rotary with precomputed cos/sin.

    Shapes:
      - x:   (N, H, D)
      - cos: (N, 1, D) broadcastable to x
      - sin: (N, 1, D)
    """
    return (x * cos * scale) + (rotate_half(x) * sin * scale)

def apply_rope_2d_partial(
    x: torch.Tensor,
    ins_pos: torch.Tensor,
    rotary_dim: Optional[int] = None,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply 2D RoPE to a prefix of the head dimension for robustness.

    - x: (N, H, D)
    - rotary_dim: if None, use the largest multiple of 4 <= D
    """
    D = x.size(-1)
    if rotary_dim is None:
        rotary_dim = (D // 4) * 4
    if rotary_dim <= 0:
        return x
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = build_2d_rope_cos_sin(
        ins_pos, head_dim=rotary_dim, base=base, device=x.device, dtype=x.dtype
    )
    x_rot = apply_rotary_pos_emb(x_rot, cos, sin, scale)
    return torch.cat([x_rot, x_pass], dim=-1)

# Utilities for Relative Position Bias (RPB) and ALiBi

def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) +               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, size, learned_pos, pos_std):
        super(RelativePositionBias, self).__init__()
        self.size = size
        self.num_heads = num_heads
        self.adjustment_tables = nn.Parameter(torch.zeros(num_heads, 2 * size - 1, 2 * size - 1), requires_grad=learned_pos)
        if learned_pos:
            trunc_normal_(self.adjustment_tables, mean=0.0, std=pos_std)

    def forward(self, x_diff, y_diff):

        x_index = x_diff + (self.size - 1)
        y_index = y_diff + (self.size - 1)

        adjustment_tables = self.adjustment_tables
        adjustments = adjustment_tables[:, x_index, y_index].permute(1, 0).unsqueeze(-1)
        return adjustments

class ALiBiPositionBias(nn.Module):
    """ALiBi: Attention with Linear Biases for 2D positions.

    Computes bias as: -slope * sqrt(dx^2 + dy^2)
    Each head uses a different slope from a geometric sequence.
    """
    def __init__(self, num_heads):
        super(ALiBiPositionBias, self).__init__()
        self.num_heads = num_heads
        # Compute slopes using the existing _get_interleave function
        slopes = _get_interleave(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes).view(num_heads, 1, 1))

    def forward(self, x_diff, y_diff):
        """Compute ALiBi bias from position differences.

        Args:
            x_diff: (E,) x-coordinate differences
            y_diff: (E,) y-coordinate differences

        Returns:
            bias: (E, num_heads, 1) attention bias
        """
        # 2D Euclidean distance
        dist = torch.sqrt((x_diff.float() ** 2 + y_diff.float() ** 2))
        # dist: (E,) -> (E, 1, 1) for broadcasting
        dist = dist.unsqueeze(-1).unsqueeze(-1)
        # slopes: (num_heads, 1, 1), dist: (E, 1, 1) -> (E, num_heads, 1)
        bias = -self.slopes.transpose(0, 1) * dist
        return bias