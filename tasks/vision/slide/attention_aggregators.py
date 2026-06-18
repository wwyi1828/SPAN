import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from src.span.functional import create_activation


def _build_attention_activation(name):
    if name is None:
        return nn.Tanh()

    if isinstance(name, str):
        key = name.strip()
        if hasattr(nn, key):
            return getattr(nn, key)()
        alias = {
            "relu": "ReLU",
            "gelu": "GELU",
            "hardswish": "Hardswish",
            "silu": "SiLU",
            "swish": "SiLU",
            "tanh": "Tanh",
            "leakyrelu": "LeakyReLU",
        }
        mapped = alias.get(key.lower())
        if mapped is not None and hasattr(nn, mapped):
            return getattr(nn, mapped)()

    try:
        return create_activation(name)
    except Exception:
        return nn.Tanh()

class MLPAttention(nn.Module):

    def __init__(self, input_dim, activation, bias, dropout, head_div):

        super().__init__()
        self.input_dim = input_dim
        head_div = max(int(head_div), 1)
        self.hidden_dim = max(input_dim // head_div, 1)
        self.output_dim = 1

        self.feature = nn.Identity()

        attention_layers = [
            nn.Linear(self.input_dim, self.hidden_dim, bias=bias),
            _build_attention_activation(activation),
        ]
        if 0 < dropout < 1:
            attention_layers.append(nn.Dropout(dropout))

        attention_layers.append(nn.Linear(self.hidden_dim, self.output_dim, bias=bias))

        self.attention_mechanism = nn.Sequential(*attention_layers)
        self.last_attention_scores = None
        self.proj = nn.Identity()
        self.attn_drop = nn.Dropout(dropout) if 0 < dropout < 1 else nn.Identity()

    def forward(self, x):

        x = self.feature(x)
        attention_scores = self.attention_mechanism(x)
        attention_scores = torch.transpose(attention_scores, -1, -2)
        self.last_attention_scores = attention_scores.clone()
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_drop(attention_weights)
        attended_features = torch.matmul(attention_weights, x)
        attended_features = self.proj(attended_features)
        return attended_features

class GatedMLPAttention(nn.Module):

    def __init__(self, input_dim=512, activation='ReLU', bias=False, dropout=0.0):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim // 4
        self.output_dim = 1

        feature_transform_layers = [
            nn.Linear(self.input_dim, self.hidden_dim, bias=bias),
            create_activation(activation)
        ]

        gating_layers = [
            nn.Linear(self.input_dim, self.hidden_dim, bias=bias),
            nn.Sigmoid()
        ]

        if 0 < dropout < 1:
            feature_transform_layers.append(nn.Dropout(dropout))
            gating_layers.append(nn.Dropout(dropout))

        self.feature_transform = nn.Sequential(*feature_transform_layers)
        self.gating_signal = nn.Sequential(*gating_layers)
        self.attention_combine = nn.Linear(self.hidden_dim, self.output_dim, bias=bias)
        self.last_attention = None

    def forward(self, x):

        transformed_features = self.feature_transform(x)
        gating_signals = self.gating_signal(x)
        attention_weights = transformed_features * gating_signals
        attention_weights = self.attention_combine(attention_weights)
        attention_weights = torch.transpose(attention_weights, -1, -2)
        self.last_attention = attention_weights.clone()
        normalized_attention = F.softmax(attention_weights, dim=-1)
        x = torch.matmul(normalized_attention, x)
        return x

class MultiAttnAggregator(nn.Module):

    def __init__(self, input_dim=512, activation='ReLU', use_gated=False, bias=True,
                 dropout=0.0, droppath=0.0, num_layers=1, num_classes=1, head_div=4):

        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.aggregators = nn.ModuleList()
        for _ in range(num_layers):
            if use_gated:
                self.aggregators.append(GatedMLPAttention(input_dim, activation, bias, dropout))
            else:
                self.aggregators.append(MLPAttention(input_dim, activation, bias, dropout, head_div))

        self.fc = nn.Linear(input_dim, num_classes)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.droppath = DropPath(droppath)
        self.scale_logits = nn.Parameter(torch.zeros(num_layers))
        self.scale_router = nn.Linear(input_dim, 1, bias=False)
        nn.init.zeros_(self.scale_router.weight)
        # Keep initial behavior close to previous version by weakly using context at start.
        self.context_alpha = nn.Parameter(torch.full((num_layers,), -2.0))
        self.context_router = nn.Linear(input_dim * 2, 1)
        nn.init.zeros_(self.context_router.weight)
        nn.init.zeros_(self.context_router.bias)

    def forward(self, x_list, context):

        assert len(x_list) == self.num_layers
        assert len(context) == self.num_layers

        scale_features = []
        routed_logits = []
        for i in range(self.num_layers):
            x = x_list[i]
            if x.size(0) > 0:
                feature_i = self.aggregators[i](x)
            else:
                # Rare fallback: no instance tokens at one scale.
                feature_i = context[i][0, None]

            if context[i].size(0) > 0:
                context_i = context[i][0, None]
                context_gate = self.context_router(torch.cat([feature_i, context_i], dim=-1)).view(1, 1)
                alpha = torch.sigmoid(self.context_alpha[i] + context_gate)
                feature_i = (1.0 - alpha) * feature_i + alpha * context_i
            scale_features.append(feature_i)
            routed_logits.append(self.scale_router(feature_i).view(1))

        stacked = torch.stack(scale_features, dim=0)
        dynamic_logits = torch.cat(routed_logits, dim=0)
        scale_weights = F.softmax(self.scale_logits + dynamic_logits, dim=0).view(self.num_layers, 1, 1)
        weighted = stacked * scale_weights

        current = weighted[0]
        for i in range(1, self.num_layers):
            current = current + self.droppath(weighted[i])

        output = self.fc(self.dropout(self.norm(current)))
        return output

class MultiAggregator(nn.Module):

    def __init__(self, input_dim=512, dropout=0.0, num_layers=1,
                 num_classes=1, lastnorm=False, weight=False, use_norm=False):

        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lastnorm = lastnorm

        self.norm = nn.LayerNorm(input_dim) if use_norm else nn.Identity()

        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(input_dim, num_classes))
        if weight:
            self.weight = nn.Linear(num_layers*input_dim, 3)
        else:
            self.weight = None

    def forward(self, context):

        context_tensor = torch.stack(context, dim=0)[:, 0, :]
        if not self.lastnorm:
            context_tensor = self.norm(context_tensor)
        if self.weight is not None:
            weight = self.weight(context_tensor.view(1, -1)).view(-1, 1)
            weights = F.softmax(weight/10, dim=0)
            context_tensor = (context_tensor * weights).sum(dim=0, keepdim=True)
        else:
            context_tensor = context_tensor.sum(dim=0, keepdim=True)
        if self.lastnorm:
            context_tensor = self.norm(context_tensor)
        output = self.fc(context_tensor)
        return output
