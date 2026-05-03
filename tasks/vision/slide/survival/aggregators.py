import torch
import torch.nn as nn
from ..attention_aggregators import (
    MultiAggregator,
    DropMultiAggregator,
    MultiAttnAggregator
)


class SurvivalMixin:

    def predict_survival(self, x):
        output = self.forward(x)
        Y_hat = torch.topk(output, 1, dim=1)[1]
        hazards = torch.sigmoid(output)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def predict_risk_score(self, x):
        _, S, _ = self.predict_survival(x)
        return -torch.sum(S, dim=1)


class SurvivalAggregator(SurvivalMixin, nn.Module):
    def __init__(self, input_dim=512, dropout=0.0, num_classes=1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        max_feat = torch.max(x, dim=0, keepdim=True)[0]
        mean_feat = torch.mean(x, dim=0, keepdim=True)
        pooled = max_feat + mean_feat
        return self.fc(self.dropout(self.norm(pooled)))


class SurvivalMultiAggregator(SurvivalMixin, MultiAggregator):
    pass


class SurvivalDropMultiAggregator(SurvivalMixin, DropMultiAggregator):
    pass


class SurvivalMultiAttnAggregator(SurvivalMixin, MultiAttnAggregator):
    pass
