import torch
import torch.nn as nn


class SurvivalAggregator(nn.Module):
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
