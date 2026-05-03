import torch


def dice_loss_fn(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    probs = torch.sigmoid(preds)
    intersection = (probs * targets).sum()
    dice_score = (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1 - dice_score


def calculate_pos_weight(data_loader, threshold: float) -> float:
    zeros = 0.0
    ones = 0.0
    for item in data_loader:
        labels = (item.y >= threshold).float()
        zeros += (labels == 0).sum().item()
        ones += (labels == 1).sum().item()
    ones = max(ones, 1.0)
    return zeros / ones

