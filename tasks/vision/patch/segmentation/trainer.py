import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig

from lib.utils.coord_aug import update_coord_aug_multiplier
from .model import build_model
from .utils import dice_loss_fn, calculate_pos_weight
from lib.utils.wandb_helper import create_logger


@dataclass
class Metrics:
    ce: float
    dice: float
    f1: float
    iou: float
    recall: Optional[float] = None
    precision: Optional[float] = None


def train_one_epoch(model, data_loader, optimizer, scheduler, padder, device, ce_loss_fn, threshold, dice_weight):
    model.train()
    total_ce = 0.0
    total_dice = 0.0
    preds_all = []
    labels_all = []
    for item in data_loader:
        ins_pos = item.pos.to(device)
        feats = item.x.to(device)
        labels = (item.y >= threshold).float().unsqueeze(1).to(device)
        ins_pos, feats = padder(ins_pos, feats)
        ins_pos, feats_multi, global_feats, spatial_shape, pos_dict = model.encoder(ins_pos, feats)
        _, decoded_feats = model.decoder(ins_pos[-1], feats_multi, global_feats, spatial_shape, pos_dict)
        logits = decoded_feats[-1][1:-1]
        ce_loss = ce_loss_fn(logits, labels)
        dice_loss = dice_loss_fn(logits, labels)
        loss = (1 - dice_weight) * ce_loss + dice_weight * dice_loss if labels.sum() > 0 else ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_ce += ce_loss.item()
        total_dice += dice_loss.item()
        preds = (torch.sigmoid(logits) >= threshold).float()
        preds_all.extend(preds.detach().cpu().numpy().flatten().tolist())
        labels_all.extend(labels.detach().cpu().numpy().flatten().tolist())
    scheduler.step()
    avg_ce = total_ce / max(len(data_loader), 1)
    avg_dice = total_dice / max(len(data_loader), 1)
    return Metrics(ce=avg_ce, dice=avg_dice, f1=f1_score(labels_all, preds_all, zero_division=0), iou=jaccard_score(labels_all, preds_all, zero_division=0))


def evaluate(model, data_loader, padder, device, ce_loss_fn, threshold, detailed: bool):
    model.eval()
    total_ce = 0.0
    total_dice = 0.0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for item in data_loader:
            ins_pos = item.pos.to(device)
            feats = item.x.to(device)
            labels = (item.y >= threshold).float().unsqueeze(1).to(device)
            ins_pos, feats = padder(ins_pos, feats)
            ins_pos, feats_multi, global_feats, spatial_shape, pos_dict = model.encoder(ins_pos, feats)
            _, decoded_feats = model.decoder(ins_pos[-1], feats_multi, global_feats, spatial_shape, pos_dict)
            logits = decoded_feats[-1][1:-1]
            ce_loss = ce_loss_fn(logits, labels)
            dice_loss = dice_loss_fn(logits, labels)
            total_ce += ce_loss.item()
            total_dice += dice_loss.item()
            preds = (torch.sigmoid(logits) >= threshold).float()
            preds_all.extend(preds.cpu().numpy().flatten().tolist())
            labels_all.extend(labels.cpu().numpy().flatten().tolist())
    avg_ce = total_ce / max(len(data_loader), 1)
    avg_dice = total_dice / max(len(data_loader), 1)
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    iou = jaccard_score(labels_all, preds_all, zero_division=0)
    if detailed:
        recall = recall_score(labels_all, preds_all, zero_division=0)
        precision = precision_score(labels_all, preds_all, zero_division=0)
        return Metrics(ce=avg_ce, dice=avg_dice, f1=f1, iou=iou, recall=recall, precision=precision)
    return Metrics(ce=avg_ce, dice=avg_dice, f1=f1, iou=iou)


def run_training(cfg: DictConfig, dataset) -> Metrics:
    model, padder = build_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger = create_logger(cfg)
    threshold = cfg.segmentation.threshold
    dice_weight = cfg.segmentation.dice_weight
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    pos_weight = calculate_pos_weight(dataset.train, threshold)
    ce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('-inf')
    best_val_metrics = None
    for epoch in range(cfg.epochs):
        update_coord_aug_multiplier(cfg, dataset.train, epoch + 1, cfg.epochs)
        train_metrics = train_one_epoch(model, dataset.train, optimizer, scheduler, padder, device, ce_loss_fn, threshold, dice_weight)
        val_metrics = evaluate(model, dataset.val, padder, device, ce_loss_fn, threshold, detailed=False)
        if (val_metrics.f1 + val_metrics.iou) / 2 >= best_val:
            best_val = (val_metrics.f1 + val_metrics.iou) / 2
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
        test_metrics = evaluate(model, dataset.test, padder, device, ce_loss_fn, threshold, detailed=True)
        if logger:
            logger.log_epoch(
                epoch + 1,
                {
                    "train/ce": float(train_metrics.ce),
                    "train/dice": float(train_metrics.dice),
                    "train/f1": float(train_metrics.f1),
                    "train/iou": float(train_metrics.iou),
                    "val/ce": float(val_metrics.ce),
                    "val/dice": float(val_metrics.dice),
                    "val/f1": float(val_metrics.f1),
                    "val/iou": float(val_metrics.iou),
                    "test/f1": float(test_metrics.f1),
                    "test/iou": float(test_metrics.iou),
                },
            )
        print(
            f"[Epoch {epoch + 1:03d}/{cfg.epochs}] "
            f"Train CE: {train_metrics.ce:.4f}, Dice: {train_metrics.dice:.4f} | "
            f"Val CE: {val_metrics.ce:.4f}, Dice: {val_metrics.dice:.4f}, F1: {val_metrics.f1:.4f}, IoU: {val_metrics.iou:.4f} | "
            f"Test F1: {test_metrics.f1:.4f}, IoU: {test_metrics.iou:.4f}"
        )
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, dataset.test, padder, device, ce_loss_fn, threshold, detailed=True)
    if logger:
        payload = {
            "val/f1": float(best_val_metrics.f1 if best_val_metrics else 0.0),
            "val/iou": float(best_val_metrics.iou if best_val_metrics else 0.0),
            "test/f1": float(final_metrics.f1),
            "test/iou": float(final_metrics.iou),
            "test/recall": float(final_metrics.recall or 0.0),
            "test/precision": float(final_metrics.precision or 0.0),
            "test/ce": float(final_metrics.ce),
            "test/dice": float(final_metrics.dice),
        }
        logger.log_best(payload)
        logger.finish()
    return final_metrics
