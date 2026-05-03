import random
import copy
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.utils.coord_aug import update_coord_aug_multiplier
from .model import build_model, SPANClassifier
from lib.utils.wandb_helper import create_logger


@dataclass
class Metrics:
    loss: float
    accuracy: float
    aucs: Sequence[float]
    avg_auc: float
    macro_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None


def build_scheduler(optimizer, epochs: int):
    return CosineAnnealingLR(optimizer, T_max=epochs)


def should_update_best(current_auc: float, best_auc: float) -> bool:
    """Determine if current macro AUC improves over the best."""
    if math.isnan(current_auc):
        return False
    if math.isnan(best_auc):
        return True
    return current_auc > best_auc


def aggregate_outputs_to_metrics(
    logits: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:

    if not logits:
        return np.empty((0, num_classes)), np.empty((0, num_classes))

    all_scores = torch.cat([F.softmax(logit, dim=1) for logit in logits], dim=0)
    all_labels = torch.cat(labels, dim=0)
    return all_scores.cpu().numpy(), all_labels.cpu().numpy()


def compute_auc_scores(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> Tuple[Sequence[float], float]:

    aucs = []
    if y_prob.size == 0 or y_true.size == 0 or num_classes == 0:
        return [float("nan")] * num_classes, float("nan")

    for idx in range(num_classes):
        try:
            auc_value = roc_auc_score(y_true[:, idx], y_prob[:, idx])
        except ValueError:
            auc_value = float("nan")
        aucs.append(auc_value)

    avg_auc = float(np.nanmean(aucs)) if aucs else 0.0
    return aucs, avg_auc


def train_one_epoch(
    model: SPANClassifier,
    data_loader: Sequence,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Metrics:

    model.train()
    total_loss = 0.0
    total_correct = 0
    all_logits, all_labels = [], []

    if isinstance(data_loader, list):
        random.shuffle(data_loader)

    for slide in data_loader:
        labels = slide.graph_y.to(device)
        feats = slide.x.to(device)
        coords = slide.pos.to(device).int()

        if labels.dim() == 1 or labels.size(-1) == 1:
            targets = labels.view(-1).long()
            labels_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            labels_one_hot = labels.float()
            targets = labels_one_hot.argmax(dim=1)

        logits = model(coords, feats)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_correct += (logits.argmax(1) == targets).sum().item()
        all_logits.append(logits.detach())
        all_labels.append(labels_one_hot.detach())

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(len(data_loader), 1)

    scores, label_array = aggregate_outputs_to_metrics(all_logits, all_labels, num_classes)
    aucs, avg_auc = compute_auc_scores(label_array, scores, num_classes)

    return Metrics(loss=avg_loss, accuracy=accuracy, aucs=aucs, avg_auc=avg_auc)


def evaluate(
    model: SPANClassifier,
    data_loader: Sequence,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    detailed: bool = False,
) -> Metrics:

    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for slide in data_loader:
            labels = slide.graph_y.to(device)
            feats = slide.x.to(device)
            coords = slide.pos.to(device).int()

            if labels.dim() == 1 or labels.size(-1) == 1:
                targets = labels.view(-1).long()
                labels_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            else:
                labels_one_hot = labels.float()
                targets = labels_one_hot.argmax(dim=1)

            logits = model(coords, feats)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            total_correct += (logits.argmax(1) == targets).sum().item()
            all_logits.append(logits)
            all_labels.append(labels_one_hot)

    if not all_logits:
        empty_auc = [float("nan")] * num_classes
        return Metrics(loss=0.0, accuracy=0.0, aucs=empty_auc, avg_auc=float("nan"))

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(len(data_loader), 1)

    scores, label_array = aggregate_outputs_to_metrics(all_logits, all_labels, num_classes)
    aucs, avg_auc = compute_auc_scores(label_array, scores, num_classes)

    if not detailed:
        return Metrics(loss=avg_loss, accuracy=accuracy, aucs=aucs, avg_auc=avg_auc)

    predictions = scores.argmax(axis=1)
    targets = label_array.argmax(axis=1)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(targets, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(targets, predictions, average="macro", zero_division=0)

    return Metrics(
        loss=avg_loss,
        accuracy=accuracy,
        aucs=aucs,
        avg_auc=avg_auc,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )


def run_training(cfg: DictConfig, dataset) -> Metrics:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, dataset.num_classes).to(device)
    logger = create_logger(cfg)

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.epochs)
    criterion = nn.CrossEntropyLoss(weight=dataset.class_weights.to(device))

    patience = cfg.training.get("early_stop_patience", 0)
    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = float('nan')
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        update_coord_aug_multiplier(cfg, dataset.train, epoch + 1, cfg.epochs)
        train_metrics = train_one_epoch(
            model, dataset.train, optimizer, criterion, device, dataset.num_classes
        )
        val_metrics = evaluate(model, dataset.val, criterion, device, dataset.num_classes, detailed=True)

        if should_update_best(val_metrics.avg_auc, best_val_auc):
            best_val_auc = val_metrics.avg_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(
                f"    -> New best model found! Val AUC: {best_val_auc:.4f}. Checkpoint updated."
            )
        else:
            epochs_no_improve += 1

        test_metrics = evaluate(model, dataset.test, criterion, device, dataset.num_classes, detailed=True)

        scheduler.step()

        if logger:
            log_dict = {
                "train/loss": float(train_metrics.loss),
                "train/accuracy": float(train_metrics.accuracy),
                "val/loss": float(val_metrics.loss),
                "val/accuracy": float(val_metrics.accuracy),
                "val/avg_auc": float(val_metrics.avg_auc),
                "test/accuracy": float(test_metrics.accuracy),
                "test/avg_auc": float(test_metrics.avg_auc),
            }
            if val_metrics.macro_f1 is not None:
                log_dict["val/macro_f1"] = float(val_metrics.macro_f1)
                log_dict["val/macro_precision"] = float(val_metrics.macro_precision)
                log_dict["val/macro_recall"] = float(val_metrics.macro_recall)
            if test_metrics.macro_f1 is not None:
                log_dict["test/macro_f1"] = float(test_metrics.macro_f1)
            logger.log_epoch(epoch + 1, log_dict)

        val_f1_str = f", Val F1: {val_metrics.macro_f1:.4f}" if val_metrics.macro_f1 is not None else ""
        test_f1_str = f", Test F1: {test_metrics.macro_f1:.4f}" if test_metrics.macro_f1 is not None else ""
        print(
            f"[Epoch {epoch + 1:03d}/{cfg.epochs}] "
            f"Train Loss: {train_metrics.loss:.4f}, Train Acc: {train_metrics.accuracy:.4f} | "
            f"Val Acc: {val_metrics.accuracy:.4f}, Val AUC: {val_metrics.avg_auc:.4f}{val_f1_str} | "
            f"Test Acc: {test_metrics.accuracy:.4f}, Test AUC: {test_metrics.avg_auc:.4f}{test_f1_str}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate(
        model, dataset.test, criterion, device, dataset.num_classes, detailed=True
    )

    if logger:
        payload = {
            "val/avg_auc": float(best_val_auc),
            "test/accuracy": float(final_metrics.accuracy),
            "test/avg_auc": float(final_metrics.avg_auc),
            "test/macro_f1": float(final_metrics.macro_f1 or 0.0),
            "test/macro_precision": float(final_metrics.macro_precision or 0.0),
            "test/macro_recall": float(final_metrics.macro_recall or 0.0),
        }
        for idx, auc in enumerate(final_metrics.aucs):
            payload[f"test/auc_class_{idx}"] = float(auc)
        logger.log_best(payload)
        logger.finish()

    return final_metrics
