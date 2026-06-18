import copy
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig
from sksurv.metrics import concordance_index_censored

from lib.utils.coord_aug import update_coord_aug_multiplier
from .model import build_model
from .utils import Metrics
from lib.utils.wandb_helper import create_logger


def nll_loss(hazards, S, Y, c, alpha, eps=1e-7):
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


class NLLSurvLoss:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


def train_one_epoch(model, data_loader, optimizer, scheduler, padder, device, criterion, aggr_method):
    total_loss = 0
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    model.train()
    for item in data_loader:
        ins_feat = item.x.to(device)
        ins_pos = item.pos.to(device)
        status = item.censorship
        label = item.survival_label
        stime = item.survival_time
        with torch.no_grad():
            ins_pos = ins_pos.int()
        ins_pos, ins_feat = padder(ins_pos, ins_feat)
        coord, feats, global_feats, spatial_shape, pos_dict = model.encoder(ins_pos, ins_feat)
        if aggr_method == 'concat':
            global_feats = [_[0, None] for _ in global_feats]
            output = torch.cat(global_feats).view(1, -1)
            output = model.classifier_1(output)
            Y_hat = torch.topk(output, 1, dim = 1)[1]
            hazards = torch.sigmoid(output)
            S = torch.cumprod(1 - hazards, dim=1)
        elif aggr_method == 'addition':
            output = torch.cat([feats[-1]], dim=0)
            hazards, S, Y_hat = model.aggregator.predict_survival(output)
        else:
            hazards, S, Y_hat = model.multi_aggregator.predict_survival(global_feats)
        loss = criterion(hazards=hazards, S=S, Y=label.unsqueeze(1).cuda(), c=status.unsqueeze(1).cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        risk = -torch.sum(S, dim=1).detach().cpu().item()
        all_risk_scores.append(risk)
        all_event_times.append(stime)
        all_censorships.append(status.item())
    scheduler.step()
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    return total_loss/len(data_loader), c_index


def evaluate(model, data_loader, padder, device, criterion, aggr_method):
    total_loss = 0
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    model.eval()
    for item in data_loader:
        ins_feat = item.x.to(device)
        ins_pos = item.pos.to(device)
        status = item.censorship
        label = item.survival_label
        stime = item.survival_time
        with torch.no_grad():
            ins_pos = ins_pos.int()
            ins_pos, ins_feat = padder(ins_pos, ins_feat)
            coord, feats, global_feats, spatial_shape, pos_dict = model.encoder(ins_pos, ins_feat)
            if aggr_method == 'concat':
                global_feats = [_[0, None] for _ in global_feats]
                output = torch.cat(global_feats).view(1, -1)
                output = model.classifier_1(output)
                Y_hat = torch.topk(output, 1, dim = 1)[1]
                hazards = torch.sigmoid(output)
                S = torch.cumprod(1 - hazards, dim=1)
            elif aggr_method == 'addition':
                output = torch.cat([feats[-1]], dim=0)
                hazards, S, Y_hat = model.aggregator.predict_survival(output)
            else:
                hazards, S, Y_hat = model.multi_aggregator.predict_survival(global_feats)
            loss = criterion(hazards=hazards, S=S, Y=label.unsqueeze(1).cuda(), c=status.unsqueeze(1).cuda())
            total_loss += loss.item()
            risk = -torch.sum(S, dim=1).detach().cpu().item()
            all_risk_scores.append(risk)
            all_event_times.append(stime)
            all_censorships.append(status.item())
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    return total_loss/len(data_loader), c_index


def run_training(cfg: DictConfig, dataset) -> Metrics:
    model, padder = build_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger = create_logger(cfg)
    criterion = NLLSurvLoss(alpha=cfg.survival.alpha)
    aggr_method = cfg.survival.aggregation
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    patience = cfg.training.get("early_stop_patience", 0)
    best_model_state = copy.deepcopy(model.state_dict())
    best_cindex = float("-inf")
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        update_coord_aug_multiplier(cfg, dataset.train, epoch + 1, cfg.epochs)
        train_loss, train_cindex = train_one_epoch(model, dataset.train, optimizer, scheduler, padder, device, criterion, aggr_method)
        val_loss, val_cindex = evaluate(model, dataset.val, padder, device, criterion, aggr_method)
        test_loss, test_cindex = evaluate(model, dataset.test, padder, device, criterion, aggr_method)
        if logger:
            logger.log_epoch(epoch + 1, {
                'train_loss': train_loss,
                'train_cindex': train_cindex,
                'val_loss': val_loss,
                'val_cindex': val_cindex,
                'test_loss': test_loss,
                'test_cindex': test_cindex
            })
        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(
                f"    -> New best model found! Val C-Index: {best_cindex:.4f}. Checkpoint updated."
            )
        else:
            epochs_no_improve += 1
        print(
            f"[Epoch {epoch + 1:03d}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train C-Index: {train_cindex:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val C-Index: {val_cindex:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test C-Index: {test_cindex:.4f}"
        )
        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs.")
            break
    model.load_state_dict(best_model_state)
    test_loss, test_cindex = evaluate(model, dataset.test, padder, device, criterion, aggr_method)
    return Metrics(loss=test_loss, c_index=test_cindex)
