import os
import random
from dataclasses import dataclass
from typing import Sequence, Optional

import torch

from lib.utils.coord_aug import build_coord_aug_transform
from .clinical import preprocess_tsv, load_bracs_subtype_index_map
from .h5 import (
    H5SlideDataset, apply_coord_aug_to_item, apply_coord_aug_to_items,
    build_h5_loaders, collect_h5_file_paths, compute_class_weights,
    load_h5_slide_sample, load_raw_h5_slide_sample, load_pickle,
    normalize_coords, single_item_collate, sort_instances,
)

@dataclass
class DatasetBundle:
    train: Sequence
    val: Sequence
    test: Sequence
    num_classes: int
    class_weights: torch.Tensor


def _cfg_get(cfg, key: str, default: Optional[str] = None) -> Optional[str]:
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    return None if value is None else str(value)


def _data_root(cfg) -> str:
    return _cfg_get(cfg, "data_root") or os.environ.get("SPAN_DATA_ROOT", "data")


def _clinical_root(cfg) -> str:
    return _cfg_get(cfg, "clinical_root") or os.environ.get(
        "SPAN_CLINICAL_ROOT",
        os.path.join(_data_root(cfg), "TCGA_clinical"),
    )


def _label_root(cfg) -> str:
    return _cfg_get(cfg, "label_root") or os.environ.get(
        "SPAN_LABEL_ROOT",
        os.path.join(_data_root(cfg), "labels"),
    )


def _feature_dir(cfg, dataset: str, variant: str, split: Optional[str] = None) -> str:
    name = f"{dataset}_{variant}" if split is None else f"{dataset}_{variant}_{split}"
    return os.path.join(_data_root(cfg), name)


def _feature_file(cfg, filename: str) -> str:
    return os.path.join(_data_root(cfg), filename)


def prepare_classification_dataset(cfg) -> DatasetBundle:
    ds = cfg.dataset
    features_variant = cfg.features_variant
    train_transform = build_coord_aug_transform(cfg)
    if ds == 'C16':
        data_dirs = [
            _feature_dir(cfg, 'C16', features_variant),
            _feature_dir(cfg, 'C16', features_variant, 'test'),
        ]
        train_loader, val_loader, test_loader = build_h5_loaders(
            data_dirs,
            (0.7, 0.15, 0.15),
            cfg.seed,
            binary=True,
            sort=True,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
        )
        class_weights = compute_class_weights(train_loader, 2)
        return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=2, class_weights=class_weights)
    if ds == 'BRACS':
        data_dirs = [
            _feature_dir(cfg, 'BRACS', features_variant),
            _feature_dir(cfg, 'BRACS', features_variant, 'val'),
            _feature_dir(cfg, 'BRACS', features_variant, 'test'),
        ]
        train_loader, val_loader, test_loader = build_h5_loaders(
            data_dirs,
            (0.7, 0.15, 0.15),
            cfg.seed,
            binary=False,
            sort=True,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
        )
        class_weights = compute_class_weights(train_loader, 3)
        return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=3, class_weights=class_weights)
    if ds == 'BRACS7':
        data_dirs = [
            _feature_dir(cfg, 'BRACS', features_variant),
            _feature_dir(cfg, 'BRACS', features_variant, 'val'),
            _feature_dir(cfg, 'BRACS', features_variant, 'test'),
        ]
        csv_path = os.path.join(_label_root(cfg), 'BRACS_subtype.csv')
        subtype_lookup = load_bracs_subtype_index_map(csv_path)
        train_loader, val_loader, test_loader = build_h5_loaders(
            data_dirs,
            (0.7, 0.15, 0.15),
            cfg.seed,
            binary=False,
            sort=True,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
            label_lookup=subtype_lookup,
            n_classes=7,
        )
        class_weights = compute_class_weights(train_loader, 7)
        return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=7, class_weights=class_weights)
    if ds == 'TCGA':
        train_pkl = _feature_file(cfg, f'TCGA_Lung_{features_variant}.pkl')
        slides = load_pickle(train_pkl)
        random.Random(cfg.seed).shuffle(slides)
        split_idx = int(0.85 * len(slides))
        test_set = slides[split_idx:]
        train_pool = slides[:split_idx]
        val_ratio = 0.176470588235294
        val_count = int(len(train_pool) * val_ratio)
        train_set = train_pool[:-val_count] or train_pool
        val_set = train_pool[-val_count:] if val_count > 0 else []
        normalize_coords(train_set)
        normalize_coords(val_set)
        normalize_coords(test_set)
        apply_coord_aug_to_items(train_set, train_transform)
        sort_instances(train_set, val_set, test_set)
        class_weights = compute_class_weights(train_set, 2)
        return DatasetBundle(train=train_set, val=val_set, test=test_set, num_classes=2, class_weights=class_weights)
    if ds in ('BRAC', 'trastuzumab', 'Yale_HER2'):
        if ds == 'BRAC':
            data_dirs = [_feature_dir(cfg, 'TCGA_BRAC', features_variant)]
        else:
            data_dirs = [_feature_dir(cfg, ds, features_variant)]
        train_loader, val_loader, test_loader = build_h5_loaders(
            data_dirs,
            (0.7, 0.15, 0.15),
            cfg.seed,
            binary=True,
            sort=True,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
        )
        class_weights = compute_class_weights(train_loader, 2)
        return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=2, class_weights=class_weights)
    raise ValueError(ds)


def prepare_segmentation_dataset(cfg) -> DatasetBundle:
    ds = cfg.dataset
    features_variant = cfg.get('features_variant', 'R50')
    train_transform = build_coord_aug_transform(cfg)
    if ds == 'C16':
        train_paths = [_feature_dir(cfg, 'C16', features_variant)]
        test_paths = [_feature_dir(cfg, 'C16', features_variant, 'test')]
        exclusion_ranges = {
            "tumor_095": ([[94713, 156154]], [[56950, 102674]]),
            "tumor_092": ([[108351, None]], [[None, None]]),
            "tumor_054": ([[None, None]], [[138890, None]]),
            "tumor_046": ([[None, None]], [[116731, None], [None, 70137]]),
        }
        train_loader = [load_raw_h5_slide_sample(fp) for fp in collect_h5_file_paths(train_paths)]
        test_loader = [load_raw_h5_slide_sample(fp) for fp in collect_h5_file_paths(test_paths)]
        for item in train_loader:
            if item.slide_index in exclusion_ranges:
                mask = create_mask(item.pos * 2, item.slide_index, exclusion_ranges)
                item.x = item.x[~mask]
                item.pos = item.pos[~mask]
                item.y = item.y[~mask]
        split = 0.15
        random.Random(cfg.seed).shuffle(train_loader)
        train_samples = int(len(train_loader) * (1 - split))
        train_set = train_loader[:train_samples]
        val_set = train_loader[train_samples:]
        normalize_coords(train_set)
        normalize_coords(val_set)
        normalize_coords(test_loader)
        apply_coord_aug_to_items(train_set, train_transform)
        sort_instances(train_set, val_set, test_loader)
        class_weights = torch.tensor([1.0, 1.0])
        return DatasetBundle(train=train_set, val=val_set, test=test_loader, num_classes=2, class_weights=class_weights)
    h5_splits = {
        'C17': (2 / 3 * 0.85, 2 / 3 * 0.15, 1 / 3),
        'SegCAMELYON': (0.7, 0.15, 0.15),
        'BACH': (0.7, 0.1, 0.2),
        'trastuzumab': (0.7, 0.1, 0.2),
        'Yale_HER2': (0.7, 0.1, 0.2),
    }
    if ds not in h5_splits and ds != 'BRAC':
        raise ValueError(ds)
    if ds == 'BRAC':
        data_dirs = [_feature_dir(cfg, 'TCGA_BRAC', features_variant)]
        split_ratio = (0.7, 0.15, 0.15)
        train_loader, val_loader, test_loader = build_h5_loaders(
            data_dirs,
            split_ratio,
            cfg.seed,
            binary=True,
            sort=True,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
        )
        class_weights = torch.tensor([1.0, 1.0])
        return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=2, class_weights=class_weights)
    data_dir_map = {
        'C17': _feature_dir(cfg, 'C17', features_variant),
        'SegCAMELYON': _feature_dir(cfg, 'SegCAMELYON', features_variant),
        'BACH': _feature_dir(cfg, 'BACH', features_variant),
        'trastuzumab': _feature_dir(cfg, 'trastuzumab', features_variant),
        'Yale_HER2': _feature_dir(cfg, 'Yale_HER2', features_variant),
    }
    data_dirs = [data_dir_map[ds]]
    split_ratio = h5_splits[ds]
    filter_inconsistent_positive_ratios = ds == 'C17'
    train_loader, val_loader, test_loader = build_h5_loaders(
        data_dirs,
        split_ratio,
        cfg.seed,
        binary=True,
        sort=True,
        num_workers=cfg.num_workers,
        filter_inconsistent_positive_ratios=filter_inconsistent_positive_ratios,
        train_transform=train_transform,
    )
    class_weights = torch.tensor([1.0, 1.0])
    return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=2, class_weights=class_weights)


def create_mask(positions: torch.Tensor, slide_name: str, exclusion_ranges):
    x_ranges, y_ranges = exclusion_ranges[slide_name]
    mask_x = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    mask_y = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    for x_range in x_ranges:
        x_start, x_end = x_range
        x_start = float('-inf') if x_start is None else x_start
        x_end = float('inf') if x_end is None else x_end
        mask_x |= ((positions[:, 0] >= x_start) & (positions[:, 0] <= x_end))
    for y_range in y_ranges:
        y_start, y_end = y_range
        y_start = float('-inf') if y_start is None else y_start
        y_end = float('inf') if y_end is None else y_end
        mask_y |= ((positions[:, 1] >= y_start) & (positions[:, 1] <= y_end))
    return mask_x & mask_y


def prepare_survival_dataset(cfg) -> DatasetBundle:
    ds = cfg.dataset
    n_groups = cfg.n_groups
    features_variant = cfg.get('features_variant', 'UNI')  # Use uppercase: R50, UNI, CONCH, V2
    train_transform = build_coord_aug_transform(cfg)

    if features_variant == 'CONCH':
        lung_dirs = [_feature_dir(cfg, 'TCGA_Lung', features_variant)]
        lgg_dirs = [_feature_dir(cfg, 'TCGA_LGG_unique', features_variant)]
        brac_dirs = [_feature_dir(cfg, 'TCGA_BRAC', features_variant)]
    elif features_variant == 'UNI':
        lung_dirs = [_feature_dir(cfg, 'TCGA_Lung', features_variant)]
        lgg_dirs = [_feature_dir(cfg, 'TCGA_LGG', features_variant)]
        brac_dirs = [_feature_dir(cfg, 'TCGA_BRAC', 'R50')]
    else:
        lung_dirs = [os.path.join(_data_root(cfg), 'TCGA_LUNG')]
        lgg_dirs = [_feature_dir(cfg, 'TCGA_LGG_unique', 'R50')]
        brac_dirs = [_feature_dir(cfg, 'TCGA_BRAC', 'R50')]

    dataset_configs = {
        'LUAD': {
            'data_dirs': lung_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'luad_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
        'LUSC': {
            'data_dirs': lung_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'lusc_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
        'LGG': {
            'data_dirs': lgg_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'lgg_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
        'BRAC': {
            'data_dirs': brac_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
        'TCGA_BRAC': {
            'data_dirs': brac_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
        'TCGA_LGG': {
            'data_dirs': lgg_dirs,
            'tsv_path': os.path.join(_clinical_root(cfg), 'lgg_tcga_pan_can_atlas_2018_clinical_data.tsv')
        },
    }

    config = dataset_configs[ds]
    data_dirs = config['data_dirs']
    tsv_path = config['tsv_path']

    # Use split ratio: 0.7/0.15/0.15
    train_loader, val_loader, test_loader = build_h5_loaders(
        data_dirs,
        split_ratio=(0.7, 0.15, 0.15),
        seed=cfg.seed,
        binary=True,
        sort=True,
        num_workers=cfg.num_workers,
        tsv_data=tsv_path,
        n_groups=n_groups,
        train_transform=train_transform,
    )

    class_weights = torch.ones(n_groups)
    return DatasetBundle(train=train_loader, val=val_loader, test=test_loader, num_classes=n_groups, class_weights=class_weights)
