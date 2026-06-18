import os
import random
import pickle
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Optional, Dict

import torch
import pandas as pd
import h5py
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader

from lib.utils.coord_aug import build_coord_aug_transform
from src.span.preprocessing import (
    get_sorted_indices,
    ComposeTransforms,
)


@dataclass
class DatasetBundle:
    train: Sequence
    val: Sequence
    test: Sequence
    num_classes: int
    class_weights: torch.Tensor

def compute_class_weights(train_data, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from training data."""
    iterable = train_data.dataset if isinstance(train_data, DataLoader) else train_data
    counts = torch.zeros(num_classes)
    for item in iterable:
        y = item.graph_y
        cls = y.argmax(dim=-1).view(-1) if y.dim() >= 2 else y.view(-1).long()
        for c in cls:
            counts[c.item()] += 1
    inv_freq = 1.0 / counts.clamp(min=1)
    return inv_freq / inv_freq.sum() * num_classes


def _filter_pointwise_attrs_with_mask(sample: SimpleNamespace, mask: torch.Tensor) -> None:
    if mask is None:
        return

    for attr_name in ("x", "y"):
        if not hasattr(sample, attr_name):
            continue
        value = getattr(sample, attr_name)
        if isinstance(value, torch.Tensor) and value.shape[0] == mask.shape[0]:
            setattr(sample, attr_name, value[mask])


def apply_coord_aug_to_item(sample: SimpleNamespace, transform: Optional[ComposeTransforms]) -> None:
    if transform is None or not hasattr(sample, "pos"):
        return

    out = transform(sample.pos.float())
    if isinstance(out, tuple):
        transformed_pos, mask = out
    else:
        transformed_pos, mask = out, None

    if not isinstance(transformed_pos, torch.Tensor) or transformed_pos.numel() == 0:
        return

    sample.pos = transformed_pos.int()
    _filter_pointwise_attrs_with_mask(sample, mask)


def apply_coord_aug_to_items(items: Sequence, transform: Optional[ComposeTransforms]) -> None:
    if transform is None:
        return
    for item in items:
        apply_coord_aug_to_item(item, transform)

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


def preprocess_tsv(tsv_data, n_groups=5):
    duplicates = tsv_data.duplicated('Patient ID', keep=False)
    uniq_tsv_data = tsv_data[~duplicates].copy()
    status_mapping = {'1:DECEASED': 0, '0:LIVING': 1}
    uniq_tsv_data.loc[:, 'Overall Survival Status'] = uniq_tsv_data['Overall Survival Status'].map(status_mapping)
    uncensored_data = uniq_tsv_data[uniq_tsv_data['Overall Survival Status'] == 0]
    survival_months = uncensored_data['Overall Survival (Months)']
    discrete_labels, bins = pd.qcut(survival_months, q=n_groups, retbins=True, labels=False)
    bins[0] = uniq_tsv_data['Overall Survival (Months)'].min() - 1e-6
    bins[-1] = uniq_tsv_data['Overall Survival (Months)'].max() + 1e-6
    uniq_tsv_data.loc[:, 'Discrete Label'] = pd.cut(uniq_tsv_data['Overall Survival (Months)'], bins=bins, labels=False, include_lowest=True)
    data_dict = {}
    for idx, row in uniq_tsv_data.iterrows():
        if pd.notna(row['Overall Survival (Months)']) and row['Overall Survival (Months)'] != 0:
            key = row['Sample ID'][:15]
            censorship = torch.tensor([row['Overall Survival Status']], dtype=torch.float)
            survival_label = torch.tensor([row['Discrete Label']], dtype=torch.long)
            survival_time = row['Overall Survival (Months)']
            data_dict[key] = (censorship, survival_label, survival_time)
    return data_dict


def load_bracs_subtype_index_map(csv_path: str) -> Dict[str, int]:
    label_df = pd.read_csv(csv_path)
    type_to_index = {
        "Type_IC": 0,
        "Type_DCIS": 1,
        "Type_ADH": 2,
        "Type_FEA": 3,
        "Type_UDH": 4,
        "Type_N": 5,
        "Type_PB": 6,
    }
    mapping: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        subtype = row["type"]
        if subtype not in type_to_index:
            continue
        mapping[row["filename"].replace(".svs", "")] = type_to_index[subtype]
    return mapping


def normalize_coords(items: Sequence, coord_div: float = 224.0) -> None:
    with torch.no_grad():
        for item in items:
            ins_pos = item.pos.float() / float(coord_div)
            adjusted = ins_pos - ins_pos.min(dim=0, keepdim=True).values
            item.pos = adjusted.int()


def sort_instances(*loaders: Sequence) -> None:
    for loader in loaders:
        for item in loader:
            if not hasattr(item, 'pos') or not hasattr(item, 'x'):
                continue
            sorted_indices = get_sorted_indices(item.pos)
            item.pos = item.pos[sorted_indices]
            item.x = item.x[sorted_indices]
            # Some classification datasets may not have per-instance labels (y),
            # or y may be None. Guard to avoid indexing None.
            if hasattr(item, 'y') and isinstance(item.y, torch.Tensor):
                item.y = item.y[sorted_indices]



def load_pickle(path: str) -> List:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def collect_h5_file_paths(paths: Sequence[str]) -> List[str]:
    if isinstance(paths, str):
        paths = [paths]
    file_names: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.endswith('.h5'):
                    file_names.append(os.path.join(path, name))
        elif os.path.isfile(path) and path.endswith('.h5'):
            file_names.append(path)
    return file_names


def load_raw_h5_slide_sample(file_path: str) -> SimpleNamespace:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with h5py.File(file_path, 'r') as file:
        sample = SimpleNamespace()
        for key in file.keys():
            ds = file[key]
            if ds.ndim == 0 or ds.dtype.kind == 'O':
                continue
            data = torch.tensor(ds[:])
            setattr(sample, key, data)

    if hasattr(sample, 'label'):
        sample.graph_y = sample.label
        delattr(sample, 'label')
    if hasattr(sample, 'ratios'):
        sample.y = sample.ratios
        delattr(sample, 'ratios')
    if hasattr(sample, 'feats'):
        sample.x = sample.feats
        delattr(sample, 'feats')
    if hasattr(sample, 'cords'):
        sample.pos = sample.cords
        delattr(sample, 'cords')

    if not hasattr(sample, 'x') or not hasattr(sample, 'pos'):
        raise KeyError(f"Missing required keys in {file_path}; expected ('feats'/'x') and ('cords'/'pos').")

    sample.slide_index = os.path.basename(file_path).split('.')[0]
    return sample


def load_h5_slide_sample(
    file_path: str,
    coord_div: float = 224.0,
    sort: bool = True,
    transform: Optional[ComposeTransforms] = None,
) -> SimpleNamespace:
    sample = load_raw_h5_slide_sample(file_path)
    normalize_coords([sample], coord_div=coord_div)
    apply_coord_aug_to_item(sample, transform)
    if sort:
        sort_instances([sample])

    return sample


class H5SlideDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        binary: bool,
        filter_inconsistent_positive_ratios: bool = False,
        sort: bool = False,
        tsv_data: str = None,
        n_groups: int = None,
        transform: Optional[ComposeTransforms] = None,
        label_lookup: Optional[Dict[str, int]] = None,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(paths, str):
            paths = [paths]
        self.binary = binary
        self.sort = sort
        self.transform = transform
        self.label_lookup = label_lookup
        if label_lookup is not None:
            inferred = max(label_lookup.values()) + 1 if label_lookup else 0
            self.n_classes = int(n_classes if n_classes is not None else inferred)
        else:
            self.n_classes = None
        self.filter_inconsistent_positive_ratios = bool(filter_inconsistent_positive_ratios)
        self.file_names: List[str] = collect_h5_file_paths(paths)
        if self.filter_inconsistent_positive_ratios:
            filtered: List[str] = []
            for fp in self.file_names:
                with h5py.File(fp, 'r') as file:
                    label = torch.tensor(file['label'][:])[0]
                    ratios = torch.tensor(file['ratios'][:]) if 'ratios' in file else None
                    if label.argmax() > 0 and (ratios is None or ratios.sum() == 0):
                        continue
                filtered.append(fp)
            self.file_names = filtered
        if self.label_lookup is not None:
            self.file_names = [
                fp for fp in self.file_names
                if os.path.basename(fp).split('.')[0] in self.label_lookup
            ]
        if tsv_data is not None:
            tsv_df = pd.read_csv(tsv_data, sep='\t')
            self.clinical = preprocess_tsv(tsv_df, n_groups=n_groups)
            self.file_names = [f for f in self.file_names if os.path.basename(f).split('.')[0][:15] in self.clinical.keys()]
        else:
            self.clinical = None

    def __len__(self) -> int:
        return len(self.file_names)

    def set_aug_multiplier(self, multiplier: float) -> None:
        if self.transform and hasattr(self.transform, "transforms"):
            for transform in self.transform.transforms:
                if hasattr(transform, "set_multiplier"):
                    transform.set_multiplier(multiplier)

    def __getitem__(self, idx: int) -> SimpleNamespace:
        file_path = self.file_names[idx]
        sample = load_h5_slide_sample(
            file_path=file_path,
            coord_div=224.0,
            sort=self.sort,
            transform=self.transform,
        )

        if self.label_lookup is not None:
            subtype_idx = self.label_lookup.get(sample.slide_index)
            if subtype_idx is None:
                raise KeyError(f"Missing subtype label for slide: {sample.slide_index}")
            n_rows = int(sample.graph_y.size(0)) if hasattr(sample, 'graph_y') and isinstance(sample.graph_y, torch.Tensor) else 1
            one_hot = torch.zeros((n_rows, int(self.n_classes)), dtype=torch.float32)
            one_hot[:, int(subtype_idx)] = 1.0
            sample.graph_y = one_hot

        if self.binary and hasattr(sample, 'graph_y'):
            y = sample.graph_y
            if y.dim() == 2 and y.size(1) > 1:
                one_hot = torch.zeros((y.shape[0], 2))
                positive = (y[:, 1:] > 0).any(dim=1)
                one_hot[torch.arange(y.shape[0]), positive.long()] = 1
                sample.graph_y = one_hot
        if self.clinical is not None:
            key = sample.slide_index[:15]
            sample.censorship = self.clinical[key][0]
            sample.survival_label = self.clinical[key][1]
            sample.survival_time = self.clinical[key][2]
        return sample


def single_item_collate(batch):
    return batch[0] if len(batch) == 1 else batch


def build_h5_loaders(
    paths: Sequence[str],
    split_ratio: Tuple[float, float, float],
    seed: int,
    binary: bool,
    sort: bool,
    num_workers: int,
    filter_inconsistent_positive_ratios: bool = False,
    tsv_data: str = None,
    n_groups: int = None,
    train_transform: Optional[ComposeTransforms] = None,
    label_lookup: Optional[Dict[str, int]] = None,
    n_classes: Optional[int] = None,
):
    base_dataset = H5SlideDataset(
        paths,
        binary=binary,
        filter_inconsistent_positive_ratios=filter_inconsistent_positive_ratios,
        sort=sort,
        tsv_data=tsv_data,
        n_groups=n_groups,
        transform=None,
        label_lookup=label_lookup,
        n_classes=n_classes,
    )
    total = len(base_dataset)
    r_train, r_val, r_test = split_ratio
    n_train = int(total * r_train)
    n_val = int(total * r_val)
    n_test = total - n_train - n_val

    all_indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:n_train + n_val]
    test_indices = all_indices[n_train + n_val:n_train + n_val + n_test]

    train_files = [base_dataset.file_names[i] for i in train_indices]
    val_files = [base_dataset.file_names[i] for i in val_indices]
    test_files = [base_dataset.file_names[i] for i in test_indices]

    train_ds = H5SlideDataset(
        train_files,
        binary=binary,
        filter_inconsistent_positive_ratios=filter_inconsistent_positive_ratios,
        sort=sort,
        tsv_data=tsv_data,
        n_groups=n_groups,
        transform=train_transform,
        label_lookup=label_lookup,
        n_classes=n_classes,
    )
    val_ds = H5SlideDataset(
        val_files,
        binary=binary,
        filter_inconsistent_positive_ratios=filter_inconsistent_positive_ratios,
        sort=sort,
        tsv_data=tsv_data,
        n_groups=n_groups,
        transform=None,
        label_lookup=label_lookup,
        n_classes=n_classes,
    )
    test_ds = H5SlideDataset(
        test_files,
        binary=binary,
        filter_inconsistent_positive_ratios=filter_inconsistent_positive_ratios,
        sort=sort,
        tsv_data=tsv_data,
        n_groups=n_groups,
        transform=None,
        label_lookup=label_lookup,
        n_classes=n_classes,
    )

    loader_args = dict(batch_size=1, num_workers=num_workers, collate_fn=single_item_collate, persistent_workers=False)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


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
