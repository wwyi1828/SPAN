import os
import pickle
from types import SimpleNamespace
from typing import Sequence, List, Tuple, Optional, Dict

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.span.preprocessing import get_sorted_indices, ComposeTransforms
from .clinical import preprocess_tsv

def compute_class_weights(train_data, num_classes: int) -> torch.Tensor:
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
        if value is None:
            continue
        if value.shape[0] != mask.shape[0]:
            raise ValueError(f"Patch attribute {attr_name!r} does not match the coordinate count")
        setattr(sample, attr_name, value[mask])


def apply_coord_aug_to_item(sample: SimpleNamespace, transform: Optional[ComposeTransforms]) -> None:
    if transform is None:
        return
    transformed_pos, mask = transform(sample.pos.float())

    sample.pos = transformed_pos.int()
    _filter_pointwise_attrs_with_mask(sample, mask)


def apply_coord_aug_to_items(items: Sequence, transform: Optional[ComposeTransforms]) -> None:
    if transform is None:
        return
    for item in items:
        apply_coord_aug_to_item(item, transform)


def normalize_coords(items: Sequence, coord_div: float = 224.0) -> None:
    with torch.no_grad():
        for item in items:
            ins_pos = item.pos.float() / float(coord_div)
            adjusted = ins_pos - ins_pos.min(dim=0, keepdim=True).values
            item.pos = adjusted.int()


def sort_instances(*loaders: Sequence) -> None:
    for loader in loaders:
        for item in loader:
            sorted_indices = get_sorted_indices(item.pos)
            item.pos = item.pos[sorted_indices]
            item.x = item.x[sorted_indices]
            # Some classification datasets may not have per-instance labels (y),
            # or y may be None. Guard to avoid indexing None.
            if hasattr(item, 'y') and isinstance(item.y, torch.Tensor):
                item.y = item.y[sorted_indices]


def load_pickle(path: str) -> List:
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

    if sample.pos.ndim != 2 or sample.pos.shape[1] != 2 or sample.pos.shape[0] == 0:
        raise ValueError(f"{file_path}: coordinates must have nonempty shape [num_patches, 2]")
    if sample.x.ndim != 2 or sample.x.shape[0] != sample.pos.shape[0]:
        raise ValueError(f"{file_path}: features must have shape [num_patches, feature_dim] matching coordinates")
    if hasattr(sample, 'y') and sample.y.shape[0] != sample.pos.shape[0]:
        raise ValueError(f"{file_path}: patch labels must match the coordinate count")

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
