# SPAN: Sparse Hierarchical Modeling for Digital Pathology

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-b31b1b?logo=adobeacrobatreader\&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wu_Learning_Spatial-Preserving_Hierarchical_Representations_for_Digital_Pathology_CVPRF_2026_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="assets/sparse_compute.gif" alt="How SPAN computes on a sparse patch grid: index, window attention, downsample, upsample" width="85%">
</p>

Official implementation of **SPAN: Learning Spatial-Preserving Hierarchical Representations for Digital Pathology**.

Download the preprocessed datasets from [Google Drive](https://drive.google.com/drive/folders/1N7yNi_QigARVNeOuxuzFia53UUTspAPd?usp=sharing).

SPAN provides the paper's reference model and reusable sparse hierarchical
blocks for whole-slide image (WSI) modeling. Configure downsampling, upsampling,
and local/global attention through `src.span` and `slide_configs`.

Out of the box it supports:

* slide-level classification
* patch-level segmentation
* slide-level survival analysis
* custom sparse hierarchical models assembled through `src.span` and the `slide_configs` spec

Unlike standard MIL models that treat a slide as an unordered bag of patch features, SPAN preserves the spatial layout of WSI patches and builds sparse hierarchical representations directly on the patch grid.

---

## News

* [HiST](https://github.com/wwyi1828/HiST), our follow-up work on spatial gene expression prediction from histology, is now available with code (ICML 2026). [ICML 2026 Poster](https://icml.cc/virtual/2026/poster/61463)

---

## Overview

Whole-slide images are extremely large, but tissue patches usually occupy only a sparse subset of the full slide canvas. SPAN is designed to model this sparse spatial structure directly.

The recommended preprocessing workflow is:

```text
raw WSIs
  -> modified CLAM patch extraction
  -> PatchPreprocess feature extraction
  -> SPAN training / evaluation through data_root
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/wwyi1828/SPAN.git
cd SPAN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Related repositories

| Repository | Role |
|---|---|
| [Coordinate-aligned CLAM](https://github.com/wwyi1828/CLAM) | Extract WSI patches on a globally aligned grid. |
| [PatchPreprocess](https://github.com/wwyi1828/PatchPreprocess#wsi-feature-extraction) | Encode extracted patches and export feature files for SPAN. |
| [HiST](https://github.com/wwyi1828/HiST) | Related hierarchical sparse transformer for spatial gene expression prediction from H&E histology. |

## Data preparation

For the downloadable processed datasets linked above, proceed directly to
[Usage](#usage). For your own WSIs, follow the preprocessing route below.

SPAN does **not** read raw WSI files directly. It consumes pre-extracted slide-level feature files that contain both patch features and patch coordinates.

We recommend [PatchPreprocess](https://github.com/wwyi1828/PatchPreprocess#wsi-feature-extraction) with PNG patches from [coordinate-aligned CLAM](https://github.com/wwyi1828/CLAM). Match the saved coordinate step to the loader as described below.

### Why the coordinate convention matters

The model uses integer lattice coordinates for spatial indexing, downsampling,
and attention windows. The default loader divides stored coordinates by 224, subtracts the per-axis
minimum, and casts them to integers. It does not check whether this conversion
merges sites.
See [`normalize_coords`](tasks/vision/shared/data.py).

| Stored `cords` | After normalization and integer conversion | Interpretation |
|---|---|---|
| `(0, 0), (224, 0), (448, 224)` | `(0, 0), (1, 0), (2, 1)` | Aligned grid with distinct sites. |
| `(0, 0), (224, 0), (240, 0)` | `(0, 0), (1, 0), (1, 0)` | Two sites collapse to the same lattice coordinate. |

Use globally aligned coordinates with a step matching the loader's divisor.
Check the saved coordinate units when changing patch level, magnification, or
step size. For a different coordinate step, adapt `coord_div` in the data-loading
code; it is not exposed as a top-level Hydra setting. After conversion, each
patch must have a unique, non-negative integer coordinate pair.

Each H5 feature file should correspond to one slide and contain at least:

```text
feats: patch features, shape [num_patches, feature_dim]
cords: patch coordinates, shape [num_patches, 2]
```

Optional fields include:

```text
label:   slide-level label
ratios:  patch-level annotation overlap ratios
caption: optional slide caption
```

Custom preprocessing pipelines can also be used if they produce the feature fields
and coordinate convention described above.

Set `data_root` to the folder containing the prepared feature files:

```bash
python -m tasks.vision.slide.classification.main \
  data_root=/path/to/features
```

The exact dataset-specific folder names are defined by the corresponding config and dataset loader.

---

## Quick feature-file check

Before training, you can quickly inspect one H5 file:

```python
import h5py

path = "/path/to/features/example_slide.h5"

with h5py.File(path, "r") as f:
    print(list(f.keys()))
    print("feats:", f["feats"].shape)
    print("cords:", f["cords"].shape)
    print("first coords:", f["cords"][:5])
```

Expected fields include `feats` and `cords`. For the default 224 x 224 non-overlapping patch setup, coordinates usually follow a 224-step WSI grid.

---

## Usage

### Slide-level classification

```bash
python -m tasks.vision.slide.classification.main \
  data_root=/path/to/features
```

You can override the dataset and feature variant from the command line:

```bash
python -m tasks.vision.slide.classification.main \
  dataset=C16 \
  features_variant=R50 \
  data_root=/path/to/features
```

### Patch-level segmentation

```bash
python -m tasks.vision.patch.segmentation.main \
  dataset=C16 \
  features_variant=R50 \
  data_root=/path/to/features
```

For patch-level segmentation, each slide should provide patch-level labels or annotation overlap ratios, usually stored as `ratios` in H5.

### Slide-level survival analysis

```bash
python -m tasks.vision.slide.survival.main \
  data_root=/path/to/features
```

If the clinical metadata is stored separately, set:

```bash
export SPAN_CLINICAL_ROOT=/path/to/clinical_metadata
```

or pass it directly:

```bash
python -m tasks.vision.slide.survival.main \
  data_root=/path/to/features \
  clinical_root=/path/to/clinical_metadata
```

---

## Configuration

SPAN is configured with [Hydra](https://hydra.cc/). Everything lives under `configs/`:

```text
configs/
├── classification.yaml          # task config: slide-level classification
├── segmentation.yaml            # task config: patch-level segmentation
├── survival.yaml                # task config: slide-level survival
├── model/
│   ├── span.yaml                # model config used by classification / survival
│   └── segmentation_span.yaml   # model config used by segmentation (encoder–decoder)
└── logging/
    └── default.yaml             # logging / W&B config
```

### How the configs compose

Each **task config** (`classification.yaml`, `segmentation.yaml`, `survival.yaml`) is the
entrypoint config for that task. Its `defaults:` list pulls in a **model config** and the
**logging config**, then `_self_` lets the task file override any of their values:

```yaml
defaults:
  - model: span        # load configs/model/span.yaml
  - logging: default   # load configs/logging/default.yaml
  - _self_             # values in this file win over the loaded defaults
```

Override individual settings on the command line with dotted paths:

```bash
python -m tasks.vision.slide.classification.main \
  data_root=/path/to/features \
  model.trans_type=swin \
  logging.wandb.enabled=true
```

### Task config (`classification.yaml` / `segmentation.yaml` / `survival.yaml`)

Top-level run settings shared by all tasks:

| Key | Meaning |
| --- | --- |
| `task` | Task name, used for run naming and result paths. |
| `dataset` | Which dataset/loader to use (e.g. `C16`, `TCGA_BRAC`). |
| `features_variant` | Which pre-extracted feature set to read (e.g. `R50`, `UNI`). |
| `data_root` | Folder holding the H5 feature files. Defaults to `$SPAN_DATA_ROOT` or `data`. |
| `seed` / `epochs` / `num_workers` | Standard training run settings. |
| `n_groups` | (survival only) Number of discrete-time risk bins. |

Each task file also has a block of **task-specific** options:

| Block | Key fields | Meaning |
| --- | --- | --- |
| `training` | `lr`, `weight_decay`, `early_stop_patience` | Optimization settings. |
| `training.coord_aug` | `enabled`, `schedule`, `mirror` / `shear` / `rotate` / `drop` | Coordinate-space augmentation on the patch grid, with an optional cosine schedule that anneals the augmentation strength over training. |
| `classification` | `aggr_method`, `cls_drop`, `head_act`, `head_div`, `head_norm` | Slide-level aggregation head and its dropout/activation/norm. |
| `segmentation` | `dice_weight`, `threshold` | Dice/BCE loss mix and the decision threshold for patch prediction. |
| `survival` | `alpha`, `aggregation`, `cls_drop`, `head_act` | Survival loss balance and the aggregation head. |

### Model config (`model/span.yaml`, `model/segmentation_span.yaml`)

This file defines the network architecture. See [Supported modules](#supported-modules) for
the block-type menus; the remaining fields are:

| Key | Meaning |
| --- | --- |
| `slide_configs` | Network spec: each digit is the number of transformer layers in a stage; `-` separates the encoder from an optional decoder. |
| `channel_factor` | Channel-width multiplier applied at each downsampling stage. |
| `embed_dim_div` | Sets the working width: `embed_dim = feature_dim / embed_dim_div` (and `num_heads = embed_dim / 64`). |
| `econvs_type` / `dconvs_type` | Encoder / decoder down/upsampling block type. |
| `trans_type` / `dtrans_type` | Encoder / decoder attention block type. |
| `window_size` | Attention window extent (windows span `2 * window_size` grid cells). |
| `ff_ratio`, `attn_drop`, `proj_drop`, `drop_path` | Transformer feed-forward ratio and dropout/stochastic-depth rates. |
| `input_projection` | The first-stage linear projection (`mlp_ratio`, `dropout`, `prenorm`, `bias`, …) that maps raw patch features into the working width. |
| `token_init_types` | Global summary tokens prepended to the sequence; each entry is `max` / `mean` / `std` pooling, or a `fix…` (fixed) / `lrn…` (learnable) randomly-initialized token. |
| `global_strategy` | How the global token interacts with conv stages: `identity`, `proj`, or `kernel`. |
| `pos_emb` | Positional encoding for attention: `rpb`, `alibi`, `rope`, or `none`. |
| `kernel_size` / `stride` / `dilation` | Sparse conv geometry on the patch grid. |
| `edge_mode` | Boundary handling for sparse convs: `none`, `count`, or `partial`. |
| `conv_bias`, `conv_norm_position`, `share_qkv` | Conv bias, pre/post norm placement, and whether Q/K/V projections are shared. |
| `skip_first` / `connection` | (Decoder only) encoder→decoder skip connections and how they merge (`concat`, `add`, `none`). |

### Logging config (`logging/default.yaml`)

| Key | Meaning |
| --- | --- |
| `mode` | Logging mode; does not select the best model. `disabled` requests offline W&B logging. |
| `results.dir` | Directory for metrics and logs. |
| `wandb.enabled` | W&B logging, **off by default**; enable with `logging.wandb.enabled=true`. |
| `wandb.project` / `wandb.entity` / `wandb.tags` / `wandb.notes` | W&B run metadata. |

---

## Supported modules

SPAN is built from interchangeable sparse blocks selected through the model config (`configs/model/*.yaml`) and overridable on the command line. Because every block operates on the integer patch grid, downsampling and upsampling stay index-consistent across resolution levels.

### Down/upsampling (`econvs_type` / `dconvs_type`)

| Type | Role | Notes |
| --- | --- | --- |
| `dwpatchconv` / `patchconv` | Sparse strided convolution downsampling | Depthwise (`dw`) or full conv; default encoder block |
| `patchmerge` | Sparse Swin-style patch merging | Concatenates a `stride × stride` neighborhood, then projects |
| `dwpatchinvs` / `patchinvs` | Sparse strided **upsample** (transpose conv) | Used in the decoder for segmentation |
| `pixelshuffle` | Sparse pixel-shuffle upsample | Sub-pixel expansion on the grid |
| `convnext` / `linear` / `projection` / `identity` | Same-resolution feature blocks | No spatial change |

Most blocks provide sparse (also available with the `s_` prefix) and dense
(`d_`) implementations of the same operation, such as `d_patchmerge` and
`d_pixelshuffle`. The default configurations use sparse implementations, which
were faster in our benchmarks at typical tissue sparsity. Depthwise blocks such
as `dwpatchconv` were an exception: the dense implementation was slightly faster.
Performance still depends on tissue occupancy, coordinate range, and hardware.

### Window / global attention (`trans_type` / `dtrans_type`)

| Mode | Description |
| --- | --- |
| `long` | Longformer-style: shifted + non-shifted window self-attention plus a global-token relay (default) |
| `long_noshift` | Same as `long` without the shifted window |
| `swin` / `tradswin` | Pure shifted/non-shifted window self-attention (Swin-style), no global tokens |
| `swin_global` | Window attention with an added global-token stage |

The window extent is controlled by `window_size` (windows span `2 * window_size` grid cells). Example overrides:

```bash
python -m tasks.vision.slide.classification.main \
  data_root=/path/to/features \
  model.econvs_type=patchmerge \
  model.trans_type=swin \
  model.window_size=4
```

---

## Repository layout

```text
configs/          Hydra configs for vision tasks and model variants
src/span/         Core SPAN modules
tasks/vision/     Classification, segmentation, and survival entrypoints
lib/utils/        Runtime helpers used by the vision tasks
assets/           Figures and README assets
```

---

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{wu2026learning,
  title={Learning Spatial-Preserving Hierarchical Representations for Digital Pathology},
  author={Wu, Weiyi and Diao, Xingjian and Zhang, Chunhui and Gao, Chongyang and Xu, Xinwen and Li, Siting and Gui, Jiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5484--5494},
  year={2026}
}
```

---

## License

This code is released under the MIT License.
