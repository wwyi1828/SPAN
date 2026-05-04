# SPAN

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-b31b1b?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2406.09333)

This repository contains the SPAN implementation for:

- slide-level classification
- patch-level segmentation
- slide-level survival analysis
- the shared `src.span` model implementation

Feature files can be prepared from whole-slide images using [PatchPreprocess](https://github.com/wwyi1828/PatchPreprocess), then passed to SPAN through `data_root`.

## Layout

```text
configs/          Hydra configs for vision tasks and model variants
src/span/         Core SPAN modules
tasks/vision/     Classification, segmentation, survival entrypoints
lib/utils/        Runtime helpers used by the vision tasks
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Prepare slide-level feature files in the layout expected by the selected task config, then point `data_root` to that directory. `data_root` defaults to `SPAN_DATA_ROOT` and then `data`.

```bash
export SPAN_DATA_ROOT=/path/to/features
```

You can also edit `data_root` in:

- `configs/classification.yaml`
- `configs/segmentation.yaml`
- `configs/survival.yaml`

The `dataset` field in each config selects the corresponding loader in `tasks/vision/shared/data.py`.

Survival tasks also require the corresponding clinical metadata.

## Run

```bash
python -m tasks.vision.slide.classification.main data_root=/path/to/features
python -m tasks.vision.patch.segmentation.main data_root=/path/to/features
python -m tasks.vision.slide.survival.main data_root=/path/to/features
```

W&B logging is disabled by default. Enable it explicitly when needed:

```bash
python -m tasks.vision.slide.classification.main logging.wandb.enabled=true
```

## License

This code is released under the MIT License.
