# SPAN Release

This release snapshot contains the vision-facing SPAN code path:

- slide-level classification
- patch-level segmentation
- slide-level survival analysis
- the shared `src.span` model implementation

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

Task configs read feature files from `data_root`, which defaults to `SPAN_DATA_ROOT` and then `data`.

```bash
export SPAN_DATA_ROOT=/path/to/features
```

You can also edit `data_root` in:

- `configs/classification.yaml`
- `configs/segmentation.yaml`
- `configs/survival.yaml`

For survival tasks, clinical TSV files default to `${SPAN_DATA_ROOT}/TCGA_clinical`. Override with `SPAN_CLINICAL_ROOT` or `clinical_root=/path/to/clinical`.

For `BRACS7`, subtype labels default to `${SPAN_DATA_ROOT}/labels`. Override with `SPAN_LABEL_ROOT` or `label_root=/path/to/labels`.

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
