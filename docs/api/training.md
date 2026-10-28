# 🏋️ Training

The Lightning pieces a training run is assembled from: the module that owns the model, loss and optimizer, the data module that builds the loaders, and the two callbacks the recipe needs. Generated from the docstrings in `src/lucid_yolo/ptl/`.

## Module and data

::: lucid_yolo.ptl.DetectionLitModule

::: lucid_yolo.ptl.DetectionDataModule

## Callbacks

::: lucid_yolo.ptl.EMACallback

::: lucid_yolo.ptl.CloseMosaicCallback

## Detection loss

`DualBranchLoss` takes one `BranchPrediction(logits, boxes)` per branch, followed by shared anchor points and padded targets. Its output records the attribute alpha float by default, or the supplied alpha tensor when overridden.

::: lucid_yolo.losses.BranchPrediction

::: lucid_yolo.losses.DualBranchLoss

::: lucid_yolo.losses.DualLossOutput
