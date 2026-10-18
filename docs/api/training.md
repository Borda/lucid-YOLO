# 🏋️ Training

The Lightning pieces a training run is assembled from: the module that owns the model, loss and optimizer, the data module that builds the loaders, and the two callbacks the recipe needs. Generated from the docstrings in `src/lucid_yolo/ptl/`.

## Module and data

::: lucid_yolo.ptl.DetectionLitModule

::: lucid_yolo.ptl.DetectionDataModule

## Callbacks

::: lucid_yolo.ptl.EMACallback

::: lucid_yolo.ptl.CloseMosaicCallback
