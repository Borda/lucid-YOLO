# 🧠 Models

The four task models and the builders that construct them from a variant name. Each builder takes the `n`/`s`/`m`/`l`/`x` variant and a class count and returns a ready `nn.Module`; the model returns the dual-head outputs that training supervises and decoding reads. Generated from the docstrings in `src/lucid_yolo/models/`.

## Builders

::: lucid_yolo.models.build_detector

::: lucid_yolo.models.build_segmenter

::: lucid_yolo.models.build_obb_detector

::: lucid_yolo.models.build_keypoint_detector

## Task models

::: lucid_yolo.models.Detector

::: lucid_yolo.models.Segmenter

::: lucid_yolo.models.OrientedDetector

::: lucid_yolo.models.KeypointDetector

## Heads and outputs

::: lucid_yolo.models.DualDetectionHead

::: lucid_yolo.models.DualHeadOutput

::: lucid_yolo.models.SegmentOutput
