# 📦 Export

Writing a trained checkpoint as an ONNX file: `graph_from_module` rebuilds a task's suppression-free one-to-one graph from a loaded checkpoint, and `export_graph` writes it. The `lucid-export` command wraps both; how to run it is under [Exporting to ONNX](../TRAINING.md#exporting-to-onnx). Generated from the docstrings in `src/lucid_yolo/export.py`.

## From a checkpoint

::: lucid_yolo.export.graph_from_module

::: lucid_yolo.export.export_graph

::: lucid_yolo.export.ONNX_OPSET

## Graphs

::: lucid_yolo.export.E2EExportGraph

::: lucid_yolo.export.DetectExportGraph

::: lucid_yolo.export.SegmentExportGraph

::: lucid_yolo.export.OrientedExportGraph

::: lucid_yolo.export.KeypointExportGraph
