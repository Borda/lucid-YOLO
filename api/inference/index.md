# 🔎 Inference

Turning a trained model's raw outputs into detections: loading a checkpoint, one-call prediction for a single image per task, scoring on COCO, and the decoders behind them, NMS-free top-k beside the NMS baseline. Generated from the docstrings in `src/lucid_yolo/predict.py`, `src/lucid_yolo/eval/` and `src/lucid_yolo/decode/`. The names in the first three sections are also importable from the top-level package, lazily (`lucid_yolo.load_eval_module`, `lucid_yolo.predict_image`, `lucid_yolo.score_coco`), so `import lucid_yolo` loads neither torch nor Lightning until one of them is used. [Using a checkpoint from Python](https://borda.github.io/lucid-YOLO/TRAINING/#using-a-checkpoint-from-python) shows them together.

## Loading a checkpoint

::: lucid_yolo.eval.checkpoint.load_eval_module

::: lucid_yolo.eval.checkpoint.pick_device

## Prediction

Each function takes the image as a file path or as a decoded `(3, H, W)` RGB tensor, one image per call.

::: lucid_yolo.predict.predict_image

::: lucid_yolo.predict.predict_segmentation

::: lucid_yolo.predict.predict_oriented

::: lucid_yolo.predict.predict_keypoints

## Scoring

::: lucid_yolo.eval.coco_score.score_coco

::: lucid_yolo.eval.coco_score.CocoScore

## Decoders

::: lucid_yolo.decode.TopKDecoder

::: lucid_yolo.decode.RotatedTopKDecoder

::: lucid_yolo.decode.NMSDecoder

::: lucid_yolo.decode.RotatedNMSDecoder

## Decode helpers

The per-task decode compositions that prediction, evaluation and export share, and the constants and gathers beneath them.

::: lucid_yolo.decode.decode_with_coefficients

::: lucid_yolo.decode.decode_with_keypoints

::: lucid_yolo.decode.gather_by_anchor

::: lucid_yolo.decode.gather_keypoint_sets

::: lucid_yolo.decode.points_to_letterboxed_original
