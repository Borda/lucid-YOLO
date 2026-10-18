# 🔎 Inference

Turning a trained model's raw outputs into detections: one-call prediction for a single image per task, and the decoders behind it, NMS-free top-k beside the NMS baseline. Generated from the docstrings in `src/lucid_yolo/predict/` and `src/lucid_yolo/decode/`.

## Prediction

::: lucid_yolo.predict.predict_image

::: lucid_yolo.predict.predict_segmentation

::: lucid_yolo.predict.predict_oriented

::: lucid_yolo.predict.predict_keypoints

## Decoders

::: lucid_yolo.decode.TopKDecoder

::: lucid_yolo.decode.NMSDecoder

::: lucid_yolo.decode.RotatedNMSDecoder
