# SPDX-License-Identifier: Apache-2.0
"""Private gathering metrics and image identity for Lightning validation.

Holds the whole-epoch rotated mAP and OKS AP metrics and the dataset-position identity the validation reader
attaches. The datamodule wraps its sequential reader in ``_ValidationDataset``, and the Lightning module feeds decoded
tensors to the task metric. The metrics call the existing scoring functions without changing their protocols and
return their float dictionaries. ``ptl.module``, ``ptl.datamodule`` and private target transport use these records;
mismatched prediction, target and identity counts raise before accumulation.

Records travel as byte views with integer shape/dtype metadata in two cat-reduced list states. This lets ragged
per-image tensors pass through ordinary tensor collectives while keeping their original precision, including empty
keypoint shapes. States follow the metric's device so a CUDA metric does not ask NCCL to gather CPU tensors.

The first occurrence of each identity is kept. Local calls retain insertion order; after a gather the module sorts
by dataset position, the order its sequential validation loader already traverses. Restoring that order also
preserves stable score ties after DistributedSampler sharding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from lucid_yolo.data.targets import Targets
from lucid_yolo.eval.coco_eval import evaluate_keypoints
from lucid_yolo.eval.dota_eval import evaluate_rotated_map

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.int32, torch.bool)


@dataclass
class _ValidationTargets(Targets):
    """Carry a validation reader's position after its geometry transforms finish."""

    _image_id: int = -1


class _ValidationDataset(Dataset[tuple[Tensor, Targets]]):
    """Attach stable dataset positions without changing the public sample pair."""

    def __init__(self, source: Dataset[tuple[Tensor, Targets]]) -> None:
        """Retain the transformed validation reader."""
        self._source = source

    def __len__(self) -> int:
        """Return the underlying reader's sample count."""
        return len(self._source)  # type: ignore[arg-type]

    @property
    def image_ids(self) -> tuple[int, ...] | None:
        """Preserve reader identities used by the evaluator's tile-pairing guard."""
        image_ids = getattr(self._source, "image_ids", None)
        return None if image_ids is None else tuple(int(value) for value in image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Attach position identity to a fresh container, leaving reader caches alone."""
        image, target = self._source[index]
        return image, _ValidationTargets(**vars(target), _image_id=index)


class _ValidationMetric(Metric):
    """Gather lossless per-image tensors and retain the first record per identity."""

    full_state_update = False
    _prediction_fields: tuple[str, ...]
    _target_fields: tuple[str, ...]
    _metadata: list[Tensor]
    _values: list[Tensor]

    def __init__(self, *, dataset_order: bool = False) -> None:
        """Register ragged states; optionally restore sequential dataset traversal."""
        super().__init__()
        self._dataset_order = dataset_order
        self.add_state("_metadata", default=[], dist_reduce_fx="cat")
        self.add_state("_values", default=[], dist_reduce_fx="cat")
        self._image_count = 0
        self._seed_empty_states()

    @property
    def image_count(self) -> int:
        """Return the number of local images since the last reset."""
        return self._image_count

    def _seed_empty_states(self) -> None:
        """Give empty ranks the same collective dtype and dimensionality as populated ranks."""
        self._metadata.append(torch.empty(0, dtype=torch.int64, device=self.device))
        self._values.append(torch.empty(0, dtype=torch.uint8, device=self.device))

    def reset(self) -> None:
        """Clear epoch data while retaining typed empty collective participants."""
        super().reset()
        self._image_count = 0
        self._seed_empty_states()

    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]], image_ids: list[int]) -> None:
        """Append aligned images in caller order, retaining tensor shapes and dtypes."""
        if not len(preds) == len(targets) == len(image_ids):
            raise ValueError("predictions, targets and image ids must align")
        self._image_count += len(preds)
        for prediction, target, image_id in zip(preds, targets, image_ids, strict=True):
            metadata = [image_id]
            for record, fields in ((prediction, self._prediction_fields), (target, self._target_fields)):
                for name in fields:
                    value = record[name].detach().contiguous()
                    metadata.extend((_DTYPES.index(value.dtype), value.ndim, *value.shape))
                    self._values.append(value.view(torch.uint8).flatten().to(self.device))
            self._metadata.append(torch.tensor(metadata, dtype=torch.int64, device=self.device))

    def _images(self) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Decode gathered records, deduplicate before scoring and preserve traversal order."""
        metadata = dim_zero_cat(self._metadata).cpu().tolist()
        values = dim_zero_cat(self._values).cpu()
        images: dict[int, tuple[dict[str, Tensor], dict[str, Tensor]]] = {}
        offset = cursor = 0
        while offset < len(metadata):
            image_id = metadata[offset]
            offset += 1
            records = []
            for fields in (self._prediction_fields, self._target_fields):
                record: dict[str, Tensor] = {}
                for name in fields:
                    dtype, ndim = _DTYPES[metadata[offset]], metadata[offset + 1]
                    shape = metadata[offset + 2 : offset + 2 + ndim]
                    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                    # Clone aligns byte offsets for view(dtype), including after bool fields.
                    record[name] = values[cursor : cursor + size].clone().view(dtype).reshape(shape)
                    offset += 2 + ndim
                    cursor += size
                records.append(record)
            images.setdefault(image_id, (records[0], records[1]))
        # Torchmetrics replaces list states by tensors only during collective sync.
        # Local callers keep their supplied order, even with nonmonotonic dataset IDs.
        gathered = isinstance(self._metadata, Tensor)
        ordered = sorted(images) if self._dataset_order and gathered else list(images)
        return [images[key][0] for key in ordered], [images[key][1] for key in ordered]


class _RotatedMAP(_ValidationMetric):
    """Compute the existing rotated mAP protocol after gathering complete images."""

    _prediction_fields = ("rboxes", "scores", "labels")
    _target_fields = ("rboxes", "labels", "difficult")

    def compute(self) -> dict[str, float]:
        """Score first-occurrence images with the unchanged rotated evaluator."""
        return evaluate_rotated_map(*self._images())


class _KeypointAP(_ValidationMetric):
    """Compute the existing OKS protocol after gathering complete images."""

    _prediction_fields = ("keypoints", "scores", "labels")
    _target_fields = ("keypoints", "visibility", "labels")

    def __init__(self, sigmas: Sequence[float], *, dataset_order: bool = False) -> None:
        """Retain the same sigma vector used by the module's OKS loss."""
        super().__init__(dataset_order=dataset_order)
        self._sigmas = tuple(sigmas)

    def compute(self) -> dict[str, float]:
        """Score first-occurrence images with the unchanged keypoint evaluator."""
        return evaluate_keypoints(*self._images(), sigmas=self._sigmas)
