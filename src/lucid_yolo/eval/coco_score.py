# SPDX-License-Identifier: Apache-2.0
"""Score a loaded checkpoint on the COCO protocol its task defines, and return the numbers.

:func:`score_coco` is the Python door to the dual-path acceptance protocol: the same annotation load, letterboxed
batching, single forward pass per batch and E2E-plus-NMS decode that ``lucid-eval`` runs for a ``detect``, ``segment``
or ``keypoints`` checkpoint, handed back as a :class:`CocoScore` rather than printed. The command's two COCO runners,
:func:`lucid_yolo.eval.detect_eval.run` and :func:`lucid_yolo.eval.pose_eval.run`, are thin wrappers over the same two
private stages, so a caller in a notebook and an operator at a terminal cannot be scored by different code.

The stages are split where the command needs to speak. :func:`_prepare_coco_run` does everything that can fail before
the expensive part — reads the ground truth, resolves the device, builds the evaluator — and :func:`_score_run` is
the expensive part. A runner prints its banner between the two, so an operator learns how many images and which device
before minutes of scoring rather than after; the library path calls both with nothing in between and prints nothing.

Which protocol applies is the checkpoint's property, the rule ``lucid-eval`` follows: a ``segment`` checkpoint scores
masks unless ``masks=False``, a ``keypoints`` checkpoint is scored against COCO's person-keypoints file with the OKS
sigma table, and an ``obb`` checkpoint is refused, because rotated mAP over tiles is a different protocol
(:func:`lucid_yolo.eval.rotated_eval.run`).

Provenance: R1 Table 7, R1 sec. 4.4 (dual-path protocol), R1 Table S9 (segmentation protocol), R12 (COCO keypoint
protocol and the OKS sigma table). Assumptions: A9, A10, A37.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

from lucid_yolo.assign.grid import require_grid_side
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import COCO_IMG_SIZE
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import letterboxed_batches, load_eval_annotations
from lucid_yolo.eval.checkpoint import pick_device
from lucid_yolo.eval.coco_eval import COCO_KEYPOINT_OKS_SIGMAS, DualPathEvaluator
from lucid_yolo.validate import require_at_least, require_one_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    import torch

    from lucid_yolo.eval.annotations import EvalImage
    from lucid_yolo.models.build import SegmentOutput
    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = ["CocoScore", "score_coco"]

#: Points in COCO's person schema, and so in the keypoint protocol's ground truth, its sigma table and any checkpoint
#: it can score.
_COCO_PERSON_POINTS = len(COCO_KEYPOINT_OKS_SIGMAS)

#: Checkpoint tasks the COCO protocol scores. ``obb`` is absent on purpose: rotated mAP over tiles merged back onto
#: source images is :mod:`lucid_yolo.eval.rotated_eval`.
_COCO_TASKS = ("detect", "segment", "keypoints")

#: Images per forward pass when the caller does not choose; ``lucid-eval``'s own default for every COCO task.
_DEFAULT_BATCH_SIZE = 32

#: Annotation file under ``<data_root>/annotations/`` per task. A keypoint checkpoint is one category, and scored
#: against the 80-category instances file its mAP would be divided by eighty (see :mod:`lucid_yolo.eval.pose_eval`).
_ANNOTATION_FILES = {
    "detect": "instances_val2017.json",
    "segment": "instances_val2017.json",
    "keypoints": "person_keypoints_val2017.json",
}

#: Image directory under ``data_root`` shared by every COCO task.
_IMAGES_DIR = "val2017"


@dataclass(frozen=True)
class CocoScore:
    """The dual-path COCO report of one checkpoint, as numbers rather than printed text.

    Attributes:
        report: ``{"e2e": stats, "nms": stats}``. Each ``stats`` holds the twelve COCO box statistics under their bare
            names (``map`` is mAP50-95, then ``map_50``, ``map_75`` and the size and recall breakdowns), plus the
            twelve ``segm_`` mask statistics for a segmentation checkpoint scored with masks, or the ten ``oks_``
            keypoint statistics for a keypoint checkpoint.
        images: How many images were scored, after any ``limit``.
        seconds: Wall time of the scoring pass, excluding the annotation load.

    Examples:
        ```pycon
        >>> score = CocoScore(report={"e2e": {"map": 0.31}, "nms": {"map": 0.32}}, images=5000, seconds=90.0)
        >>> round(score.report["nms"]["map"] - score.report["e2e"]["map"], 2)
        0.01

        ```
    """

    report: dict[str, dict[str, float]]
    images: int
    seconds: float


class _SegmentationForward(nn.Module):
    """Present a segmentation checkpoint under the forward contract the evaluator expects.

    :meth:`DetectionLitModule.forward` returns the detection head's output alone — deliberately, since that is what
    the E2E decode, the epoch metric and the export path all ask for. The evaluator decides to score masks
    structurally, by whether the forward returns a :class:`~lucid_yolo.models.build.SegmentOutput`, so handing it
    the module unwrapped would silently produce a *detection* report from a segmentation checkpoint: no prototypes,
    no predicted masks, twelve keys, no error, and the cost of the ground-truth masks paid for nothing.

    The wrapped module is registered as a child, so ``.to()`` and ``.eval()`` reach it, and the forward delegates
    to the module's own :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_segmentation` rather than
    recomposing the backbone, neck and mask branches here — that composition exists once, on purpose.

    Args:
        module: A checkpoint-loaded module whose ``task`` is ``"segment"``.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.ptl.module import DetectionLitModule
        >>> module = DetectionLitModule(
        ...     depth=0.34, width=0.25, max_channels=1024, num_classes=4, task="segment"
        ... ).eval()
        >>> with torch.no_grad():
        ...     out = _SegmentationForward(module)(torch.zeros(1, 3, 160, 160))
        >>> out.prototypes.shape[1]
        32

        ```
    """

    def __init__(self, module: DetectionLitModule) -> None:
        super().__init__()
        self.module = module

    def forward(self, images: Tensor) -> SegmentOutput:
        """Run the detection and mask branches over ``images``."""
        return self.module.forward_segmentation(images)


@dataclass(frozen=True)
class _CocoRun:
    """Everything a COCO scoring pass needs, built before the pass starts.

    Attributes:
        images: The eval image records to score, already cut to any ``limit``.
        targets: Ground truth keyed by image id, in original image coordinates.
        images_dir: Directory holding the image files the records name.
        letterbox: The validation letterbox every image is fed through.
        batch_size: Images per forward pass.
        device: The resolved device the pass runs on.
        evaluator: The dual-path evaluator wrapping the module.
    """

    images: list[EvalImage]
    targets: Mapping[int, dict[str, Tensor]]
    images_dir: Path
    letterbox: Letterbox
    batch_size: int
    device: torch.device
    evaluator: DualPathEvaluator

    @property
    def num_batches(self) -> int:
        """How many batches :meth:`batches` yields — the progress bar's total."""
        return math.ceil(len(self.images) / self.batch_size)

    def batches(self) -> Iterator[tuple[Tensor, list[int], list[tuple[int, int]]]]:
        """Yield the letterboxed ``(images, image_ids, orig_sizes)`` batches to score."""
        return letterboxed_batches(self.images, self.images_dir, self.letterbox, self.batch_size)


def _require_coco_person_schema(module: DetectionLitModule) -> int:
    """Return a keypoint checkpoint's point count, refusing any ``K`` but COCO's 17.

    Refused rather than scored on an invented sigma table. This protocol's ground truth is COCO's 17-point person
    schema, so a checkpoint predicting some other ``K`` has no correspondence to it at all -- point ``i`` of the
    prediction is not point ``i`` of the annotation, and R12's sigmas measure annotator variance on joints that are
    not the ones being predicted (A67 makes the same argument for the synthetic symbol schema, which is scored by
    its own gate on its own uniform sigma, never here).

    Args:
        module: A ``keypoints`` checkpoint module.

    Returns:
        The checkpoint's ``num_keypoints``.

    Raises:
        ValueError: If that count is not COCO's 17. ``lucid-eval`` prints the message
            after ``FAIL:`` and exits ``1``.

    Examples:
        ```pycon
        >>> from types import SimpleNamespace
        >>> _require_coco_person_schema(SimpleNamespace(hparams={"num_keypoints": 17}))
        17
        >>> _require_coco_person_schema(SimpleNamespace(hparams={"num_keypoints": 5}))
        Traceback (most recent call last):
            ...
        ValueError: this protocol scores COCO's 17-point person schema; checkpoint has K=5

        ```
    """
    num_keypoints = int(module.hparams["num_keypoints"])
    if num_keypoints != _COCO_PERSON_POINTS:
        raise ValueError(
            f"this protocol scores COCO's {_COCO_PERSON_POINTS}-point person schema; checkpoint has K={num_keypoints}"
        )
    return num_keypoints


def _coco_paths(task: str, data_root: Path | None, ann_file: Path | None, images_dir: Path | None) -> tuple[Path, Path]:
    """Resolve the annotation file and image directory, each override winning on its own.

    Args:
        task: The checkpoint's task, choosing the annotation file name.
        data_root: COCO root holding ``val2017/`` and ``annotations/``, or ``None`` when
            both overrides are given.
        ann_file: Annotation JSON override.
        images_dir: Image directory override.

    Returns:
        ``(ann_file, images_dir)``.

    Raises:
        ValueError: If ``data_root`` is ``None`` and either override is missing.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> [p.as_posix() for p in _coco_paths("keypoints", Path("/coco"), None, None)]
        ['/coco/annotations/person_keypoints_val2017.json', '/coco/val2017']

        ```
    """
    if ann_file is not None and images_dir is not None:
        return ann_file, images_dir
    if data_root is None:
        raise ValueError("data_root is required unless both ann_file and images_dir are given")
    resolved_ann = data_root / "annotations" / _ANNOTATION_FILES[task] if ann_file is None else ann_file
    resolved_images = data_root / _IMAGES_DIR if images_dir is None else images_dir
    return resolved_ann, resolved_images


def _prepare_coco_run(
    module: DetectionLitModule,
    *,
    ann_file: Path,
    images_dir: Path,
    img_size: int,
    batch_size: int,
    device_name: str,
    limit: int,
    eval_backend: str,
    masks: bool,
    keypoints: bool,
) -> _CocoRun:
    """Build everything a scoring pass needs, without running it.

    Reads the ground truth (with masks and keypoints as asked), cuts it to ``limit``, resolves the device and builds the
    evaluator. The evaluator always carries COCO's sigma table, which only a keypoint checkpoint's points ever reach;
    a segmentation one with ``masks`` is wrapped so its forward returns the mask branch too.

    Args:
        module: The eval-mode module.
        ann_file: COCO annotation JSON.
        images_dir: Directory holding the images ``ann_file`` names.
        img_size: Letterbox side.
        batch_size: Images per forward pass.
        device_name: Device string, or ``auto``.
        limit: Score only the first N images; ``0`` scores all.
        eval_backend: Box/mask scoring engine, ``"faster_coco_eval"`` or ``"hotcoco"``.
        masks: Whether to score masks; the module is then wrapped in
            :class:`_SegmentationForward`, so only a ``segment`` module may ask.
        keypoints: Whether the ground truth carries keypoints for OKS scoring.

    Returns:
        The prepared run.
    """
    images, targets, label_to_category = load_eval_annotations(ann_file, with_masks=masks, with_keypoints=keypoints)
    if limit:
        images = images[:limit]
    device = pick_device(device_name)
    letterbox = Letterbox(img_size)
    model: nn.Module = _SegmentationForward(module) if masks else module
    evaluator = DualPathEvaluator(
        model,
        TopKDecoder(),
        NMSDecoder(),
        label_to_category,
        letterbox,
        keypoint_sigmas=COCO_KEYPOINT_OKS_SIGMAS,
        backend=eval_backend,
    )
    return _CocoRun(images, targets, images_dir, letterbox, batch_size, device, evaluator)


def _score_run(run: _CocoRun, batches: Iterable[tuple[Tensor, Sequence[int], Sequence[tuple[int, int]]]]) -> CocoScore:
    """Run the scoring pass over ``batches`` and time it.

    Args:
        run: The prepared run.
        batches: ``run.batches()``, or a progress-bar wrapper around it — the bar is the
            command's concern, not this function's.

    Returns:
        The report, the image count and the wall time.
    """
    start = time.perf_counter()
    report = run.evaluator.evaluate(batches, run.targets, run.device)
    return CocoScore(report=report, images=len(run.images), seconds=time.perf_counter() - start)


def _print_timing(score: CocoScore) -> None:
    """Print the runners' shared ``done in`` line."""
    print(f"done in {score.seconds:.1f}s ({score.images / score.seconds:.1f} img/s)")


def _write_report(output: Path, info: dict[str, object], score: CocoScore) -> None:
    """Write the runners' JSON report to ``output``, creating its directory.

    Created rather than required: the scoring pass is the expensive part and this file
    is its only durable form (see the note in ``rotated_eval.run``).

    Args:
        output: Report path.
        info: The loader's provenance dict, as extended by the runner.
        score: The scored run.
    """
    payload = {"info": info, "images": score.images, "seconds": round(score.seconds, 1), "report": score.report}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"report -> {output}")


def score_coco(
    module: DetectionLitModule,
    data_root: Path | None = None,
    *,
    ann_file: Path | None = None,
    images_dir: Path | None = None,
    img_size: int = COCO_IMG_SIZE,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    device: str = "auto",
    limit: int = 0,
    masks: bool = True,
    eval_backend: str = "faster_coco_eval",
) -> CocoScore:
    """Score a checkpoint on COCO val2017 through both decode paths and return the numbers.

    The protocol is the one ``lucid-eval`` runs, chosen by the module's own ``task``: box mAP for ``detect``, box
    plus ``segm_`` mask mAP for ``segment`` (box only with ``masks=False``), box plus ``oks_`` keypoint mAP against
    ``person_keypoints_val2017.json`` for ``keypoints``. Nothing is printed and no file is written.

    Args:
        module: An eval-mode module, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it. Moved to
            ``device`` in place.
        data_root: COCO root holding ``val2017/`` and ``annotations/``. Optional when
            both ``ann_file`` and ``images_dir`` are given.
        ann_file: Annotation JSON. Defaults to ``annotations/instances_val2017.json``
            under ``data_root``, or ``annotations/person_keypoints_val2017.json`` for a
            keypoint checkpoint.
        images_dir: Image directory. Defaults to ``val2017`` under ``data_root``.
        img_size: Letterbox side, a positive multiple of every head stride. Defaults to
            ``640``.
        batch_size: Images per forward pass. Defaults to ``32``.
        device: ``"auto"``, ``"cpu"``, ``"mps"`` or ``"cuda"``.
        limit: Score only the first N images; ``0`` scores all.
        masks: For a segmentation checkpoint, whether to score masks too.
        eval_backend: Box/mask scoring engine, ``"faster_coco_eval"`` or ``"hotcoco"``.
            The ``oks_`` statistics always come from faster_coco_eval.

    Returns:
        The two paths' statistics, the scored image count and the scoring time.

    Raises:
        ValueError: If the module's task is not ``detect``, ``segment`` or ``keypoints``; if a keypoint checkpoint's
            ``K`` is not COCO's 17; if ``limit``, ``batch_size``, ``img_size`` or ``eval_backend`` is outside its
            domain; if ``data_root`` is missing while an override is; or if ``device`` names a backend this machine
            does not have.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> from lucid_yolo.eval.checkpoint import load_eval_module
        >>> module, _ = load_eval_module(Path("det.ckpt"), use_ema=True)  # needs a checkpoint  # doctest: +SKIP
        >>> score = score_coco(module, Path("~/data/coco2017").expanduser())  # needs COCO  # doctest: +SKIP
        >>> score.report["e2e"]["map"]  # doctest: +SKIP
        0.3912

        ```
    """
    task = str(module.task)
    require_one_of("task", task, _COCO_TASKS)
    require_at_least("limit", limit, 0)
    require_at_least("batch_size", batch_size, 1)
    require_grid_side("img_size", img_size)
    require_one_of("eval_backend", eval_backend, ("faster_coco_eval", "hotcoco"))
    if task == "keypoints":
        _require_coco_person_schema(module)
    resolved_ann, resolved_images = _coco_paths(task, data_root, ann_file, images_dir)
    run = _prepare_coco_run(
        module,
        ann_file=resolved_ann,
        images_dir=resolved_images,
        img_size=img_size,
        batch_size=batch_size,
        device_name=device,
        limit=limit,
        eval_backend=eval_backend,
        masks=task == "segment" and masks,
        keypoints=task == "keypoints",
    )
    return _score_run(run, run.batches())
