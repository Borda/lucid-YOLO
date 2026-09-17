# SPDX-License-Identifier: Apache-2.0
"""Object-Keypoint-Similarity loss for keypoint regression (R1 sec. 3.4.2, R12; WP-178).

R1 sec. 3.4.2 ("Pose Estimation") says of YOLO26's pose training that it "uses an
Object Keypoint Similarity (OKS)-based loss [48], which normalizes keypoint
localization error by person scale and per-keypoint OKS constants", and that
"YOLO26 extends this scheme with Residual Log-Likelihood Estimation (RLE)". So the
shipped objective is **two** point terms side by side -- an OKS-based term and the
RLE term :class:`~lucid_yolo.losses.rle_loss.RLELoss` already supplies -- and R1
Table 9 ablates their weights ``(w_OKS, w_RLE)`` on COCO keypoints val (YOLO26s):
``(48, 0)`` 61.5, ``(48, 1)`` 60.8, ``(24, 1)`` 63.0, ``(24, 2)`` 62.5, ``(12, 1)``
62.6, ``(0, 1)`` 61.9, ``(0, 2)`` 62.4 mAP.

**R1 gives no formula for the OKS loss.** The sentence above is the whole of what
it states, and its citation [48] is not on this project's source allowlist, so it
was neither read nor is it cited here as a source. What *is* allowlisted is R12,
which defines the OKS *metric* the loss is named after::

    OKS = sum_i exp(-d_i^2 / (2 s^2 k_i^2)) [v_i > 0]  /  sum_i [v_i > 0]

with ``d_i`` the Euclidean distance between predicted and annotated point ``i``,
``s^2`` the object scale, ``k_i = 2 sigma_i`` from the per-keypoint sigma table,
and ``v_i`` COCO visibility. This module therefore takes the one form that is
derivable from an allowlisted definition alone: **``1 - OKS`` per instance, meaned
over the instances that have at least one labelled point.** ``OKS`` is bounded in
``(0, 1]``, is ``1`` at a perfect prediction, and decays with a per-point Gaussian
whose width is the point's own tolerance -- so ``1 - OKS`` is a loss that is zero
at the optimum, saturates at ``1`` rather than growing without bound, and is
already normalized by object scale and per-keypoint constant in exactly the sense
R1's sentence describes. That the published loss is this form and not, say, a
log-OKS or a per-point ``1 - exp(...)`` sum is **not** established by any source
this project may read; it is the recorded assumption A75.

**Scale and constants (A75).** R12 evaluates ``s^2`` as the annotated segment area.
Detection targets carry a box and no segment, so this loss takes ``s^2`` as the
assigned ground-truth box's area ``w * h``, floored at ``1.0`` square pixel -- the
same floor :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box` puts on a
per-axis box extent (A36's box-area precedent): a degenerate box carries no scale
to normalize by, and the alternative is a division whose answer to a finite
annotation is an infinity. ``k_i`` is ``2 sigma_i`` over the sigma table the
caller supplies, which the training module picks by the rule it already uses for
``val/oks_mAP`` (:data:`~lucid_yolo.eval.coco_eval.COCO_KEYPOINT_OKS_SIGMAS` for
COCO's 17-point schema, A67's uniform sigma for any other count). The sigmas are
passed **in** rather than imported: :mod:`lucid_yolo.losses` sits below
:mod:`lucid_yolo.eval` in the layering (``eval`` imports the models and the
metrics backend), and a loss that reached up into the evaluator would invert it.

**Coordinates are input pixels, not the A71 frame.** The RLE term needs a
box-normalized frame because its ``sigma_hat`` is sigmoid-bounded into ``(0, 1)``;
OKS carries its own normalization in ``s^2`` and is defined on pixel distances, so
feeding it normalized coordinates would divide by the box scale twice and turn a
metric-shaped loss into one nothing publishes. The training module passes this
loss the decoded pixels it passes the RLE term *before* normalizing them, and the
box area from the same assigned frame.

**Visibility (A66).** ``v = 0`` points are excluded before any arithmetic touches
them, as :class:`~lucid_yolo.losses.rle_loss.RLELoss` does and for its reason: an
unlabelled point's coordinate is not ground truth and may be anything, and a
distance computed on it and multiplied by zero afterwards still puts ``0 * inf``
into the gradient of every predicted coordinate that shares the tensor. Here the
labelled points are boolean-selected out of the flattened ``(N * K)`` rows, scored,
and scattered back into a per-instance sum; a ``v = 0`` row is not part of the
graph. An instance with **no** labelled point has no OKS -- R12's denominator is
zero -- and is left out of the mean rather than scored as ``1 - 0/0``; a batch
with no such instance at all, including ``N = 0``, returns a finite zero that is
still attached to ``mu_hat``'s graph, mirroring the RLE term's empty case.

**No parameters, no persistent state.** ``k`` is held as a **non-persistent**
buffer: it moves with the module's device and dtype, but it is not in
``state_dict()``, because a keypoints module's checkpoint keys are pinned to "the
detection keys plus the point stems plus ``rle_loss.``" and every pre-WP-178 pose
checkpoint would stop loading strictly if a new key appeared. ``.parameters()`` is
empty and construction draws no RNG, so a module built with this term has
bit-for-bit the weights of one built without it.

**Why the default gain is zero.** R1 Table 9's best row is ``(24, 1)``, but a
weight measured on YOLO26s on COCO under a loss whose formula this project had to
derive is not a measured value *here*, and shipping it as a default would make an
untested number look like evidence (the argument A68 makes for ``keypoint_gain``).
``oks_gain=0.0`` keeps the shipped RLE-only objective bit-exact; the ``(24, 1)``
pair is the published target a dose-response run would test against.

No detection-repository code of any kind was consulted while writing this module
-- it is R12's metric definition, R1's stated composition, and this repository's
own :class:`~lucid_yolo.losses.rle_loss.RLELoss` conventions, per AGENTS.md sec. 6
and sec. 7.

Provenance: R1 sec. 3.4.2, Table 9 (composition and weights, no formula); R12 (OKS
definition, sigma table, visibility semantics).
Assumptions: A75 (loss form, box area as s^2, k_i = 2 sigma_i, zero default gain),
A66 (visibility mask policy), A67 (uniform sigma off-COCO).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OKSLoss"]

#: Floor, in square input pixels, on the object scale ``s^2`` the per-point distance is
#: divided by. One square pixel, mirroring the one-pixel per-axis floor
#: :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box` applies (A36's precedent):
#: a box thinner than that carries no scale, and the unclamped division answers a
#: finite annotation with an infinity.
_MIN_AREA = 1.0

#: R12's ``k_i = 2 sigma_i``: the per-keypoint constant is twice the tabulated sigma.
_K_PER_SIGMA = 2.0


class OKSLoss(nn.Module):
    """``1 - OKS`` over keypoint predictions, meaned per labelled instance (R12, A75).

    Args:
        sigmas: Per-keypoint OKS sigmas, length ``K`` -- R12's table for COCO's
            17-point human schema, or A67's uniform value repeated for any other
            count. Stored as ``k = 2 * sigma`` in a non-persistent buffer.

    Raises:
        ValueError: If ``sigmas`` is empty or holds a non-positive value -- a
            zero sigma would make every point's tolerance zero and the loss
            constant ``1`` everywhere but the exact optimum.

    Examples:
        >>> import torch
        >>> loss_fn = OKSLoss(sigmas=(0.1, 0.2))
        >>> list(loss_fn.parameters()), loss_fn.state_dict()  # nothing to train, nothing to checkpoint
        ([], OrderedDict())
        >>> mu_hat = torch.zeros(1, 2, 2, requires_grad=True)
        >>> mu_gt = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]])
        >>> loss = loss_fn(mu_hat, mu_gt, area=torch.tensor([100.0]), visibility=torch.tensor([[2, 1]]))
        >>> loss.ndim, bool(loss.requires_grad)
        (0, True)
        >>> loss_fn(mu_gt, mu_gt, torch.tensor([100.0]), torch.tensor([[2, 1]]))  # a perfect prediction
        tensor(0.)
    """

    k: Tensor

    def __init__(self, sigmas: Sequence[float]) -> None:
        super().__init__()
        if not len(sigmas) or any(sigma <= 0 for sigma in sigmas):
            raise ValueError(f"sigmas must be a non-empty sequence of positive values, got {tuple(sigmas)!r}")
        self.register_buffer("k", _K_PER_SIGMA * torch.tensor(tuple(sigmas), dtype=torch.float32), persistent=False)

    def forward(self, mu_hat: Tensor, mu_gt: Tensor, area: Tensor, visibility: Tensor) -> Tensor:
        """Compute the mean ``1 - OKS`` over the instances with at least one labelled point.

        Args:
            mu_hat: Predicted point coordinates, shape ``(N, K, 2)``, **in input
                pixels** -- the decoded output of
                :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` for the
                positive instances an assigner has already selected, *not* mapped
                through :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box`:
                OKS normalizes by ``area`` itself (module docstring).
            mu_gt: Ground-truth point coordinates, shape ``(N, K, 2)``, in the same
                pixel frame.
            area: Object scale ``s^2`` per instance, shape ``(N,)`` -- the assigned
                ground-truth box's ``w * h`` in square pixels (A75), floored here at
                one square pixel.
            visibility: COCO-style visibility, shape ``(N, K)`` int64 (WP-121's
                ``Targets.keypoint_vis``). Points with ``v == 0`` do not contribute
                (A66); ``v >= 1`` do. The exclusion is applied **before** any
                arithmetic, so an unlabelled coordinate is never subtracted or
                squared and cannot reach a gradient.

        Returns:
            A scalar (zero-dimensional) tensor: the mean of ``1 - OKS`` over the
            instances with ``>= 1`` labelled point. A finite zero, still attached
            to ``mu_hat``'s graph, when there is no such instance (``N = 0``
            included).

        Raises:
            ValueError: If ``mu_hat``'s point count differs from the sigma table's
                ``K`` -- a head predicting 17 points scored against a 3-sigma table
                is a misconfiguration no broadcast should paper over.
        """
        instances, num_points = int(visibility.shape[0]), int(visibility.shape[1])
        if num_points != self.k.numel():
            raise ValueError(f"the sigma table has {self.k.numel()} keypoints but the inputs carry {num_points}")
        labelled = (visibility > 0).reshape(-1)  # (N * K,)
        rows = torch.arange(instances, device=visibility.device).repeat_interleave(num_points)[labelled]  # (V,)
        k = self.k.to(mu_hat.dtype).repeat(instances)[labelled]  # (V,)
        scale = area.clamp(min=_MIN_AREA).repeat_interleave(num_points)[labelled]  # (V,) s^2 per point
        offset = mu_gt.reshape(-1, 2)[labelled] - mu_hat.reshape(-1, 2)[labelled]  # (V, 2)
        similarity = torch.exp(-offset.square().sum(dim=-1) / (2.0 * scale * k.square()))  # (V,)

        per_instance = mu_hat.new_zeros(instances).index_add(0, rows, similarity)  # (N,) sum_i exp(...)
        counts = (visibility > 0).sum(dim=1)  # (N,) sum_i [v_i > 0]
        scored = counts > 0
        oks = per_instance[scored] / counts[scored].to(per_instance.dtype)
        return (1.0 - oks).sum() / max(oks.numel(), 1)
