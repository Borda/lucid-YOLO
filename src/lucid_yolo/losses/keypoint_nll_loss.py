# SPDX-License-Identifier: Apache-2.0
"""Laplace negative-log-likelihood keypoint loss -- R14's own flow-free ablation (WP-135).

This is R14 Table 7's **"Laplace, learnable variance"** row, built as the
comparison point WP-125's acceptance needs. That tier does not gate on an
absolute pose number; it gates on RLE's *mechanism* claim -- that the learned
residual density is what buys the improvement, not the reparameterization or
the learned scale alone. A claim of that shape cannot be settled by one run,
because a single mAP figure is consistent with the flow doing everything, with
it doing nothing, and with the two terms cancelling. It needs a second run
whose only difference is the mechanism under test.

R14 ran exactly that ablation and published it: on COCO, "Laplace, learnable
variance" scores **67.4 AP** against full RLE's **70.5 AP** (Table 7). So the
direction of effect is already known from the literature, and this module is
not proposing a new modeling choice -- it reproduces a published control so
this project's own runs can be checked against a figure someone else measured.
That is why no new assumption id accompanies it: nothing here is an open
project decision.

**The formula is :class:`~lucid_yolo.losses.rle_loss.RLELoss`'s with one term
removed.** R14 Eq. 8 (with the ``log s`` normalization R14's own implementation
drops) reads::

    L_rle = -log Q(x_bar) - log G_phi(x_bar) + log sigma_hat

over the standardized residual ``x_bar = (mu_g - mu_hat) / sigma_hat``, where
``Q`` is the fixed unit Laplace and ``G_phi`` the RealNVP flow's learned
correction. Dropping ``G_phi`` -- and nothing else -- leaves::

    L_nll = -log Q(x_bar) + log sigma_hat

which is a plain Laplace negative log-likelihood over the prediction error, with
the scale predicted per point and per axis rather than fixed. The residual is
formed the same way, ``sigma_hat`` comes through the same sigmoid (R14 sec. 3.3,
A65), the ``log sigma_hat`` term is the same reparameterization log-Jacobian,
the sum is over the same two coordinate axes, and the masking and reduction are
the same. Holding all of that fixed is the point: any difference a paired run
shows is attributable to the flow, because the flow is the only thing that
differs.

**"Learnable variance" names what is kept, not what is dropped.** ``sigma_hat``
is still a head output and is still trained -- the ``log sigma_hat`` term is
what gives the likelihood a reason to prefer a small scale where the model is
accurate. What the ablation removes is the learned *shape* of the residual
density: ``L_nll`` assumes the error is Laplace-distributed and only fits how
wide, while RLE fits how wide **and** what shape. The degenerate baseline in
which the scale is fixed too is not this row and is not built here; R14 Table 7
lists it separately, and it is just L1.

**No parameters of its own.** This is the one structural difference from
:class:`~lucid_yolo.losses.rle_loss.RLELoss`, whose flow carries real weights an
optimizer has to pick up. Here there is nothing to train beside the network:
``.parameters()`` is empty, ``state_dict()`` is empty, and construction draws no
RNG at all -- so an ablation module's weights are bit-for-bit a same-seed
detection module's, where a RLE module's diverge from the point the flow's
``Linear`` layers are initialized. It is still an :class:`~torch.nn.Module`
rather than a function, because
:class:`~lucid_yolo.ptl.module.DetectionLitModule` holds whichever loss the run
selected in one attribute and calls it through one call site; a module keeps the
substitution to the constructor, where it belongs, instead of spreading a branch
through the step.

**The frame is still the assigned box's (A71).** ``sigma_hat`` is sigmoid-bounded
into ``(0, 1)`` here exactly as in RLE, so a residual it can meaningfully scale is
still one whose errors are ``O(1)``, and callers still map both point arguments
through :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box` first. The
*consequence* of ignoring that differs, though, and this module claims only what
is true of itself: the unit Laplace's log-density is linear in ``|x_bar|``, so an
unnormalized residual costs a large finite number here, where in RLE it meets a
flow whose latent is exponential in a compounding log-scale and whose base
density is quadratic in that latent (A72's non-finite run). Milder, not benign --
a loss dominated by a handful of far-out points is still a loss training the
wrong thing.

**The shared helper is duplicated rather than extracted.** ``_reduce_masked``
and the two constants below also exist in
:mod:`~lucid_yolo.losses.rle_loss`. Extracting them into a shared private module
would put a refactor of a **landed, accepted** loss inside an ablation's commit,
for six lines and two literals; this project's own rule is that three similar
lines beat a premature shared abstraction (``CLAUDE.md``, Core Principles). The
duplication also keeps the two losses free to diverge -- a control that silently
follows every later change to the thing it controls for is not a control.

No detection-repository code of any kind was consulted while writing this module
-- it is R14's own published table plus this repository's existing
:class:`~lucid_yolo.losses.rle_loss.RLELoss`, per AGENTS.md sec. 6 and sec. 7.

Provenance: R14 Table 7, sec. 3.2, sec. 3.3 (sigma_hat sigmoid, visibility masking shared with RLELoss).
Assumptions: A65 (sigmoid), A66 (visibility mask policy), A71 (box-normalized frame).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Laplace

__all__ = ["LaplaceNLLLoss"]

#: Scale of the fixed base density Q (a standardized unit Laplace; sigma_hat
#: already carries the residual's scale, so Q itself needs none). Duplicated from
#: :mod:`~lucid_yolo.losses.rle_loss` on purpose -- see the module docstring.
_BASE_LAPLACE_SCALE = 1.0
#: Minimum sigma_hat used only to keep the reparameterization's division finite at
#: the sigmoid's asymptote; sigma_hat in (0, 1) never reaches this in practice, and
#: the floor changes no in-range value.
_MIN_SIGMA = 1e-6


def _reduce_masked(per_point: Tensor, mask: Tensor) -> Tensor:
    """Mean a per-point loss over ``mask``, or a finite zero if nothing is masked in.

    Mirrors :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss`'s
    zero-positives handling: a sum divided by ``max(count, 1)`` stays finite and
    attached to ``per_point``'s graph rather than producing a ``NaN`` ``0 / 0``
    mean.

    Args:
        per_point: Per-point loss values, any shape.
        mask: Boolean mask of the same shape; ``True`` entries contribute.

    Returns:
        A scalar (zero-dimensional) tensor.

    Examples:
        >>> import torch
        >>> _reduce_masked(torch.tensor([2.0, 4.0, 6.0]), torch.tensor([True, False, True]))
        tensor(4.)
        >>> _reduce_masked(torch.zeros(3), torch.zeros(3, dtype=torch.bool))
        tensor(0.)
    """
    selected = per_point[mask]
    return selected.sum() / max(selected.numel(), 1)


class LaplaceNLLLoss(nn.Module):
    """Laplace NLL over keypoint predictions with a learned per-axis scale (R14 Table 7).

    The flow-free counterpart of :class:`~lucid_yolo.losses.rle_loss.RLELoss`,
    sharing its call signature exactly so a run selects between them at
    construction and the training step reads one attribute either way
    (``keypoint_loss`` on
    :class:`~lucid_yolo.ptl.module.DetectionLitModule`). Unlike that class this
    one holds **no parameters**, which is the property the ablation turns on: the
    residual density is fixed rather than learned.

    Examples:
        >>> import torch
        >>> loss_fn = LaplaceNLLLoss()
        >>> list(loss_fn.parameters())  # no flow, hence no weights of its own
        []
        >>> mu_hat = torch.zeros(2, 1, 2, requires_grad=True)
        >>> sigma_raw = torch.zeros(2, 1, 2, requires_grad=True)
        >>> mu_gt = torch.tensor([[[0.2, -0.1]], [[0.0, 0.0]]])
        >>> visibility = torch.tensor([[2], [0]])  # second instance's point is unlabeled
        >>> loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        >>> loss.ndim
        0
        >>> bool(loss.requires_grad)
        True
    """

    def forward(self, mu_hat: Tensor, sigma_raw: Tensor, mu_gt: Tensor, visibility: Tensor) -> Tensor:
        """Compute the masked mean Laplace negative log-likelihood.

        Args:
            mu_hat: Predicted point coordinates, shape ``(N, K, 2)`` -- the decoded
                output of
                :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` for the
                positive instances an assigner has already selected, mirroring
                :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss`'s
                already-gathered-positives convention. **In a normalized frame, not
                input pixels** (A71): ``sigma_hat`` is bounded into ``(0, 1)`` by
                R14's own sigmoid, so a residual it can scale is one whose errors
                are ``O(1)``. Callers map both point arguments through
                :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box` first;
                passing pixels yields a large finite loss dominated by the worst
                points rather than a useful one.
            sigma_raw: Raw, unactivated per-axis uncertainty, shape ``(N, K, 2)``
                (:class:`~lucid_yolo.models.heads.detect.DualDetectionHead`'s
                ``keypoint_sigma`` output, WP-122, A65). Passed through
                :func:`torch.sigmoid` here, per R14 sec. 3.3 -- the same activation
                :class:`~lucid_yolo.losses.rle_loss.RLELoss` applies, so the two
                losses read an identical ``sigma_hat`` from identical head weights.
            mu_gt: Ground-truth point coordinates, shape ``(N, K, 2)``, in the same
                normalized frame as ``mu_hat`` (A71).
            visibility: COCO-style visibility, shape ``(N, K)`` int64 (WP-121's
                ``Targets.keypoint_vis``). Points with ``v == 0`` do not contribute
                (A66); ``v >= 1`` do, regardless of occlusion.

        Returns:
            A scalar (zero-dimensional) tensor: the mean per-point, per-axis loss
            over every visible point. A finite zero, still attached to ``mu_hat``
            and ``sigma_raw``'s graphs, when no point is visible.
        """
        sigma_hat = torch.sigmoid(sigma_raw).clamp(min=_MIN_SIGMA)
        residual = (mu_gt - mu_hat) / sigma_hat

        laplace = Laplace(0.0, _BASE_LAPLACE_SCALE)
        base_log_prob = cast("Tensor", laplace.log_prob(residual)).sum(dim=-1)  # type: ignore[no-untyped-call]  # (N, K)

        log_sigma_term = torch.log(sigma_hat).sum(dim=-1)  # (N, K)
        per_point = -base_log_prob + log_sigma_term  # (N, K), R14 Eq. 8 without the flow term

        mask = visibility > 0
        return _reduce_masked(per_point, mask)
