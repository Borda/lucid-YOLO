# SPDX-License-Identifier: Apache-2.0
"""Per-branch oriented box terms for the OBB training path (WP-088, A49-A51).

The rotated counterpart of :mod:`lucid_yolo.losses.detection_loss`'s two box terms.
It scores one detection branch's *assembled* oriented predictions (A44) against the
rotated ground truths the **same** branch's assignment already selected, and returns
the three pre-gain terms the oriented objective is composed from. The caller —
:class:`~lucid_yolo.ptl.module.DetectionLitModule` — owns the gains, exactly as it
owns them for the mask terms of WP-087 and as
:func:`~lucid_yolo.losses.angle_loss.square_angle_loss` already documents for A22.

What ``task="obb"`` does with each term of the dual detection objective
    The oriented objective is not the detection objective plus something. Term by
    term, against :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss`:

    ==================  ==========  =======================================================
    term                fate        why
    ==================  ==========  =======================================================
    ``L_cls``           kept        TAL-weighted BCE over the same assignment at the same
                                    gain. Orientation changes what a box *is*, not what a
                                    class score means.
    ``L_box`` (CIoU)    replaced    :func:`oriented_branch_terms`'s ``rbox``. The axis
                                    -aligned Complete-IoU of the decoded rectangle is blind
                                    to ``theta``, so keeping it beside a rotated term would
                                    supervise the same four ltrb distances twice, once
                                    towards a target the other one is turning away from.
    ``L_l1``            replaced    ``rl1`` below — the same stride-normalized L1, retargeted
                                    onto the rotated box's own ``(cx, cy, w, h)``.
    ``L_angle``         added       R1 Eq. 15, :func:`~lucid_yolo.losses.angle_loss.square_angle_loss`.
    assignment          kept        One assignment, reused (below). Candidacy alone becomes
                                    rotated (A25, WP-061).
    ==================  ==========  =======================================================

Why the L1 term is retargeted rather than dropped (A50)
    R1 does not say whether the L1 term survives beside a rotated IoU term. Keeping it
    **as it stands** is not an option: its target is ``assign.target_boxes``, the
    axis-aligned *envelope* of the ground truth, while A44's composition reads the
    decoded ``xyxy`` as the oriented box's own extents before rotating it about its
    centre. For a 40x10 box at 45 degrees the envelope is roughly 35x35, so the two
    terms would pull the same ``w``/``h`` towards targets that disagree by the whole
    of the rotation. Measured in this frame the conflict is not subtle: the envelope
    and the rotated box agree only at ``theta = 0``.

    Dropping it instead would leave the extents supervised by ProbIoU alone, and
    ProbIoU floors a collapsed side at ``min_side`` with **zero** gradient there
    (A41) — precisely the state an untrained head starts in, since the raw ltrb
    distances routinely decode to ``r < -l``. The L1 term is what pulls a collapsed
    extent back out. So it is kept and its target moved to the rotated box's own
    ``(cx, cy, w, h)``, which is the quantity the ltrb distances actually decode to.
    ``theta`` is deliberately **not** an L1 column: it is not a stride-unit quantity,
    and R1 Eq. 15 already owns the angular residual.

One assignment, never two (WP-087's rule)
    Every per-positive quantity here — the predicted rotated box, the target rotated
    box, the angle residual and the weight — is gathered from the ``AssignResult``
    the box terms of this branch were scored against: the anchor rows by ``fg_mask``,
    the instances by ``gt_index``. Re-assigning at this call site would be a second
    selection path, free to pair one anchor's orientation with another instance's
    box — a defect no loss value reveals.

Which ProbIoU (A49)
    WP-059 exposes both losses R17 proposes and leaves the choice here. The default
    is the bounded Hellinger form ``L1 = H_D = 1 - ProbIoU``:

    * it occupies the slot the Complete-IoU loss vacated and shares its ``[0, 1]``
      range, so A13's ``box_gain = 7.5`` transfers without inventing a magnitude —
      which the unbounded ``B_D``, whose scale depends on how far apart the boxes are,
      would require;
    * R17's argument for starting on ``B_D`` is its non-vanishing far-field gradient,
      and the far field is largely unreachable here: an anchor is a candidate only if
      its centre lies inside the rotated ground truth (A25), and a positive's weight
      is ``s**alpha * u**beta`` normalized, which is already near zero for a badly
      placed box;
    * R17's "start on ``L2``, switch to ``L1``" schedule would add a switch epoch and
      a second gain, neither of which the paper states.

    ``"bhattacharyya"`` selects the unbounded form; the choice is a constructor
    argument rather than a hard-coded call so the alternative is one config key away.

Normalization
    Both weighted terms divide by ``assign.align_weights.sum()`` floored at one — the
    same normalizer :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss`
    uses, so an oriented term and a detection term of the same magnitude mean the same
    thing. R1 Eq. 15's ``S`` is that identical sum, because ``align_weights`` is
    exactly zero at every background anchor.

    The floor is R1's own rule and it is not neutral. Once the weights total less than
    one — few positives, or positives whose alignment is uniformly small — the divisor
    stops tracking them and each term becomes a weighted **sum** rather than a weighted
    mean, so such a batch contributes proportionally less than a well-aligned one
    instead of the same. Above unit total weight the two readings coincide. The same
    sentence applies verbatim to the two detection terms and to
    :func:`~lucid_yolo.losses.angle_loss.square_angle_loss`, which is the point: the
    three normalizers agree, including in where they stop being means.

Provenance: R1 sec. 3.4.3, R1 Eq. 13-15, R17. Assumptions: A22, A25, A41, A44, A49, A50.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lucid_yolo.losses.angle_loss import square_angle_loss
from lucid_yolo.losses.probiou import probiou_bhattacharyya_loss, probiou_hellinger_loss

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lucid_yolo.assign.tal import AssignResult

__all__ = ["DEFAULT_ROTATED_IOU_FORM", "ROTATED_IOU_FORMS", "OrientedLossOutput", "oriented_branch_terms"]

#: The two rotated-IoU losses R17 proposes, by name (A49). ``"hellinger"`` is the
#: bounded ``L1 = H_D = 1 - ProbIoU``; ``"bhattacharyya"`` the unbounded ``L2 = B_D``.
ROTATED_IOU_FORMS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "hellinger": probiou_hellinger_loss,
    "bhattacharyya": probiou_bhattacharyya_loss,
}

#: The form the oriented path uses unless a config says otherwise (A49; module docstring).
DEFAULT_ROTATED_IOU_FORM = "hellinger"

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Columns of a rotated box the L1 term measures: ``(cx, cy, w, h)``, angle excluded.
_L1_COLUMNS = 4


@dataclass(frozen=True)
class OrientedLossOutput:
    """The three **pre-gain** oriented terms of one detection branch.

    Follows :class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput`'s
    convention of reporting raw objectives and leaving the gains to the caller,
    with no ``total`` field: the caller weights all three *and* blends the two
    branches by the WP-035 ``alpha`` in one place, so a per-branch total here
    would be a number nothing consumes.

    Attributes:
        rbox: Alignment-weighted rotated-IoU term over the branch's positives.
        rl1: Alignment-weighted stride-normalized L1 on ``(cx, cy, w, h)``.
        angle: R1 Eq. 15's square-object angle term over the same positives.
    """

    rbox: Tensor
    rl1: Tensor
    angle: Tensor


def oriented_branch_terms(
    pred_rboxes: Tensor,
    pred_theta: Tensor,
    gt_rboxes: Tensor,
    assign: AssignResult,
    strides: Tensor,
    form: str = DEFAULT_ROTATED_IOU_FORM,
) -> OrientedLossOutput:
    """Score one branch's oriented predictions against its own assignment.

    Args:
        pred_rboxes: ``(B, A, 5)`` canonical rotated boxes, as
            :func:`~lucid_yolo.models.heads.obb.decode_rboxes` assembles them from
            this branch's ltrb distances and angles (A44). Passing the decoded boxes
            rather than re-deriving them here is what keeps the trained composition
            and the deployed one the same composition.
        pred_theta: ``(B, A)`` **raw** orientation of R1 Eq. 13 (``theta_hat = z``),
            the angle residual of Eq. 14 is measured on. The canonical ``theta`` in
            ``pred_rboxes`` differs from it by whole multiples of ``pi/2``, which
            ``sin^2(2 x)`` cannot see, so the two agree on the loss and the raw value
            is the one the paper writes.
        gt_rboxes: ``(B, N, 5)`` padded ground-truth rotated boxes, entry ``(b, n)``
            describing the same instance as the ``gt_boxes[b, n]`` the assignment ran
            on. ``N = 0`` (a batch with no ground truth) yields three exact zeros that
            still carry gradient.
        assign: The :class:`~lucid_yolo.assign.tal.AssignResult` this branch's box
            terms were scored against — never a fresh assignment (module docstring).
        strides: ``(A,)`` per-anchor level stride, so the L1 term is measured in the
            head's native stride units (A13 revision, WP-078).
        form: Which rotated-IoU loss to use, a key of :data:`ROTATED_IOU_FORMS`
            (A49). Defaults to :data:`DEFAULT_ROTATED_IOU_FORM`.

    Returns:
        The pre-gain :class:`OrientedLossOutput`.

    Raises:
        ValueError: If ``form`` is not a known rotated-IoU form.

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign.tal import AssignResult
        >>> assign = AssignResult(
        ...     fg_mask=torch.tensor([[True, False]]),
        ...     gt_index=torch.tensor([[0, -1]]),
        ...     target_labels=torch.tensor([[0, -1]]),
        ...     target_boxes=torch.zeros(1, 2, 4),
        ...     align_weights=torch.tensor([[1.0, 0.0]]),
        ... )
        >>> box = torch.tensor([[4.0, 4.0, 8.0, 2.0, 0.25]])
        >>> pred = box.expand(1, 2, 5)
        >>> terms = oriented_branch_terms(
        ...     pred, torch.tensor([[0.25, 0.0]]), box.unsqueeze(0), assign, torch.tensor([8.0, 8.0])
        ... )
        >>> float(terms.rbox), float(terms.rl1), float(terms.angle)  # an exact match scores zero
        (0.0, 0.0, 0.0)
        >>> empty = torch.zeros(1, 0, 5)
        >>> zeroed = oriented_branch_terms(
        ...     pred, torch.tensor([[0.25, 0.0]]), empty, assign, torch.tensor([8.0, 8.0])
        ... )
        >>> float(zeroed.rbox), float(zeroed.angle)  # no ground truth: finite, not NaN
        (0.0, 0.0)
    """
    rotated_iou_loss = _resolve_form(form)
    if gt_rboxes.shape[1] == 0:
        return _no_ground_truth(pred_rboxes, pred_theta)

    fg_mask = assign.fg_mask
    weight_sum = assign.align_weights.sum().clamp(min=1.0)
    instance_index = assign.gt_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, _RBOX_DIM)
    target_pos = gt_rboxes.gather(1, instance_index)[fg_mask]  # (P, 5)
    pred_pos = pred_rboxes[fg_mask]  # (P, 5)
    weights = assign.align_weights[fg_mask]  # (P,)

    rbox_terms = rotated_iou_loss(pred_pos, target_pos)  # (P,)
    stride_pos = strides.expand(fg_mask.shape)[fg_mask].unsqueeze(-1)  # (P, 1)
    l1_terms = ((pred_pos[:, :_L1_COLUMNS] - target_pos[:, :_L1_COLUMNS]) / stride_pos).abs().sum(dim=-1)  # (P,)
    return OrientedLossOutput(
        rbox=(rbox_terms * weights).sum() / weight_sum,
        rl1=(l1_terms * weights).sum() / weight_sum,
        angle=square_angle_loss(pred_theta[fg_mask], target_pos[:, 4], target_pos[:, 2], target_pos[:, 3], weights),
    )


def _no_ground_truth(pred_rboxes: Tensor, pred_theta: Tensor) -> OrientedLossOutput:
    """Return three exact zeros that still carry gradient to the predictions.

    A batch whose every image is empty pads to ``N = 0``, which no ``gather`` can
    index. Summing an empty *slice* of each prediction gives the same exact zero the
    all-background path produces while keeping both tensors in the graph, so the step
    needs no special case downstream.

    Examples:
        >>> import torch
        >>> terms = _no_ground_truth(torch.zeros(1, 3, 5), torch.zeros(1, 3))
        >>> float(terms.rbox), float(terms.rl1), float(terms.angle)
        (0.0, 0.0, 0.0)
    """
    box_zero = pred_rboxes.reshape(-1, _RBOX_DIM)[:0].sum()
    return OrientedLossOutput(rbox=box_zero, rl1=box_zero, angle=pred_theta.reshape(-1)[:0].sum())


def _resolve_form(form: str) -> Callable[[Tensor, Tensor], Tensor]:
    """Look up a rotated-IoU loss by name, naming the alternatives when it is unknown.

    Examples:
        >>> _resolve_form("hellinger").__name__
        'probiou_hellinger_loss'
    """
    loss = ROTATED_IOU_FORMS.get(form)
    if loss is None:
        raise ValueError(f"unknown rotated-IoU form {form!r}; expected one of {sorted(ROTATED_IOU_FORMS)}")
    return loss
