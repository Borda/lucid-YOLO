# SPDX-License-Identifier: Apache-2.0
"""Residual log-likelihood estimation loss for keypoint regression (R14, WP-123).

R14 replaces a fixed-shape regression loss (L1, equivalent to assuming the
prediction error follows a Laplace distribution of constant scale) with a
**learned** residual distribution: a small RealNVP normalizing flow that
captures whatever shape the true error distribution actually has, while the
regression head still predicts a location ``mu_hat`` and a per-axis scale
``sigma_hat`` exactly as before.

**The reparameterization (R14 sec. 3.2).** Every keypoint's ground-truth
location ``mu_g`` is standardized against the network's own prediction into a
residual ``x_bar = (mu_g - mu_hat) / sigma_hat``. Training maximizes the
likelihood of ``x_bar`` under a residual density that is itself a sum of two
parts (R14 Eq. 7): a fixed, untrained density ``Q`` (Laplace, "the preset
density function") plus a learned correction ``G_phi`` — the density the
RealNVP flow induces over ``x_bar`` via the standard normalizing-flow
change-of-variables identity, using a standard normal latent
(``z_bar ~ N(0, I)``, R14 sec. 3.2: "the flow model f_phi is leveraged to map a
zero-mean initial distribution z_bar ~ N(0, I)"). ``Q`` is *not* trained and
carries no parameters; only ``G_phi`` (the flow) has weights. R14's own
motivation for the split (sec. 3.2) is a training-stability property: the
gradient of ``log Q(x_bar)`` with respect to ``mu_hat``/``sigma_hat`` does not
depend on the flow at all, so early in training — while the flow is still
close to its untrained identity-like state and would otherwise contribute
near-useless gradient — the regression head still receives a clean, direct
signal through ``Q``.

**The loss (R14 Eq. 8, dropping the ``log s`` normalization term R14's own
implementation drops):**

    L_rle = -log Q(x_bar) - log G_phi(x_bar) + log sigma_hat

summed over both coordinate axes of a point (``Q`` is evaluated per axis and
summed; ``G_phi`` already sums both axes internally, since the flow maps
``R^2 -> R^2`` as one bijection rather than two independent 1-D ones; the
``log sigma_hat`` correction is the log-Jacobian of the ``x_bar``
reparameterization itself, one term per axis).

**The flow (R14 sec. 4, Appendix A Eq. 12).** ``K = 6`` stacked affine
coupling layers over the 2-D residual: each layer holds one coordinate fixed
and transforms the other by a scale-and-shift pair produced from the fixed
one (Eq. 12: ``f_k(z_{k-1,0:d}, z_{k-1,d:D}) = (z_{k-1,0:d}, z_{k-1,d:D} *
exp(g_k(z_{k-1,0:d})) + h_k(z_{k-1,0:d}))``, ``D = 2, d = 1``), with the
transformed/fixed coordinate alternating every layer — RealNVP's own
standard construction (Dinh et al., cited by R14), not a detail R14 restates,
since a coupling stack that never alternates could never move one of its two
coordinates at all and would not be a valid bijection over R^2. Each layer's
``(g_k, h_k)`` conditioner is R14's stated default: 3 fully-connected layers
of 64 units, every one followed by a Leaky-ReLU (R14 sec. 4: "Lfc=3 and Nn=64
by default... Each fully-connected layer is followed by a Leaky-RELU"),
mapping the one conditioning scalar to the transformed coordinate's
``(log_scale, shift)`` pair. R14's sentence says *each* fully-connected layer, and
this module reads it literally: the final ``Linear(64, 2)`` carries a Leaky-ReLU
too, so both emitted quantities are pre-squashed — ``shift`` is effectively
non-negative up to the leak, and ``log_scale = scale * tanh(raw)`` inherits the
same asymmetry. **This literal reading is an assumption, not a transcription**
(A74; the module has carried the activation since WP-125, and the row was minted
when the audit asked what backed it). An output projection is not a hidden layer,
and a flow whose job
is to learn the *shape* of a residual density starts structurally biased when its
head is one-sided. R14 states the conditioner's depth and width but neither its
figure nor its appendix distinguishes hidden layers from the output projection, so
the sentence admits both readings and the paper does not settle it. What would
settle it here is the paired same-seed run the register asks for: the pose smoke
recipe trained twice at identical seed, with and without the trailing activation,
compared on the WP-125 acceptance metric. That run is a work package of its own —
a behaviour change to a landed loss whose RLE-versus-Laplace-NLL comparison is the
thing it would move — and is deliberately **not** made here; only the reading is
recorded, so the next reader inherits an open question rather than a silent choice.

The log-scale leaves that conditioner through a
**tanh times a learned scale**, which is RealNVP's own parameterization of
``s`` and is given there as a stability measure ("To compute the scaling
functions s, we use a hyperbolic tangent function multiplied by a learned
scale", Dinh et al. sec. 4); A72 records what a raw linear log-scale cost when
this loss was first composed into a training run.

**sigma_hat (R14 sec. 3.3, resolving A65's deferred question):** "The
deviation sigma_hat_i is predicted with a sigmoid function. Hence we have
sigma_hat_i in (0, 1)." The keypoint head's raw sigma output
(:class:`~lucid_yolo.models.heads.detect.DualDetectionHead`, WP-122) is
therefore passed through :func:`torch.sigmoid` here, in the loss, rather than
in the head — matching A23's precedent of normalizing a raw head output at
the point that consumes it, not at the point that emits it.

**Visibility masking (A66).** COCO's ``v = 0`` means no annotation exists at
all (WP-121); training against it would optimize toward a coordinate that is
not ground truth. ``v = 1``/``v = 2`` (occluded-but-labeled / visible) both
carry a real, human-placed coordinate and both contribute — occlusion is
exactly the case a regression model must learn to infer from context, and
COCO's own OKS evaluation protocol (R12) already scores against ``v >= 1``
ground truth regardless of visibility, so training against the same set the
metric scores against is the consistent reading rather than a narrower one
this project invented.

No detection-repository code of any kind was consulted while writing this
module — implemented directly from R14's equations and its own citation of
RealNVP (Dinh et al.), per AGENTS.md sec. 6 and sec. 7.

Provenance: R14 (Eq. 5, 7, 8, sec. 3.2, sec. 3.3, sec. 4, Appendix A Eq. 12), R12 (visibility semantics).
Assumptions: A65 (resolved: sigmoid), A66 (visibility mask policy), and the
conditioner output-head activation described above (A74).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Laplace, Normal

__all__ = ["RLELoss"]

#: Number of stacked affine coupling layers (R14 sec. 4: "K is set to 6").
_FLOW_COUPLING_LAYERS = 6
#: Conditioner hidden width (R14 sec. 4: "Nn=64 by default").
_CONDITIONER_HIDDEN = 64
#: Scale of the fixed, untrained base density Q (a standardized unit Laplace;
#: sigma_hat already carries the residual's scale, so Q itself needs none).
_BASE_LAPLACE_SCALE = 1.0
#: Minimum sigma_hat used only to keep the reparameterization's division finite
#: at the sigmoid's asymptote; sigma_hat in (0, 1) never reaches this in
#: practice, and the floor changes no in-range value.
_MIN_SIGMA = 1e-6


class _CouplingConditioner(nn.Module):
    """The ``(g_k, h_k)`` pair: 3 FC/Leaky-ReLU layers mapping one scalar to (log_scale, shift).

    R14 sec. 4's stated default conditioner shape, shared by every coupling
    layer's own independent instance (parameters are not tied across layers).

    The trailing ``LeakyReLU`` after the output ``Linear(64, 2)`` is the literal
    reading of R14's "Each fully-connected layer is followed by a Leaky-RELU" and
    is a registered assumption, not a transcription — it pre-squashes both emitted
    quantities. The module docstring carries the argument on both sides and the
    paired same-seed run that would settle it; this note exists so the activation
    is not mistaken for an accident and quietly deleted.

    Examples:
        >>> import torch
        >>> conditioner = _CouplingConditioner()
        >>> log_scale, shift = conditioner(torch.zeros(5, 1))
        >>> log_scale.shape, shift.shape
        (torch.Size([5]), torch.Size([5]))
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, _CONDITIONER_HIDDEN),
            nn.LeakyReLU(),
            nn.Linear(_CONDITIONER_HIDDEN, _CONDITIONER_HIDDEN),
            nn.LeakyReLU(),
            nn.Linear(_CONDITIONER_HIDDEN, 2),
            nn.LeakyReLU(),
        )
        #: RealNVP's "learned scale" multiplying the tanh — see :meth:`forward`. One
        #: scalar per layer, initialized to 1 so the layer starts able to express a
        #: log-scale in ``(-1, 1)`` and grows the range only if the data pay for it.
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        """Map the conditioning scalar to a ``(log_scale, shift)`` pair.

        The log-scale is emitted **through a tanh times a learned scale**, which is
        RealNVP's own parameterization of ``s`` and is stated as a stability measure:
        "To compute the scaling functions s, we use a hyperbolic tangent function
        multiplied by a learned scale" (Dinh et al., sec. 4 — the same construction
        R14 Appendix A Eq. 12 cites rather than restates). A raw linear log-scale is
        not a simplification of that; it is a different layer, and A72 records what it
        cost here — the stack multiplies ``exp(log_scale)`` six times, so an
        unbounded log-scale compounds, and a measured residual of 27 reached the
        latent as ``4e11`` and the loss as ``7e22`` before the run went non-finite.

        Args:
            conditioning: The fixed coordinate, shape ``(..., 1)``.

        Returns:
            A ``(log_scale, shift)`` pair, each shape ``(...,)``. Only the log-scale
            is bounded: the shift is additive, compounds linearly rather than
            multiplicatively, and is not what the paper stabilizes.
        """
        raw = self.net(conditioning)
        return self.scale * torch.tanh(raw[..., 0]), raw[..., 1]


class _AffineCoupling(nn.Module):
    """One RealNVP affine coupling layer over a 2-D residual (R14 Appendix A Eq. 12).

    Holds one coordinate fixed and scales-and-shifts the other, conditioned on
    the fixed one. :meth:`synthesize` is the generative direction
    (``z -> x``, forward); :meth:`invert` is its closed-form inverse
    (``x -> z``), which is what density evaluation needs — RealNVP's defining
    property is that both directions are available without solving anything.

    Args:
        transform_index: Which coordinate (``0`` or ``1``) this layer
            transforms; the other is the conditioning input. Alternating this
            across the stack is what lets a 2-D coupling stack move both
            coordinates overall (standard RealNVP construction).

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> layer = _AffineCoupling(transform_index=1)
        >>> z = torch.randn(3, 2)
        >>> x, log_det = layer.synthesize(z)
        >>> torch.equal(x[..., 0], z[..., 0])  # the untransformed coordinate rides through unchanged
        True
        >>> z_back, inv_log_det = layer.invert(x)
        >>> torch.allclose(z_back, z, atol=1e-5)
        True
        >>> torch.allclose(log_det, -inv_log_det)  # synthesis and its inverse have opposite log-determinants
        True
    """

    def __init__(self, transform_index: int) -> None:
        super().__init__()
        if transform_index not in (0, 1):
            raise ValueError(f"transform_index must be 0 or 1; got {transform_index}")
        self.transform_index = transform_index
        self.condition_index = 1 - transform_index
        self.conditioner = _CouplingConditioner()

    def synthesize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the layer in the generative direction ``z -> x``.

        Args:
            z: Input of shape ``(..., 2)``.

        Returns:
            ``(x, log_det)``: the transformed pair, shape ``(..., 2)``, and
            this layer's forward log-determinant contribution, shape ``(...,)``.
        """
        fixed = z[..., self.condition_index]
        moving = z[..., self.transform_index]
        log_scale, shift = self.conditioner(fixed.unsqueeze(-1))
        transformed = moving * log_scale.exp() + shift
        x = (
            torch.stack((fixed, transformed), dim=-1)
            if self.condition_index == 0
            else torch.stack((transformed, fixed), dim=-1)
        )
        return x, log_scale

    def invert(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the layer's closed-form inverse ``x -> z``.

        Args:
            x: Input of shape ``(..., 2)``.

        Returns:
            ``(z, log_det)``: the recovered pair, shape ``(..., 2)``, and this
            layer's inverse log-determinant contribution, shape ``(...,)``
            (the negation of :meth:`synthesize`'s, since inverting an
            ``exp(log_scale)`` scaling divides rather than multiplies).
        """
        fixed = x[..., self.condition_index]
        moving = x[..., self.transform_index]
        log_scale, shift = self.conditioner(fixed.unsqueeze(-1))
        original = (moving - shift) * (-log_scale).exp()
        z = (
            torch.stack((fixed, original), dim=-1)
            if self.condition_index == 0
            else torch.stack((original, fixed), dim=-1)
        )
        return z, -log_scale


class _RealNVPFlow(nn.Module):
    """The stacked ``K``-layer RealNVP flow with a standard-normal latent base.

    :meth:`log_density` is :math:`\\log G_\\phi(\\bar{x})`: the flow-induced
    density of the standardized residual, via the change-of-variables
    identity ``log G_phi(x) = log p_z(f_phi^{-1}(x)) + log|det d f_phi^{-1} /
    dx|`` (R14 sec. 3.2), computed as a sum of per-layer inverse log-determinants
    plus the standard-normal base log-density at the recovered latent.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> flow = _RealNVPFlow()
        >>> x = torch.randn(4, 2)
        >>> flow.log_density(x).shape
        torch.Size([4])
    """

    def __init__(self, num_layers: int = _FLOW_COUPLING_LAYERS) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_AffineCoupling(transform_index=i % 2) for i in range(num_layers))

    def synthesize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Push a latent sample through the stack in the generative direction.

        Args:
            z: Latent input, shape ``(..., 2)``.

        Returns:
            ``(x, log_det)``: the synthesized residual and the stack's total
            forward log-determinant, shape ``(...,)``.

        Examples:
            >>> import torch
            >>> flow = _RealNVPFlow()
            >>> x, log_det = flow.synthesize(torch.zeros(3, 2))
            >>> x.shape, log_det.shape
            (torch.Size([3, 2]), torch.Size([3]))
        """
        log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        x = z
        for module in self.layers:
            layer = cast("_AffineCoupling", module)
            x, layer_log_det = layer.synthesize(x)
            log_det = log_det + layer_log_det
        return x, log_det

    def invert(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Recover the latent sample and the inverse stack's log-determinant.

        Applies each layer's inverse in reverse order — the stack was built
        ``z -> layer_0 -> ... -> layer_{K-1} -> x``, so undoing it runs
        ``layer_{K-1}`` first.

        Args:
            x: Residual input, shape ``(..., 2)``.

        Returns:
            ``(z, log_det)``: the recovered latent and the stack's total
            inverse log-determinant, shape ``(...,)``.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> flow = _RealNVPFlow()
            >>> z = torch.randn(5, 2)
            >>> x, _ = flow.synthesize(z)
            >>> z_back, _ = flow.invert(x)
            >>> torch.allclose(z_back, z, atol=1e-5)
            True
        """
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        z = x
        for module in reversed(self.layers):
            layer = cast("_AffineCoupling", module)
            z, layer_log_det = layer.invert(z)
            log_det = log_det + layer_log_det
        return z, log_det

    def log_density(self, x: Tensor) -> Tensor:
        """Return :math:`\\log G_\\phi(x)`, the flow's induced density at ``x``.

        Args:
            x: Standardized residuals, shape ``(..., 2)``.

        Returns:
            Log-density values, shape ``(...,)``.
        """
        z, log_det = self.invert(x)
        base = cast("Tensor", Normal(0.0, 1.0).log_prob(z)).sum(dim=-1)  # type: ignore[no-untyped-call]
        return base + log_det


def _select_visible(mu_hat: Tensor, sigma_raw: Tensor, mu_gt: Tensor, visibility: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten ``(N, K, 2)`` point tensors to ``(V, 2)`` rows of *labelled* points only.

    The masking A66 describes, applied to the **inputs** rather than to the reduction.
    Selecting afterwards makes the loss *value* right and everything else wrong: an
    unlabelled point's coordinate is not ground truth and may be anything at all, and
    until it was dropped here every one of them was still divided by ``sigma_hat``,
    pushed through the flow, and evaluated under a density — so a single non-finite
    unlabelled annotation produced ``NaN`` gradients on the shared flow weights and on
    every visible point's own ``mu_hat``, while the reported loss stayed healthy and the
    masked value stayed exactly correct. With :mod:`torch.distributions`' argument
    validation left at its default the same input does not even get that far: the base
    density rejects the non-finite residual and a masked-out annotation raises
    :class:`ValueError` out of the loss.

    Selecting first is also cheaper — the flow runs on the labelled fraction — and makes
    the guarantee structural: a ``v == 0`` row is not part of the graph, so no gradient
    can reach it and none can leave it.

    Args:
        mu_hat: Predicted point coordinates, shape ``(N, K, 2)``.
        sigma_raw: Raw per-axis uncertainty, shape ``(N, K, 2)``.
        mu_gt: Ground-truth point coordinates, shape ``(N, K, 2)``.
        visibility: COCO-style visibility, shape ``(N, K)``; ``v > 0`` is kept (A66).

    Returns:
        ``(residual_inputs, sigma_hat)``: the ``(V, 2)`` ground-truth-minus-prediction
        offsets of the labelled points, and their ``(V, 2)`` sigmoid-activated,
        floor-clamped scales. ``V`` is ``0`` when nothing is labelled, and both tensors
        stay attached to their arguments' graphs.

    Examples:
        >>> import torch
        >>> mu_hat = torch.zeros(1, 2, 2)
        >>> mu_gt = torch.tensor([[[0.5, 0.25], [float("inf"), float("inf")]]])
        >>> offset, sigma_hat = _select_visible(mu_hat, torch.zeros(1, 2, 2), mu_gt, torch.tensor([[2, 0]]))
        >>> offset.tolist()  # the unlabelled point never reaches the arithmetic
        [[0.5, 0.25]]
        >>> sigma_hat.tolist()
        [[0.5, 0.5]]
        >>> _select_visible(mu_hat, torch.zeros(1, 2, 2), mu_gt, torch.tensor([[0, 0]]))[0].shape
        torch.Size([0, 2])
    """
    visible = (visibility > 0).reshape(-1)
    sigma_hat = torch.sigmoid(sigma_raw.reshape(-1, 2)[visible]).clamp(min=_MIN_SIGMA)
    return mu_gt.reshape(-1, 2)[visible] - mu_hat.reshape(-1, 2)[visible], sigma_hat


def _reduce_selected(per_point: Tensor) -> Tensor:
    """Mean an already-selected per-point loss, or a finite zero if nothing was selected.

    Mirrors :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss`'s
    zero-positives handling: a sum divided by ``max(count, 1)`` stays finite
    and attached to ``per_point``'s graph rather than producing a ``NaN``
    ``0 / 0`` mean. The selection itself happened in :func:`_select_visible`,
    before any arithmetic — this only divides.

    Args:
        per_point: Per-point loss values of the labelled points, shape ``(V,)``.

    Returns:
        A scalar (zero-dimensional) tensor.

    Examples:
        >>> import torch
        >>> _reduce_selected(torch.tensor([2.0, 6.0]))
        tensor(4.)
        >>> _reduce_selected(torch.zeros(0))
        tensor(0.)
    """
    return per_point.sum() / max(per_point.numel(), 1)


class RLELoss(nn.Module):
    """Residual log-likelihood loss over keypoint predictions (R14 Eq. 8).

    Unlike every other loss in :mod:`lucid_yolo.losses` (all parameter-free
    pure functions over already-produced network outputs), RLE's flow
    :math:`G_\\phi` carries trainable weights of its own (R14 sec. 4) — it is
    not part of the keypoint head (WP-122 emits only ``mu_hat``/``sigma_raw``)
    but it must still be optimized, checkpointed, and device-placed alongside
    the rest of the model. An :class:`nn.Module` that a
    :class:`~pytorch_lightning.LightningModule` registers as a submodule
    (``self.rle_loss = RLELoss()``) is what makes ``.parameters()`` and
    ``state_dict()`` pick the flow up automatically; a bare function hiding a
    lazily-built global instance would not be discoverable by an optimizer at
    all, and would violate this project's own prohibition on global mutable
    state.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> loss_fn = RLELoss()
        >>> mu_hat = torch.zeros(2, 1, 2, requires_grad=True)
        >>> sigma_raw = torch.zeros(2, 1, 2, requires_grad=True)
        >>> mu_gt = torch.zeros(2, 1, 2)
        >>> visibility = torch.tensor([[2], [0]])  # second instance's point is unlabeled
        >>> loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        >>> loss.ndim
        0
        >>> bool(loss.requires_grad)
        True
    """

    def __init__(self) -> None:
        super().__init__()
        self.flow = _RealNVPFlow()

    def forward(self, mu_hat: Tensor, sigma_raw: Tensor, mu_gt: Tensor, visibility: Tensor) -> Tensor:
        """Compute the masked mean RLE loss.

        Args:
            mu_hat: Predicted point coordinates, shape ``(N, K, 2)`` — the
                decoded output of
                :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`
                for the positive instances an assigner has already selected,
                mirroring
                :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss`'s
                already-gathered-positives convention. **In a normalized
                frame, not input pixels** (A71): ``sigma_hat`` is bounded
                into ``(0, 1)`` by R14's own sigmoid, so a residual it can
                scale is one whose errors are ``O(1)``. Callers map both
                point arguments through
                :func:`~lucid_yolo.ptl.module.normalize_keypoints_to_box`
                first; passing pixels drives the flow non-finite rather than
                merely training badly.
            sigma_raw: Raw, unactivated per-axis uncertainty, shape
                ``(N, K, 2)``
                (:class:`~lucid_yolo.models.heads.detect.DualDetectionHead`'s
                ``keypoint_sigma`` output, WP-122, A65). Passed through
                :func:`torch.sigmoid` here, per R14 sec. 3.3.
            mu_gt: Ground-truth point coordinates, shape ``(N, K, 2)``, in
                the same normalized frame as ``mu_hat`` (A71).
            visibility: COCO-style visibility, shape ``(N, K)`` int64
                (WP-121's ``Targets.keypoint_vis``). Points with ``v == 0``
                do not contribute (A66); ``v >= 1`` do, regardless of
                occlusion. The exclusion is applied by :func:`_select_visible`
                **before** any arithmetic, so an unlabelled point's coordinate
                is never divided, never entered into the flow and never
                evaluated under a density — whatever it holds, including a
                non-finite placeholder, it can neither perturb a gradient nor
                raise out of the base distribution's argument validation.

        Returns:
            A scalar (zero-dimensional) tensor: the mean per-point, per-axis
            loss over every visible point. A finite zero, still attached to
            ``mu_hat`` and ``sigma_raw``'s graphs, when no point is visible.
        """
        offset, sigma_hat = _select_visible(mu_hat, sigma_raw, mu_gt, visibility)  # both (V, 2)
        residual = offset / sigma_hat

        laplace = Laplace(0.0, _BASE_LAPLACE_SCALE)
        base_log_prob = cast("Tensor", laplace.log_prob(residual)).sum(dim=-1)  # type: ignore[no-untyped-call]  # (V,)

        flow_log_prob = self.flow.log_density(residual)  # (V,)

        log_sigma_term = torch.log(sigma_hat).sum(dim=-1)  # (V,)
        per_point = -base_log_prob - flow_log_prob + log_sigma_term  # (V,), R14 Eq. 8 (log s dropped)

        return _reduce_selected(per_point)
