# SPDX-License-Identifier: Apache-2.0
"""Newton-Schulz orthogonalization for the Muon half of the MuSGD optimizer.

The quintic Newton-Schulz iteration approximates the orthogonal factor ``U V^T``
of a matrix ``M`` with singular-value decomposition ``M = U S V^T``. It drives
every singular value toward 1 while leaving the singular vectors ``U`` and ``V``
untouched, i.e. it replaces ``S`` with the identity. Muon uses this as a cheap,
matmul-only stand-in for a full SVD when orthogonalizing the momentum buffer.

The iteration and its coefficient triple are taken from the Muon description
(arXiv:2502.16982) and the public write-up of the method; see docs/PROVENANCE.md
entries R7 and R8. The module is written by hand from the published equations
(AGENTS.md sec. 6-7): no optimizer implementation is consulted.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Quintic Newton-Schulz coefficient triple ``(a, b, c)`` from the public
#: write-up (R8). The step is ``X <- a*X + (b*A + c*A@A) @ X`` with ``A = X@X^T``.
_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


def _orthogonalize_batched(matrices: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Orthogonalize an exact-shape leading batch with native batched products.

    This is the three-dimensional counterpart of :func:`orthogonalize` for the
    optimizer's exact-shape buckets. Each ``matrices[index]`` keeps an independent
    Frobenius normalization and follows the same quintic iteration, but
    :func:`torch.baddbmm` submits each polynomial and update as one batched operation
    and folds the scale-and-add into the product's epilogue. :func:`torch.vmap` had
    already lowered each product to one batched multiply, so this drops the ``vmap``
    layer rather than a per-slice dispatch. The different reduction schedule can move the last few floating-point
    bits; it does not change the formula, coefficients, step count or scale.

    Args:
        matrices: A 3D ``(batch, rows, columns)`` tensor whose slices have one shape.
        steps: Number of Newton--Schulz iterations to run.
        eps: Constant added to each slice's Frobenius norm before normalization.

    Returns:
        A tensor with the same shape and dtype as ``matrices``.

    Raises:
        ValueError: If ``matrices`` is not 3D, or if ``steps`` is below 1.

    Note:
        The iteration below is a second copy of :func:`orthogonalize`'s, kept apart so that
        the scalar function stays the untouched reference. A change to the formula, the
        coefficients, the normalization or ``eps`` made in one has to be made in the other.
        Two tests hold the copies together: the first compares this function with the
        scalar loop over square, wide and tall slices, the second compares the optimizer
        built on it with the scalar update over 40 steps.

            ``tests/optim/test_newton_schulz.py::TestNativeBatch::test_matches_per_matrix_loop``

            ``tests/optim/test_musgd.py::TestExactShapeBatching::test_batched_trajectory_matches_unbatched_reference_over_many_steps``

    Examples:
        Each slice orthogonalizes independently, matching :func:`orthogonalize`'s
        own guarantee that no singular value is left blown up:

        ```pycon
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> q = _orthogonalize_batched(torch.randn(3, 64, 64))
        >>> q.shape
        torch.Size([3, 64, 64])
        >>> bool(torch.linalg.svdvals(q).max() < 1.5)  # never blown up, any slice
        True

        ```

        A 2D input is refused -- the leading batch axis is mandatory:

        ```pycon
        >>> _orthogonalize_batched(torch.randn(4, 4))
        Traceback (most recent call last):
            ...
        ValueError: _orthogonalize_batched expects a 3D tensor, got a 2D tensor

        ```
    """
    if matrices.ndim != 3:
        msg = f"_orthogonalize_batched expects a 3D tensor, got a {matrices.ndim}D tensor"
        raise ValueError(msg)
    if steps < 1:
        msg = f"_orthogonalize_batched needs steps >= 1 to iterate, got steps={steps}"
        raise ValueError(msg)

    in_dtype = matrices.dtype
    a, b, c = _COEFFS
    x = matrices.to(torch.promote_types(in_dtype, torch.float32))
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(in_dtype)


def orthogonalize(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor ``U V^T`` of a 2D matrix.

    Runs the quintic Newton-Schulz iteration (R7, R8) for ``steps`` iterations.
    The input is first scaled to unit Frobenius norm so the iteration converges;
    tall inputs (more rows than columns) are transposed to the wide orientation
    to halve the cost of the ``X @ X^T`` product, then transposed back.

    Arithmetic runs at ``promote_types(matrix.dtype, float32)`` — float32 or wider,
    never narrower — and the result is cast back to the input dtype. That keeps the
    A5/Phase-4 gate intact where it applies: an fp16 or bf16 momentum buffer under AMP
    still promotes up to float32, which is the case the gate is about. What it no longer
    does is *demote*: a float64 caller used to have its iteration run at fp32 and the
    fp32 answer returned to it as a float64 tensor, so the dtype it read back overstated
    the precision it got, and an input whose magnitude exceeds the fp32 range (``1e200``,
    say) overflowed to infinity and returned NaNs.

    Scale-invariance has one floor. The initial normalization divides by
    ``‖M‖_F + eps`` rather than by ``‖M‖_F``, so it is exactly scale-invariant only
    while ``‖M‖_F >> eps``. Below roughly ``1e-7`` — the default ``eps`` — the added
    constant is a growing fraction of the divisor, the normalized matrix comes out
    short of unit norm, and the iteration lands off the fixed point it converges to
    elsewhere: at ``‖M‖_F = 1e-8`` the largest singular value of the result overshoots
    by about 7%. This is the published R8 formula and the behaviour is left as it
    stands; the floor is recorded because a caller orthogonalizing a near-zero
    momentum buffer is the one who meets it. A genuinely zero matrix is unaffected —
    ``eps`` is what makes it return zeros instead of NaNs, which is the reason it is
    there.

    The input is never mutated in place, and no gradient context is entered here
    -- the caller (the optimizer) is expected to wrap the call in
    ``torch.no_grad()``. A zero matrix orthogonalizes to zeros without producing
    NaNs.

    Args:
        matrix: A 2D tensor of any floating-point dtype.
        steps: Number of Newton-Schulz iterations to run (default 5, per A5).
        eps: Small constant added to the Frobenius norm before the initial
            normalization to avoid division by zero on a zero matrix. It is also
            the scale floor described above: normalization is scale-invariant only
            for ``‖matrix‖_F`` well above this value.

    Returns:
        A tensor with the same shape and dtype as ``matrix`` whose singular
        values are approximately 1 and whose singular vectors match those of
        ``matrix`` (i.e. an approximation of ``U V^T``).

    Raises:
        ValueError: If ``matrix`` is not 2-dimensional, or if ``steps`` is below 1.
            ``steps=0`` skips the loop entirely and returns the Frobenius-normalized
            input, whose singular values are whatever the input's were rescaled -- a
            matrix that is not orthogonal and carries nothing saying so. The caller in
            :class:`~lucid_yolo.optim.musgd.MuSGD` has always refused ``ns_steps < 1``;
            this function is public and had been relying on it.

    Examples:
        The iteration does not produce an exactly orthogonal matrix — with these
        coefficients that is unattainable, as ``tests/optim/test_newton_schulz.py``
        records — so what it guarantees is that no singular value is expanded and
        the well-conditioned ones are pulled toward 1:

        ```pycon
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> q = orthogonalize(torch.randn(64, 64))
        >>> q.shape
        torch.Size([64, 64])
        >>> bool(torch.linalg.svdvals(q).max() < 1.5)  # never blown up
        True

        ```

        A step count below one runs no iteration at all, so it is refused rather than
        answering with a normalized copy of the input:

        ```pycon
        >>> orthogonalize(torch.randn(4, 4), steps=0)
        Traceback (most recent call last):
            ...
        ValueError: orthogonalize needs steps >= 1 to iterate, got steps=0

        ```

        A float64 caller keeps float64 arithmetic, so a magnitude fp32 cannot hold
        survives the iteration instead of overflowing to NaN:

        ```pycon
        >>> big = torch.eye(4, dtype=torch.float64) * 1e200
        >>> bool(torch.isfinite(orthogonalize(big)).all())
        True

        ```
    """
    if matrix.ndim != 2:
        msg = f"orthogonalize expects a 2D matrix, got a {matrix.ndim}D tensor"
        raise ValueError(msg)

    if steps < 1:
        msg = f"orthogonalize needs steps >= 1 to iterate, got steps={steps}"
        raise ValueError(msg)

    in_dtype = matrix.dtype
    a, b, c = _COEFFS

    # fp32 or the input's own dtype, whichever is wider: promote fp16/bf16 up to fp32
    # (the A5 AMP gate) without demoting fp64 down to it. The division below always
    # allocates a fresh tensor, so the input is never written to in place even when
    # ``.to`` returns it unchanged.
    x = matrix.to(torch.promote_types(in_dtype, torch.float32))

    # Iterate on the wide orientation: transpose tall matrices so ``x @ x.T`` acts
    # on the smaller ``min(rows, cols)`` dimension.
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / (torch.linalg.matrix_norm(x) + eps)

    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.T

    result: Tensor = x.to(in_dtype)
    return result
