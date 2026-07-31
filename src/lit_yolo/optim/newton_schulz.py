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


def orthogonalize(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor ``U V^T`` of a 2D matrix.

    Runs the quintic Newton-Schulz iteration (R7, R8) for ``steps`` iterations.
    The input is first scaled to unit Frobenius norm so the iteration converges;
    tall inputs (more rows than columns) are transposed to the wide orientation
    to halve the cost of the ``X @ X^T`` product, then transposed back. All
    arithmetic runs in float32 regardless of the input dtype (A5/Phase-4 gate:
    fp32 under AMP), and the result is cast back to the input dtype.

    The input is never mutated in place, and no gradient context is entered here
    -- the caller (the optimizer) is expected to wrap the call in
    ``torch.no_grad()``. A zero matrix orthogonalizes to zeros without producing
    NaNs.

    Args:
        matrix: A 2D tensor of any floating-point dtype.
        steps: Number of Newton-Schulz iterations to run (default 5, per A5).
        eps: Small constant added to the Frobenius norm before the initial
            normalization to avoid division by zero on a zero matrix.

    Returns:
        A tensor with the same shape and dtype as ``matrix`` whose singular
        values are approximately 1 and whose singular vectors match those of
        ``matrix`` (i.e. an approximation of ``U V^T``).

    Raises:
        ValueError: If ``matrix`` is not 2-dimensional.

    Examples:
        >>> import torch
        >>> m = torch.randn(64, 64)
        >>> q = orthogonalize(m)
        >>> torch.linalg.matrix_norm(q @ q.T - torch.eye(64)) < 1e-2
        tensor(True)
    """
    if matrix.ndim != 2:
        msg = f"orthogonalize expects a 2D matrix, got a {matrix.ndim}D tensor"
        raise ValueError(msg)

    in_dtype = matrix.dtype
    a, b, c = _COEFFS

    # fp32 throughout; the division below always allocates a fresh tensor, so the
    # input is never written to in place even when ``.to`` returns it unchanged.
    x = matrix.to(torch.float32)

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
