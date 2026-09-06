# SPDX-License-Identifier: Apache-2.0
"""Tests for the Newton-Schulz orthogonalization function (WP-031).

Covers the orthogonality contract across shapes and dtypes (the DoD test),
singular-vector preservation against a known SVD, the zero-matrix edge case,
the dtype round-trip, and input-validation errors.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from lucid_yolo.optim import orthogonalize


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before every test so random matrices are reproducible."""
    torch.manual_seed(0)
    yield


# NOTE (WP-031 tolerance deviation): the R8 coefficient triple
# (3.4445, -4.7750, 2.0315) is the aggressive Muon quintic. Its scalar map
# ``p(x) = a*x + b*x**3 + c*x**5`` has ``p(1) = 0.701``, so 1 is NOT a fixed
# point: the iteration pulls every singular value into a band ~[0.68, 1.14] and
# holds it there (the orthogonality error ``||R R^T - I||_F`` floors near ~2.8
# for a 64x64 matrix even at 50 steps). At the mandated 5 steps (A5) an
# ill-conditioned raw ``randn`` square can additionally leave a small-singular-
# value straggler unlifted. The blueprint's proposed ``< 1e-2`` orthogonality
# tolerance and ``< 5e-2`` ``U V^T`` closeness are therefore physically
# unattainable with these exact coefficients. What is asserted below is the R8
# iteration's real, provable guarantee: on well-conditioned inputs every singular
# value is driven into a tight band around 1, the singular vectors are preserved
# exactly, and no input is ever blown up (top singular value stays < 1.5). See
# the task return note for the DoD tolerance decision.

_BAND_LOW = 0.6
_BAND_HIGH = 1.2


def _singular_values(result: torch.Tensor) -> torch.Tensor:
    """Return the singular values of ``result`` evaluated in float32.

    Examples:
        >>> _singular_values(torch.eye(3)).tolist()
        [1.0, 1.0, 1.0]
    """
    return torch.linalg.svdvals(result.to(torch.float32))


def _random_conditioned(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """Build a random ``rows x cols`` matrix with singular values in [0.5, 2.0].

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> matrix = _random_conditioned(4, 4, torch.float32)
        >>> matrix.shape, matrix.dtype
        (torch.Size([4, 4]), torch.float32)
    """
    rank = min(rows, cols)
    left = torch.linalg.qr(torch.randn(rows, rank))[0]
    right = torch.linalg.qr(torch.randn(cols, rank))[0]
    singular_values = torch.rand(rank) * 1.5 + 0.5
    return ((left * singular_values) @ right.T).to(dtype)


class TestOrthogonality:
    """The result drives all singular values into a tight band around 1."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((64, 64), id="square"),
            pytest.param((32, 128), id="wide"),
            pytest.param((128, 32), id="tall"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_orthogonality(self, shape: tuple[int, int], dtype: torch.dtype) -> None:
        """Singular values land in [0.6, 1.2] across shapes and input dtypes.

        Uses well-conditioned random inputs (singular values in [0.5, 2.0]); this
        is the R8 iteration's real orthogonalization guarantee (band ~[0.68,
        1.14]). Exact ``||R R^T - I||_F < 1e-2`` is unreachable for these
        coefficients (see the module-level note).
        """
        matrix = _random_conditioned(*shape, dtype)

        singular_values = _singular_values(orthogonalize(matrix))

        assert singular_values.min() > _BAND_LOW
        assert singular_values.max() < _BAND_HIGH

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((64, 64), id="square"),
            pytest.param((32, 128), id="wide"),
            pytest.param((128, 32), id="tall"),
        ],
    )
    def test_no_blowup_on_raw_random(self, shape: tuple[int, int]) -> None:
        """A raw random matrix is never expanded (top singular value < 1.5, no NaN)."""
        result = orthogonalize(torch.randn(*shape))

        assert _singular_values(result).max() < 1.5
        assert not torch.isnan(result).any()

    def test_singular_vectors_preserved(self) -> None:
        """For ``M = U S V^T`` with known U, V, ``U^T R V`` is diagonal.

        Orthogonalization rescales the singular values but must leave the left and
        right singular vectors untouched, so the off-diagonal energy of
        ``U^T R V`` is negligible.
        """
        rows, cols = 48, 32
        u = torch.linalg.qr(torch.randn(rows, cols))[0]
        v = torch.linalg.qr(torch.randn(cols, cols))[0]
        singular_values = torch.rand(cols) + 0.5
        matrix = (u * singular_values) @ v.T

        projected = u.T @ orthogonalize(matrix) @ v
        off_diagonal = projected - torch.diag(torch.diag(projected))

        assert torch.linalg.matrix_norm(off_diagonal) < 1e-3


class TestEdgeCases:
    """Degenerate inputs and dtype handling."""

    def test_zero_matrix_returns_zeros(self) -> None:
        """A zero matrix orthogonalizes to zeros with no NaNs."""
        matrix = torch.zeros(16, 16)

        result = orthogonalize(matrix)

        assert torch.equal(result, torch.zeros(16, 16))
        assert not torch.isnan(result).any()

    def test_bfloat16_roundtrip(self) -> None:
        """A bfloat16 input yields a bfloat16 output of the same shape."""
        matrix = torch.randn(32, 64).to(torch.bfloat16)

        result = orthogonalize(matrix)

        assert result.dtype == torch.bfloat16
        assert result.shape == matrix.shape

    def test_input_not_mutated(self) -> None:
        """The input tensor is left unchanged by the call."""
        matrix = torch.randn(16, 16)
        original = matrix.clone()

        orthogonalize(matrix)

        assert torch.equal(matrix, original)

    def test_float64_is_not_demoted_to_float32(self) -> None:
        """A float64 input keeps float64 arithmetic instead of being computed at fp32.

        The iteration used to cast every input to float32 unconditionally, then return
        the fp32 answer wearing the caller's dtype. A magnitude float64 holds and float32
        does not makes that visible rather than merely imprecise: ``1e200`` overflows the
        fp32 range, so the normalization divided infinity by infinity and every entry
        came back NaN under a ``torch.float64`` dtype that claimed otherwise.
        """
        matrix = torch.eye(4, dtype=torch.float64) * 1e200

        result = orthogonalize(matrix)

        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()

    def test_float16_still_promotes_to_float32(self) -> None:
        """A half-precision input is computed at float32, which is the A5/AMP gate.

        The float64 fix promotes rather than pins, so this is the half it must not have
        broken: fp16 is narrower than float32 and the iteration's ``x @ x.T`` products
        overflow its range readily, which is why the promotion existed at all. The
        returned dtype is still the caller's.
        """
        matrix = torch.randn(32, 64).to(torch.float16)

        result = orthogonalize(matrix)

        assert result.dtype == torch.float16
        assert torch.isfinite(result).all()


class TestInputValidation:
    """Non-2D inputs and step counts that run no iteration are rejected."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((8,), id="1d"),
            pytest.param((4, 4, 4), id="3d"),
        ],
    )
    def test_non_2d_raises(self, shape: tuple[int, ...]) -> None:
        """A 1D or 3D input raises ValueError."""
        matrix = torch.randn(*shape)

        with pytest.raises(ValueError, match="2D matrix"):
            orthogonalize(matrix)

    @pytest.mark.parametrize(
        "steps",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_steps_below_one_raises(self, steps: int) -> None:
        """A step count that runs no iteration raises ValueError naming ``steps``.

        ``range(0)`` and ``range(-1)`` are both empty, so the loop was skipped and the
        Frobenius-normalized *input* came back wearing this function's name: a matrix
        whose singular values are the input's rescaled, not driven anywhere near 1.
        :class:`~lucid_yolo.optim.musgd.MuSGD` has always refused ``ns_steps < 1``, so
        the guard existed only on the caller's side of a public function (WP-171).
        """
        matrix = torch.randn(16, 16)

        with pytest.raises(ValueError, match="steps"):
            orthogonalize(matrix, steps=steps)

    def test_a_single_step_still_moves_the_singular_values(self) -> None:
        """``steps=1`` is accepted and is not a no-op, which is what makes 1 the boundary.

        The refusal above is a claim about where the usable range starts; asserting the
        first accepted value does something distinguishes that boundary from an
        off-by-one that would have rejected a working configuration.
        """
        matrix = _random_conditioned(16, 16, torch.float32)

        result = orthogonalize(matrix, steps=1)

        assert not torch.allclose(_singular_values(result), _singular_values(matrix))
        assert not torch.isnan(result).any()
