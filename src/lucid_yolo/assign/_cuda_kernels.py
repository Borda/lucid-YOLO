# SPDX-License-Identifier: Apache-2.0
"""Device passes for the existing dense assignment expressions.

No reduction adds floating-point values: row maxima and first-index argmax retain
TAL's choices. Native PyTorch still supplies the IoU power and rotated sine/cosine,
whose library implementations must not be approximated by the compiler.
"""

from typing import Any  # noqa: F401 -- mypy reads the DSL type comments.

from packaging.version import Version

try:
    import triton
    import triton.language as tl
except ImportError:
    _AVAILABLE = False
else:
    _AVAILABLE = Version("3.0") <= Version(triton.__version__) < Version("4")

if _AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _divide(a, b):  # type: (Any, Any) -> Any
        """Retain native IEEE division, including subnormal inputs and results.

        The compiler's div_rn intrinsic flushes subnormals in supported runtimes.
        This instruction keeps the native round-to-nearest operation without FTZ.
        """
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _maximum(a, b):  # type: (Any, Any) -> Any
        """Propagate NaN as the native maximum does, including during reduction."""
        return tl.where(a != a, a, tl.where(b != b, b, tl.maximum(a, b)))  # noqa: PLR0124

    @triton.jit  # type: ignore[untyped-decorator]
    def _minimum(a, b):  # type: (Any, Any) -> Any
        """Propagate NaN before taking the finite minimum."""
        return tl.where(a != a, a, tl.where(b != b, b, tl.minimum(a, b)))  # noqa: PLR0124

    @triton.jit(do_not_specialize=["N"])  # type: ignore[untyped-decorator]
    def _iou_pass(gt, pred, out, N, A: tl.constexpr, eps, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """Write cross IoU without materializing broadcast corner and extent tensors."""
        row = tl.program_id(0).to(tl.int64)
        anchor = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = anchor < A
        image = row // N
        g = row * 4
        p = (image * A + anchor) * 4
        gx1, gy1 = tl.load(gt + g), tl.load(gt + g + 1)
        gx2, gy2 = tl.load(gt + g + 2), tl.load(gt + g + 3)
        px1, py1 = tl.load(pred + p, valid, 0.0), tl.load(pred + p + 1, valid, 0.0)
        px2, py2 = tl.load(pred + p + 2, valid, 0.0), tl.load(pred + p + 3, valid, 0.0)
        width = _maximum(_minimum(gx2, px2) - _maximum(gx1, px1), 0.0)
        height = _maximum(_minimum(gy2, py2) - _maximum(gy1, py1), 0.0)
        intersection = width * height
        gt_area = _maximum(gx2 - gx1, 0.0) * _maximum(gy2 - gy1, 0.0)
        pred_area = _maximum(px2 - px1, 0.0) * _maximum(py2 - py1, 0.0)
        union = gt_area + pred_area - intersection
        tl.store(out + row * A + anchor, _divide(intersection, union + eps), valid)

    @triton.jit(do_not_specialize=["N", "MB", "MN", "MA"])  # type: ignore[untyped-decorator]
    def _metric_pass(  # type: ignore[no-untyped-def]  # noqa: PLR0913,PLR0917 -- pointer and stride ABI.
        scores,
        labels,
        mask,
        powered_iou,
        out,
        N,
        A: tl.constexpr,
        C: tl.constexpr,
        SB: tl.constexpr,
        SA: tl.constexpr,
        SC: tl.constexpr,
        MB,
        MN,
        MA,
        BLOCK: tl.constexpr,
    ):
        """Gather the current class score and apply the unchanged power and mask products."""
        row = tl.program_id(0).to(tl.int64)
        anchor = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = anchor < A
        image, instance = row // N, row % N
        label = tl.maximum(tl.load(labels + row), 0)
        legal = label < C
        tl.device_assert(legal, "class index out of bounds")
        score = tl.load(scores + image * SB + anchor * SA + label * SC, valid & legal, 0.0).to(tl.float32)
        power = tl.load(powered_iou + row * A + anchor, valid, 0.0)
        selected = tl.load(mask + image * MB + instance * MN + anchor * MA, valid, 0).to(tl.float32)
        tl.store(out + row * A + anchor, (score * power) * selected, valid)

    @triton.jit  # type: ignore[untyped-decorator]
    def _inside_pass(  # type: ignore[no-untyped-def]
        points, boxes, cosines, sines, out, A: tl.constexpr, PS: tl.constexpr, PC: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Test every point against one rotated row, retaining the native trig results."""
        row = tl.program_id(0).to(tl.int64)
        anchor = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = anchor < A
        base = row * 5
        dx = tl.load(points + anchor * PS, valid, 0.0) - tl.load(boxes + base)
        dy = tl.load(points + anchor * PS + PC, valid, 0.0) - tl.load(boxes + base + 1)
        width, height = tl.load(boxes + base + 2), tl.load(boxes + base + 3)
        cosine, sine = tl.load(cosines + row), tl.load(sines + row)
        local_x = dx * cosine + dy * sine
        local_y = dy * cosine - dx * sine
        inside = (tl.abs(local_x) <= width * 0.5) & (tl.abs(local_y) <= height * 0.5)
        tl.store(out + row * A + anchor, inside, valid)

    @triton.jit  # type: ignore[untyped-decorator]
    def _first_maximum(v1, i1, v2, i2):  # type: (Any, Any, Any, Any) -> tuple[Any, Any]
        """Keep the first NaN, otherwise the first maximum, as native argmax does."""
        n1 = v1 != v1  # noqa: PLR0124 -- IEEE NaN predicate.
        n2 = v2 != v2  # noqa: PLR0124 -- IEEE NaN predicate.
        left = (n1 & ((~n2) | (i1 < i2))) | ((~n1) & (~n2) & ((v1 > v2) | ((v1 == v2) & (i1 < i2))))
        return tl.where(left, v1, v2), tl.where(left, i1, i2)

    @triton.jit(do_not_specialize=["N"])  # type: ignore[untyped-decorator]
    def _conflicts_pass(M, T, OUTPUT, N, A: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """Resolve contested anchors without dense masked-score and winner tensors."""
        b = tl.program_id(0).to(tl.int64)
        a = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        n = tl.arange(0, NP)
        ok = (n[:, None] < N) & (a[None, :] < A)
        x = (b * N + n[:, None]) * A + a[None, :]
        m = tl.load(M + x, ok, 0)
        t = tl.load(T + x, ok, 0)
        ranked = tl.where(n[:, None] < N, tl.where(m, t, -1.0), -float("inf"))
        _, best = tl.reduce((ranked, tl.broadcast_to(n[:, None], ranked.shape)), 0, _first_maximum)
        many = tl.sum(m.to(tl.int32), 0) > 1
        out = m & ((~many[None, :]) | (n[:, None] == best[None, :]))
        tl.store(OUTPUT + x, out, ok)
