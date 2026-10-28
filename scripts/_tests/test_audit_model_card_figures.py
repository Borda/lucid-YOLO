# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the model-card wiring-gate audit (DOC1).

Covers ``scripts/lint/audit_model_card_figures.py`` on throwaway card and golden trees under
``tmp_path``; the ``model-card-figures`` pre-commit hook is what reads the live cards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from script_loader import load_script

audit = load_script("audit_model_card_figures")

GOLDEN_LINE = (
    "It measures train recall **{value} ± {tolerance}** against a floor (`goldens/gpu/overfit_micro_det.json`)."
)


def _tree(root: Path, line: str, *, recorded: float = 0.996622, tolerance: float = 0.04) -> None:
    """Write one card holding ``line`` and the golden it cites under ``root``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _tree(Path(tmp), "x")
        ...     sorted(path.name for path in Path(tmp).rglob("*.*"))
        ['detection.md', 'overfit_micro_det.json']
    """
    card = root / "docs" / "model_cards" / "detection.md"
    card.parent.mkdir(parents=True)
    card.write_text(f"# Card\n\n{line}\n", encoding="utf-8")
    golden = root / "goldens" / "gpu" / "overfit_micro_det.json"
    golden.parent.mkdir(parents=True)
    payload = {"values": {"train_recall_at_050": recorded}, "tolerances": {"train_recall_at_050": tolerance}}
    golden.write_text(json.dumps(payload), encoding="utf-8")


class TestFindViolations:
    def test_figure_rounded_to_card_precision_is_clean(self, tmp_path: Path) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        assert audit.find_violations(tmp_path) == []

    def test_stale_value_is_named_with_both_numbers(self, tmp_path: Path) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.8146", tolerance="0.04"))
        (message,) = audit.find_violations(tmp_path)
        assert message.startswith("docs/model_cards/detection.md:3: ")
        assert "0.8146" in message
        assert "0.996622" in message

    def test_stale_tolerance_is_named(self, tmp_path: Path) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.05"))
        (message,) = audit.find_violations(tmp_path)
        assert "tolerance 0.05" in message

    def test_historical_figure_beside_a_work_package_is_exempt(self, tmp_path: Path) -> None:
        line = GOLDEN_LINE.format(value="0.9966", tolerance="0.04") + " It first stood at 0.3357 in WP-132."
        _tree(tmp_path, line)
        assert audit.find_violations(tmp_path) == []

    def test_card_citing_no_golden_is_reported(self, tmp_path: Path) -> None:
        _tree(tmp_path, "No gate is cited here.")
        (message,) = audit.find_violations(tmp_path)
        assert "cites no goldens/gpu/overfit_micro_*.json" in message

    def test_citing_line_without_bold_pair_is_reported(self, tmp_path: Path) -> None:
        _tree(tmp_path, "Recall 0.9966 (`goldens/gpu/overfit_micro_det.json`).")
        (message,) = audit.find_violations(tmp_path)
        assert "found 0" in message

    def test_missing_golden_is_reported_not_raised(self, tmp_path: Path) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        (tmp_path / "goldens" / "gpu" / "overfit_micro_det.json").unlink()
        (message,) = audit.find_violations(tmp_path)
        assert "cannot read one metric" in message

    def test_golden_with_two_tolerances_is_reported(self, tmp_path: Path) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        golden = tmp_path / "goldens" / "gpu" / "overfit_micro_det.json"
        golden.write_text(json.dumps({"values": {"a": 1.0, "b": 2.0}, "tolerances": {"a": 0.1, "b": 0.2}}))
        (message,) = audit.find_violations(tmp_path)
        assert "cannot read one metric" in message


class TestMain:
    def test_exits_zero_and_says_clean_when_cards_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        assert audit.main(["--repo-root", str(tmp_path)]) == 0
        assert "model-card-figures-audit clean" in capsys.readouterr().out

    def test_exits_one_and_lists_the_card_when_a_figure_is_stale(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _tree(tmp_path, GOLDEN_LINE.format(value="0.5", tolerance="0.04"))
        assert audit.main(["--repo-root", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert out.startswith("model-card-figures-audit FAILED: 1 violation(s)\n  - docs/model_cards/detection.md:3:")
