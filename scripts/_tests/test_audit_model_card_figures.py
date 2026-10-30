# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the model-card wiring-gate audit (DOC1).

Covers ``scripts/lint/audit_model_card_figures.py`` on throwaway card and golden trees under ``tmp_path``; the
``model-card-figures`` pre-commit hook is what reads the live cards.
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
    def test_figure_rounded_to_card_precision_clean(self, tmp_path: Path) -> None:
        """A card figure equal to the golden's value at the card's precision reports no violation.

        The golden records 0.996622; the card quotes 0.9966, the same number rounded to the four places it prints.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        assert audit.find_violations(tmp_path) == []

    def test_stale_value_named_with_both_numbers(self, tmp_path: Path) -> None:
        """A card value that differs from the golden is reported with its location and both numbers.

        The card says 0.8146 while the golden records 0.996622; the message must carry both so the fix needs no lookup.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.8146", tolerance="0.04"))
        (message,) = audit.find_violations(tmp_path)
        assert message.startswith("docs/model_cards/detection.md:3: ")
        assert "0.8146" in message
        assert "0.996622" in message

    def test_stale_tolerance_named(self, tmp_path: Path) -> None:
        """A card tolerance that differs from the golden's is reported, naming the stale tolerance.

        The value matches the golden; only the card's tolerance (0.05 against 0.04) has drifted.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.05"))
        (message,) = audit.find_violations(tmp_path)
        assert "tolerance 0.05" in message

    def test_historical_figure_beside_work_package_exempt(self, tmp_path: Path) -> None:
        """A figure quoted beside a work-package id is history, so it is not checked against the golden.

        The line carries the live figure plus "0.3357 in WP-132"; only the live one is held to the golden.
        """
        line = GOLDEN_LINE.format(value="0.9966", tolerance="0.04") + " It first stood at 0.3357 in WP-132."
        _tree(tmp_path, line)
        assert audit.find_violations(tmp_path) == []

    def test_card_citing_no_golden_reported(self, tmp_path: Path) -> None:
        """A card that cites no ``goldens/gpu/overfit_micro_*.json`` file is reported.

        Without a citation the card's figures are not tied to any golden, which is the wiring gate this audit is.
        """
        _tree(tmp_path, "No gate is cited here.")
        (message,) = audit.find_violations(tmp_path)
        assert "cites no goldens/gpu/overfit_micro_*.json" in message

    def test_citing_line_without_bold_pair_reported(self, tmp_path: Path) -> None:
        """A line citing a golden but holding no bold ``value ± tolerance`` pair is reported as ``found 0``.

        The figure is plain text here, so there is nothing for the audit to compare and it must say so.
        """
        _tree(tmp_path, "Recall 0.9966 (`goldens/gpu/overfit_micro_det.json`).")
        (message,) = audit.find_violations(tmp_path)
        assert "found 0" in message

    def test_missing_golden_reported_not_raised(self, tmp_path: Path) -> None:
        """A cited golden that does not exist is reported as a violation instead of raising.

        The hook must give a readable verdict on a broken tree, not a traceback from the missing file.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        (tmp_path / "goldens" / "gpu" / "overfit_micro_det.json").unlink()
        (message,) = audit.find_violations(tmp_path)
        assert "cannot read one metric" in message

    def test_golden_with_two_tolerances_reported(self, tmp_path: Path) -> None:
        """A golden holding two metrics is ambiguous for a single card figure and is reported.

        The audit compares one ``value ± tolerance`` pair per citation, so a golden with metrics ``a`` and ``b`` cannot
        be read as one metric.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        golden = tmp_path / "goldens" / "gpu" / "overfit_micro_det.json"
        golden.write_text(json.dumps({"values": {"a": 1.0, "b": 2.0}, "tolerances": {"a": 0.1, "b": 0.2}}))
        (message,) = audit.find_violations(tmp_path)
        assert "cannot read one metric" in message


class TestMain:
    def test_exits_zero_and_says_clean_when_cards_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI exits 0 and prints its clean verdict when every card figure matches its golden."""
        _tree(tmp_path, GOLDEN_LINE.format(value="0.9966", tolerance="0.04"))
        assert audit.main(["--repo-root", str(tmp_path)]) == 0
        assert "model-card-figures-audit clean" in capsys.readouterr().out

    def test_exits_one_and_lists_card_when_figure_stale(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI exits 1 and lists the offending card line when a figure no longer matches its golden.

        The card says 0.5 against a recorded 0.996622; the verdict counts one violation and names ``detection.md:3``.
        """
        _tree(tmp_path, GOLDEN_LINE.format(value="0.5", tolerance="0.04"))
        assert audit.main(["--repo-root", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert out.startswith("model-card-figures-audit FAILED: 1 violation(s)\n  - docs/model_cards/detection.md:3:")
