# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the content-pin audit script.

Covers ``scripts/lint/audit_content_pins.py`` on synthetic trees under ``tmp_path``
rather than against the live repository -- the live tree is what the ``content-pins``
pre-commit hook reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]


audit = load_script("audit_content_pins")


def _write(root: Path, relative: Path | str, text: str) -> None:
    """Write ``text`` to ``root / relative``, creating parent directories.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _write(Path(tmp), "a/b.txt", "x")
        ...     (Path(tmp) / "a" / "b.txt").read_text(encoding="utf-8")
        'x'
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def clean_tree(tmp_path: Path) -> Path:
    """A tree satisfying every pin: plain top-k module, one det and one seg recipe."""
    _write(tmp_path, audit.TOPK_MODULE, "def decode(scores):\n    return scores.topk(300)\n")
    _write(tmp_path, audit.CONFIGS_DIR / "det_nano.yaml", "model:\n  num_classes: 80\n")
    _write(tmp_path, audit.CONFIGS_DIR / "seg_nano.yaml", "model:\n  task: segment\n")
    return tmp_path


class TestTopkSuppressionFree:
    """`check_topk_is_suppression_free` reads the decoder source case-insensitively."""

    def test_plain_source_passes(self, clean_tree: Path) -> None:
        """A decoder naming no forbidden word produces no violation."""
        assert audit.check_topk_is_suppression_free(clean_tree) == []

    @pytest.mark.parametrize("word", ["NMS", "box_iou", "torchvision"])
    def test_forbidden_word_named(self, clean_tree: Path, word: str) -> None:
        """Any forbidden substring, in any case, is reported by its lower-cased name."""
        _write(clean_tree, audit.TOPK_MODULE, f"# uses {word}\n")

        assert audit.check_topk_is_suppression_free(clean_tree) == [
            f"src/lucid_yolo/decode/topk_e2e.py names {word.lower()!r}"
        ]


class TestConfigDeterminism:
    """`check_configs_leave_determinism_to_cli` refuses a recipe pinning the flag."""

    def test_clean_recipes_pass(self, clean_tree: Path) -> None:
        """Recipes without the key produce no violation."""
        assert audit.check_configs_leave_determinism_to_cli(clean_tree) == []

    def test_pinned_recipe_named(self, clean_tree: Path) -> None:
        """A recipe carrying ``deterministic:`` is named, detection or not."""
        _write(clean_tree, audit.CONFIGS_DIR / "det_nano.yaml", "trainer:\n  deterministic: true\n")

        assert audit.check_configs_leave_determinism_to_cli(clean_tree) == [
            "src/lucid_yolo/configs/det_nano.yaml pins trainer.deterministic"
        ]


class TestConfigTask:
    """`check_non_detection_configs_state_task` requires ``task:`` outside ``det_*``."""

    def test_stated_task_passes(self, clean_tree: Path) -> None:
        """A non-detection recipe naming its task produces no violation."""
        assert audit.check_non_detection_configs_state_task(clean_tree) == []

    def test_omitted_task_named(self, clean_tree: Path) -> None:
        """A non-detection recipe without ``task:`` is named; the det recipe is exempt."""
        _write(clean_tree, audit.CONFIGS_DIR / "obb_nano.yaml", "model:\n  num_classes: 15\n")

        assert audit.check_non_detection_configs_state_task(clean_tree) == [
            "src/lucid_yolo/configs/obb_nano.yaml omits model.task"
        ]

    def test_empty_directory_refused(self, tmp_path: Path) -> None:
        """No non-detection recipe at all is a violation, so a moved directory cannot pass vacuously."""
        (tmp_path / audit.CONFIGS_DIR).mkdir(parents=True)

        assert audit.check_non_detection_configs_state_task(tmp_path) == [
            "no non-detection recipe found under src/lucid_yolo/configs"
        ]


class TestMain:
    """The CLI's exit code is the hook's verdict."""

    def test_exits_zero_on_clean_tree(self, clean_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A clean tree reports so and exits `0`."""
        assert audit.main(["--repo-root", str(clean_tree)]) == 0
        assert "clean" in capsys.readouterr().out

    def test_missing_module_reported_not_raised(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """An absent pinned file exits `1` with the failing check named rather than a traceback."""
        code = audit.main(["--repo-root", str(tmp_path)])

        assert code == 1
        assert "check_topk_is_suppression_free" in capsys.readouterr().out
