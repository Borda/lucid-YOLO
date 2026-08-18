# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: flat-test-group audit script (WP-128, split off WP-117).

Covers ``scripts/lint/audit_flat_test_groups.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``tests/`` tree -- the
live tree is exercised instead by the pre-commit hook itself
(``.pre-commit-config.yaml``'s ``flat-test-group-audit``) each time a test file changes.
This file is the functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_flat_test_groups.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_flat_test_groups.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_flat_test_groups'
        >>> callable(module.find_flat_groups)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_flat_test_groups", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


class TestFindFlatGroups:
    """Tests for ``audit.find_flat_groups``."""

    def test_confirms_a_group_by_its_call_target(self, tmp_path: Path) -> None:
        """Three tests sharing a prefix that is also called in the file are grouped."""
        sample = tmp_path / "test_sample.py"
        sample.write_text(
            "from mod import check_data\n"
            "def test_check_data_a(): check_data(1)\n"
            "def test_check_data_b(): check_data(2)\n"
            "def test_check_data_c(): check_data(3)\n",
            encoding="utf-8",
        )
        assert audit.find_flat_groups(sample) == {
            "check_data": ["test_check_data_a", "test_check_data_b", "test_check_data_c"]
        }

    def test_ignores_a_shared_prefix_with_no_matching_call(self, tmp_path: Path) -> None:
        """House-style sentences sharing a topic noun, with no matching callable, are not flagged."""
        sample = tmp_path / "test_sample.py"
        sample.write_text(
            "def test_a_named_worker_count_reaches_the_val_loader(): ...\n"
            "def test_an_auto_chosen_worker_count_is_still_capped(): ...\n"
            "def test_the_val_worker_cap_scales_with_the_side(): ...\n",
            encoding="utf-8",
        )
        assert audit.find_flat_groups(sample) == {}

    def test_ignores_a_group_below_the_minimum_size(self, tmp_path: Path) -> None:
        """Two confirmed members are not enough to enforce, however clean the match."""
        sample = tmp_path / "test_sample.py"
        sample.write_text(
            "from mod import verify_split\n"
            "def test_verify_split_a(): verify_split()\n"
            "def test_verify_split_b(): verify_split()\n",
            encoding="utf-8",
        )
        assert audit.find_flat_groups(sample) == {}

    def test_confirms_via_a_contiguous_token_subsequence(self, tmp_path: Path) -> None:
        """A candidate prefix confirms against a call target it sits inside of, not just prefixes."""
        sample = tmp_path / "test_sample.py"
        sample.write_text(
            "from mod import o2o_rotated_topk\n"
            "def test_rotated_topk_a(): o2o_rotated_topk()\n"
            "def test_rotated_topk_b(): o2o_rotated_topk()\n"
            "def test_rotated_topk_c(): o2o_rotated_topk()\n",
            encoding="utf-8",
        )
        assert "rotated_topk" in audit.find_flat_groups(sample)

    def test_skips_tests_already_inside_a_class(self, tmp_path: Path) -> None:
        """A group already grouped under a class is not module-level and is not reported."""
        sample = tmp_path / "test_sample.py"
        sample.write_text(
            "from mod import check_data\n"
            "class TestCheckData:\n"
            "    def test_a(self): check_data(1)\n"
            "    def test_b(self): check_data(2)\n"
            "    def test_c(self): check_data(3)\n",
            encoding="utf-8",
        )
        assert audit.find_flat_groups(sample) == {}


def test_main_returns_nonzero_when_a_flat_group_survives(tmp_path: Path) -> None:
    """The CLI exit code is 1 when a confirmed flat group is still outside a class."""
    (tmp_path / "test_sample.py").write_text(
        "from mod import check_data\n"
        "def test_check_data_a(): check_data(1)\n"
        "def test_check_data_b(): check_data(2)\n"
        "def test_check_data_c(): check_data(3)\n",
        encoding="utf-8",
    )
    assert audit.main(["--tests-dir", str(tmp_path)]) == 1


def test_main_returns_zero_on_a_clean_tree(tmp_path: Path) -> None:
    """The CLI exit code is 0 when no confirmed flat group survives outside a class."""
    (tmp_path / "test_sample.py").write_text("def test_only_one(): ...\n", encoding="utf-8")
    assert audit.main(["--tests-dir", str(tmp_path)]) == 0


def test_the_live_tests_tree_is_currently_clean() -> None:
    """The real ``tests/`` tree the pre-commit hook scans has no flat group left unclassed.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches any test file, not just a new one.
    """
    assert audit.find_all_flat_groups(REPO_ROOT / "tests") == []


def test_the_live_scripts_tests_tree_is_currently_clean() -> None:
    """The real ``scripts/_tests/`` tree the pre-commit hook scans has no flat group left.

    ``scripts/_tests/`` carries the functional-core tests for ``scripts/`` modules
    (WP-130) and is the hook's second default root, alongside ``tests/`` above.
    """
    assert audit.find_all_flat_groups(REPO_ROOT / "scripts" / "_tests") == []
