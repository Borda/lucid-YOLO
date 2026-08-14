# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-099c YOLO branch of ``lucid-data check`` and its layout dispatch.

AGENTS.md sec. 3 requires a root's layout and counts validated before anything runs against
it, and WP-099b made a YOLO root trainable; these are the cases that make the check worth
running on one. Every root is written into ``tmp_path`` (A26): a ``data.yaml``, zero-byte
image files and the label text beside them. Nothing is downloaded, and no image is ever
decoded — the check counts files by suffix and parses rows, which is the whole point of it
being cheap enough to run before a tier launch.

Covered: a well-formed root passing on its layout alone; both directions of the image/label
pairing; a malformed row and a row naming a class outside the declared ``names``, each
reported with the file and the 1-based line; a ``data.yaml`` that is missing or contradicts
itself; the oriented variant; and the dispatch — an unstated ``--dataset`` inferring the
layout through the WP-099b probe, a root satisfying no convention reporting rather than
raising, and every flag that belongs to another layout being refused.

The COCO and DOTA branches keep their own gates in ``test_coco.py`` and ``test_dota_parse.py``.
No RNG is used: every byte of every fixture is written out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from lucid_yolo.cli import data as data_cli
from lucid_yolo.data import check as check_data
from lucid_yolo.data.layout import DATA_YAML_NAME

#: Class names the fixture root declares, in index order.
_NAMES = ("car", "truck")
#: Detection rows per fixture image: two objects, one, then a background image (an empty
#: label file, which A54 reads as "no objects" rather than as a broken export).
_TRAIN_ROWS: dict[str, list[str]] = {
    "a": ["0 0.5 0.5 0.2 0.2", "1 0.25 0.25 0.1 0.1"],
    "b": ["0 0.75 0.75 0.2 0.2"],
    "c": [],
}
_VAL_ROWS: dict[str, list[str]] = {"d": ["1 0.5 0.5 0.4 0.4"]}
#: One oriented row: a class index and four normalized corners (A56).
_ORIENTED_ROW = "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3"


@dataclass(frozen=True)
class _YoloFixture:
    """A YOLO-shaped fixture root and the counts the check must find in it.

    Attributes:
        root: The dataset root, holding ``data.yaml`` and both split trees.
        images: Image files written across both splits.
        instances: Object rows written across both splits.
        classes: Distinct class ids the rows use.
    """

    root: Path
    images: int
    instances: int
    classes: int


def _write_split(root: Path, split_dir: str, rows: dict[str, list[str]]) -> None:
    """Write one split's images and label files under ``root / split_dir``.

    Args:
        root: The dataset root.
        split_dir: Directory holding the split's ``images``/``labels`` pair.
        rows: Label rows keyed by image stem; an empty list writes an empty label file.
    """
    images_dir, labels_dir = root / split_dir / "images", root / split_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem, lines in rows.items():
        (images_dir / f"{stem}.jpg").touch()
        (labels_dir / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _write_data_yaml(root: Path, body: str) -> Path:
    """Write ``body`` as the root's ``data.yaml`` and return the file path."""
    path = root / DATA_YAML_NAME
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def yolo_root(tmp_path: Path) -> _YoloFixture:
    """Build a well-formed YOLO root: a ``data.yaml``, a ``train`` and a ``valid`` split.

    The validation split is spelled ``valid`` and pointed at by a ``../valid/images`` entry,
    which is the published export's own spelling (A58) and the one no candidate table names —
    so the fixture exercises the reader's resolution rather than the naming convention.
    """
    _write_data_yaml(
        tmp_path,
        "names:\n- car\n- truck\nnc: 2\ntrain: ../train/images\nval: ../valid/images\n",
    )
    _write_split(tmp_path, "train", _TRAIN_ROWS)
    _write_split(tmp_path, "valid", _VAL_ROWS)
    return _YoloFixture(
        root=tmp_path,
        images=len(_TRAIN_ROWS) + len(_VAL_ROWS),
        instances=sum(len(rows) for rows in (*_TRAIN_ROWS.values(), *_VAL_ROWS.values())),
        classes=len(_NAMES),
    )


class TestCheckYoloRoot:
    """The YOLO layout validator over a synthetic root."""

    def test_accepts_a_well_formed_root(self, yolo_root: _YoloFixture) -> None:
        """A root whose layout, pairing and rows are all sound passes on its layout alone.

        No total is stated, which is the way an operator first runs this: the published
        counts of a third-party export are not knowable here, so a correct root has to pass
        without them (the WP-097 lesson, applied to a layout that has no published totals at
        all).
        """
        result = check_data.check_yolo_root(yolo_root.root)

        assert result.ok, result.problems

    def test_reports_each_split_in_the_shared_vocabulary(self, yolo_root: _YoloFixture) -> None:
        """A passing split prints the same ``N images, M instances, K classes`` line as the others.

        One output vocabulary across the three layouts is what lets an operator read a report
        without knowing which branch produced it; the train split's own numbers are asserted
        literally so a miscount cannot hide behind the format.
        """
        report = check_data.format_report(check_data.check_yolo_root(yolo_root.root), yolo_root.root)

        assert "PASS train — 3 images, 3 instances, 2 classes" in report
        assert "PASS val — 1 images, 1 instances, 1 classes" in report

    def test_reports_the_totals_and_the_declared_class_list(self, yolo_root: _YoloFixture) -> None:
        """Unstated totals are notes, and the ``data.yaml``'s own class list is one of them.

        The declared count is what a run's ``model.num_classes`` has to match, so the check
        prints it beside the observed one rather than leaving an operator to open the file.
        """
        result = check_data.check_yolo_root(yolo_root.root)

        assert result.ok, result.problems
        assert any(f"{yolo_root.instances} instances" in note for note in result.notes)
        assert any("data.yaml declares 2 classes: car, truck" in note for note in result.notes)

    def test_enforces_a_total_it_was_given(self, yolo_root: _YoloFixture) -> None:
        """A stated count is still enforced, naming both the wanted and the found number.

        This is how a truncated download or a half-unpacked archive is caught on a layout
        whose correct totals only the operator knows.
        """
        result = check_data.check_yolo_root(yolo_root.root, expected_images=99)

        assert not result.ok
        assert any("expected 99 images" in problem for problem in result.problems)

    def test_names_an_image_without_a_label_file(self, yolo_root: _YoloFixture) -> None:
        """A missing label file fails the split and names the image it belongs to.

        The format carries no image manifest (A54), so an absent label file cannot be told
        from a half-finished export at read time — which is exactly why the pre-run check is
        the place it gets caught, before an epoch raises on one sample.
        """
        (yolo_root.root / "train" / "labels" / "b.txt").unlink()

        result = check_data.check_yolo_root(yolo_root.root)

        assert not result.ok
        assert any("without a label file" in problem and "'b'" in problem for problem in result.splits[0].problems)

    def test_names_a_label_file_without_an_image(self, yolo_root: _YoloFixture) -> None:
        """Pairing is checked in both directions: an orphaned label file is a problem too.

        A label file whose image never arrived is a silently smaller dataset than the one the
        operator provisioned, and nothing downstream ever looks at it.
        """
        (yolo_root.root / "train" / "labels" / "z.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        result = check_data.check_yolo_root(yolo_root.root)

        assert not result.ok
        assert any("without an image" in problem and "'z'" in problem for problem in result.splits[0].problems)

    def test_names_the_file_and_line_of_a_malformed_row(self, yolo_root: _YoloFixture) -> None:
        """A row of the wrong shape is reported with its file and 1-based line, never raised.

        The bad row is placed third so the reported number is a real count rather than a
        constant, and the check must survive it: a report that aborts on the first broken
        file tells an operator to fix one file at a time.
        """
        broken = yolo_root.root / "train" / "labels" / "a.txt"
        broken.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n0 0.5 0.5\n", encoding="utf-8")

        result = check_data.check_yolo_root(yolo_root.root)

        assert not result.ok
        assert any(f"{broken}:3:" in problem for problem in result.splits[0].problems)

    def test_rejects_a_row_naming_a_class_the_data_yaml_does_not(self, yolo_root: _YoloFixture) -> None:
        """A class index outside the declared ``names`` fails, naming the file and the line.

        This is where the class count and the rows are held to agree: the index is 0-based
        into ``names`` and is never remapped (WP-099), so a tree exported against a longer
        class list trains against the wrong labels rather than failing.
        """
        broken = yolo_root.root / "train" / "labels" / "b.txt"
        broken.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        result = check_data.check_yolo_root(yolo_root.root)

        assert not result.ok
        assert any(f"{broken}:1:" in problem and "outside 0..1" in problem for problem in result.splits[0].problems)

    def test_reports_a_missing_data_yaml_as_the_one_problem(self, tmp_path: Path) -> None:
        """Without a ``data.yaml`` there is no class space, so no split is worth checking.

        A labels tree alone is not a dataset this project can read: the file *is* the label
        space, which is why the probe treats it as part of the predicate too.
        """
        result = check_data.check_yolo_root(tmp_path)

        assert not result.ok
        assert result.splits == []
        assert any(DATA_YAML_NAME in problem for problem in result.problems)

    def test_reports_a_data_yaml_that_contradicts_itself(self, yolo_root: _YoloFixture) -> None:
        """An ``nc`` disagreeing with ``names`` is reported, not raised out of the check (A57).

        A file whose own class count is wrong cannot be trusted to state the label space, and
        the operator needs it in the report beside everything else rather than as a traceback.
        """
        _write_data_yaml(yolo_root.root, "names:\n- car\n- truck\nnc: 3\ntrain: ../train/images\n")

        result = check_data.check_yolo_root(yolo_root.root)

        assert not result.ok
        assert any("nc=3 contradicts 2 names" in problem for problem in result.problems)

    def test_names_both_directories_a_split_is_missing(self, tmp_path: Path) -> None:
        """A root with only a ``data.yaml`` names the images and labels directories it wanted.

        With no split entry in the file the naming convention applies, and its verdict has to
        say which two paths were looked for — a root spelled a third way is the usual cause.
        """
        _write_data_yaml(tmp_path, "names:\n- car\nnc: 1\n")

        result = check_data.check_yolo_root(tmp_path, splits=("train",))

        assert not result.ok
        assert any("images directory missing" in problem for problem in result.splits[0].problems)
        assert any("labels directory missing" in problem for problem in result.splits[0].problems)

    def test_names_a_split_entry_that_resolves_nowhere(self, yolo_root: _YoloFixture) -> None:
        """A ``data.yaml`` pointing a split at a directory that is not there fails that split.

        Both readings of the entry are tried (A58), so the message lists every path attempted
        — which is what distinguishes a typo in the file from a split that was never unpacked.
        """
        _write_data_yaml(yolo_root.root, "names:\n- car\n- truck\nnc: 2\ntrain: ../absent/images\n")

        result = check_data.check_yolo_root(yolo_root.root, splits=("train",))

        assert not result.ok
        assert any("resolves to no directory" in problem for problem in result.splits[0].problems)


class TestCheckYoloRootOriented:
    """The oriented reading of a YOLO root, which is declared rather than sniffed."""

    @pytest.fixture
    def oriented_root(self, tmp_path: Path) -> Path:
        """Build a two-split root whose every label row is the nine-field oriented variant."""
        _write_data_yaml(tmp_path, "names:\n- car\n- truck\nnc: 2\ntrain: train/images\nval: valid/images\n")
        _write_split(tmp_path, "train", {"a": [_ORIENTED_ROW]})
        _write_split(tmp_path, "valid", {"b": [_ORIENTED_ROW]})
        return tmp_path

    def test_accepts_oriented_rows_when_told_to_expect_them(self, oriented_root: Path) -> None:
        """Nine-field rows validate under ``oriented=True``, the flag the run itself sets.

        An oriented YOLO export is the layout an OBB run on third-party data would use, and
        it has to be checkable without being rewritten first.
        """
        result = check_data.check_yolo_root(oriented_root, oriented=True)

        assert result.ok, result.problems

    def test_rejects_oriented_rows_read_as_detection(self, oriented_root: Path) -> None:
        """The same root fails under the default reading, naming the file and the line.

        The variant is never sniffed from the field count (WP-099): a mis-stated flag has to
        produce a loud, located failure rather than a plausible reinterpretation of the file.
        """
        result = check_data.check_yolo_root(oriented_root)

        assert not result.ok
        assert any(":1: expected 5 fields" in problem for problem in result.splits[0].problems)

    def test_the_cli_carries_the_flag_through(self, oriented_root: Path) -> None:
        """``lucid-data check --oriented true`` reaches the oriented reading and exits 0.

        The flag is only useful in the spelling an operator types: it is derived from the
        function signature by jsonargparse, so nothing but an end-to-end parse proves that a
        boolean the CLI renders as ``--oriented {true,false}`` arrives where the rows are read.
        """
        assert data_cli.main(["check", "--data_root", str(oriented_root), "--oriented", "true"]) == 0


class TestCheckDatasetDispatch:
    """Which layout ``check_dataset`` validates, and which flags belong to it."""

    def test_infers_the_yolo_layout_of_an_unstated_root(self, yolo_root: _YoloFixture, capsys) -> None:
        """An unstated ``--dataset`` probes the root, so a YOLO tree validates as itself.

        This is the pre-flight for a ``fit`` that has no ``--data.layout`` either, and that
        run dispatches on the same probe: a check defaulting to COCO would report a missing
        ``train2017`` for a root that trains perfectly well.
        """
        code = check_data.check_dataset(yolo_root.root)

        assert code == 0
        assert "PASS: dataset layout valid" in capsys.readouterr().out

    def test_the_cli_infers_it_too(self, yolo_root: _YoloFixture) -> None:
        """``lucid-data check --data_root <yolo root>`` exits 0 with no ``--dataset`` at all.

        The command an operator actually runs is the one that has to work unaided; the flag
        exists for the layouts inference cannot reach.
        """
        assert data_cli.main(["check", "--data_root", str(yolo_root.root)]) == 0

    def test_infers_the_coco_layout_of_a_coco_shaped_root(self, tmp_path: Path, capsys) -> None:
        """A COCO-spelled root still reaches the COCO branch when nothing is stated.

        Asserted through the published per-split count in the report, which only that branch
        knows: the root is deliberately far too small to pass, so what is under test is the
        dispatch and not the verdict.
        """
        (tmp_path / "train2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train2017.json").write_text(json.dumps({"images": []}))

        code = check_data.check_dataset(tmp_path)

        assert code == 1
        assert f"expected {check_data.COCO_TRAIN_COUNT} images" in capsys.readouterr().out

    def test_reports_rather_than_raises_when_no_convention_matches(self, tmp_path: Path, capsys) -> None:
        """An unrecognised root is a FAIL line and exit 1, and points a DOTA operator at the flag.

        The probe raises on this state (A63); a traceback out of the first command run against
        a fresh provisioning is exactly the failure mode WP-097 was about, and the one operator
        inference cannot serve is the one whose layout no probe covers.
        """
        code = check_data.check_dataset(tmp_path)

        out = capsys.readouterr().out
        assert code == 1
        assert "satisfies no dataset convention" in out
        assert f"--dataset {check_data.DOTA_DATASET}" in out

    def test_rejects_an_unknown_layout_name(self) -> None:
        """A ``--dataset`` naming no layout is refused with the list of the ones there are.

        A typo that fell through to inference would validate *something*, and report a pass
        for a layout the operator never asked about.
        """
        with pytest.raises(ValueError, match="unknown dataset"):
            check_data.check_dataset(Path("/nonexistent"), dataset="yolov8")

    def test_rejects_the_oriented_flag_on_another_layout(self) -> None:
        """``oriented`` names a YOLO row grammar, so it is refused elsewhere rather than ignored.

        A flag silently dropped makes the report read as though it had been honoured, which is
        the same reasoning that already governs the count expectations.
        """
        with pytest.raises(ValueError, match="oriented applies to --dataset yolo"):
            check_data.check_dataset(Path("/nonexistent"), dataset="coco", oriented=True)

    def test_rejects_a_count_expectation_aimed_at_coco(self) -> None:
        """The count expectations belong to the two layouts with no published totals.

        COCO's per-split counts are published and are this check's own defaults, so a count
        passed there would be silently ignored.
        """
        with pytest.raises(ValueError, match="apply to --dataset dota"):
            check_data.check_dataset(Path("/nonexistent"), dataset="coco", expected_images=1)

    def test_accepts_a_count_expectation_on_a_yolo_root(self, yolo_root: _YoloFixture) -> None:
        """The same expectations do apply to a YOLO root, whose totals nobody publishes either.

        A third-party export's correct counts are known only to the operator who downloaded
        it, which is the same position a DOTA provisioning is in.
        """
        code = check_data.check_dataset(
            yolo_root.root,
            dataset="yolo",
            expected_images=yolo_root.images,
            expected_instances=yolo_root.instances,
            expected_classes=yolo_root.classes,
        )

        assert code == 0
