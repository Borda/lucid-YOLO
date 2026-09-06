# lucid-yolo developer entry points (blueprint section 14.2).
# `make gate` is the single command that decides whether a commit may land.

VENV      := .venv
PY        := $(VENV)/bin/python
UV        := uv
MINOR     ?=
TASK      ?= det
DATA_ROOT ?=
DATASET   ?=

TAG       ?=

.PHONY: setup lint test test-gpu precommit gate gate-gpu golden golden-gpu freeze-goldens overfit shapes check-data build dist-pypi docs docs-serve clean

setup:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --python $(PY) -e . --group dev --group typing
	$(VENV)/bin/pre-commit install

# All linters (ruff check/format, mypy, hygiene hooks) run through pre-commit.
lint: precommit

# Offline unit suite: no network, no dataset, no GPU (gpu/data marks excluded).
# `--doctest-modules src scripts` runs the Examples sections as tests: every public
# function is required to carry one, so leaving them uncollected let four rot in src
# and three more in scripts. Coverage stays scoped to the lucid_yolo package: the
# scripts are entry points driving it, not the surface under measurement.
#
# Pytest's exit 5 (nothing collected) used to be swallowed here and reported as a
# pass, a scaffold from before WP-002 when there were no tests to collect. There are
# 2668, and WP-168 made this target what CI runs, so the swallow would have meant a
# job that collected nothing reporting green — the failure this row exists to close.
test:
	$(PY) -m pytest -m "not gpu and not data" --doctest-modules --cov=lucid_yolo --cov-report=term src scripts tests

precommit:
	$(VENV)/bin/pre-commit run --all-files

# Golden gate suite + frozen-golden regression (harness lands in WP-005).
golden:
	@if [ -f scripts/check_goldens.py ]; then $(PY) scripts/check_goldens.py; else echo "golden harness not yet installed (WP-005) — skipping"; fi

gate: precommit test golden

# The half `make gate` cannot run: tests marked gpu or data, which need an
# accelerator and a generated dataset. Excluded from the offline gate by design,
# not by accident — but nothing ran them on a schedule either, which is how the
# frozen detection overfit golden drifted for six days while WP-078 changed the
# objective underneath it. This target was called "that schedule" while nothing
# invoked it; the schedule is .github/workflows/gate-gpu.yml (nightly plus manual
# dispatch), which runs `gate-gpu` below on the self-hosted runner named by the
# GPU_RUNNER_LABEL repository variable.
test-gpu:
	$(PY) -m pytest -m "gpu or data" tests scripts

# Recompute every golden including the goldens/gpu/ subtree. These producers
# retrain models rather than reading a file: overfit_micro_det and
# overfit_micro_seg are minutes each, shapes_regression_det trains on 1800
# generated scenes and is the long pole.
golden-gpu:
	$(PY) scripts/check_goldens.py --include-gpu

# Full accelerator gate. Runs the overfit pipeline twice on purpose — the marked
# tests assert the WP-040/087 acceptance thresholds, the goldens assert the frozen
# values, and it was the second of those that silently went stale. Run before any
# release tier and after any change to the loss, the assignment or the decode.
gate-gpu: test-gpu golden-gpu

# Release-time snapshot of the current goldens (release WPs only, D10).
# Skips any golden marked "freezable": false — a generator-derived golden can
# never satisfy "current code still satisfies every frozen value" once its
# external dependency moves, so it must never enter goldens/frozen/ (WP-154c).
freeze-goldens:
	@test -n "$(MINOR)" || { echo "usage: make freeze-goldens MINOR=0.N"; exit 1; }
	@test -d goldens || { echo "no goldens/ directory to freeze"; exit 1; }
	$(PY) scripts/freeze_goldens.py $(MINOR)

# Overfit-100 integration goldens (WP-040/054/064; needs accelerator).
# Trains an n-scale detector on a fixed ~100-image synthetic slice and gates on
# train recall@0.5 >= 0.95; add FREEZE=1 to (re)write goldens/gpu/overfit_micro_det.json.
overfit:
	$(PY) scripts/overfit_micro.py --task $(TASK) $(if $(FREEZE),--freeze,)

# Synthetic-shapes generalization regression (WP-083; needs accelerator, ~2 min).
# Trains on 1800 generated scenes and scores the held-out 200 through both decode
# paths; add FREEZE=1 to (re)write goldens/gpu/shapes_regression_det.json.
shapes:
	$(PY) scripts/shapes_regression.py --task $(TASK) $(if $(FREEZE),--freeze,)

# Real-dataset layout validation (WP-014/056/099c; synthetic stand-in per docs/ASSUMPTIONS.md A26).
# DATASET names the layout and is optional: unset, the root is probed (coco or yolo), which
# is what `lucid-yolo fit` itself does — passing a default here would make the pre-flight
# check a different question than the run. `DATASET=dota` names the one layout no probe
# covers. A convenience wrapper only — the check ships as `lucid-data check`, so a remote
# tier run needs no checkout (WP-096).
check-data:
	@test -n "$(DATA_ROOT)" || { echo "usage: make check-data DATA_ROOT=/path/to/data [DATASET=coco|dota|yolo]"; exit 1; }
	$(PY) -m lucid_yolo.cli.data check --data_root $(DATA_ROOT) $(if $(DATASET),--dataset $(DATASET),)

# Standard PEP 517 build (setuptools backend): sdist + wheel into dist/. The README
# ships exactly as committed — relative links, correct in a checkout, dead on PyPI.
build:
	rm -rf dist
	$(PY) -m build

# Named for where the artifacts are shaped to go, not for where they can be sent: D20's git
# pin makes this distribution unpublishable to PyPI. What the rewrite still buys is a long
# description whose links resolve for a reader outside the repository.
# TAG names the tag the README's links are
# pinned to, and is the flag that makes this a release build rather than a local one; the
# rewrite is reverted whether the build succeeds or fails, so the tree is left as found.
dist-pypi:
	@test -n "$(TAG)" || { echo "usage: make dist-pypi TAG=v0.MINOR.PATCH"; exit 1; }
	rm -rf dist
	@$(PY) scripts/absolutize_readme.py --ref $(TAG) || exit 1; \
	status=0; $(PY) -m build || status=$$?; \
	$(PY) scripts/absolutize_readme.py --revert || exit 1; \
	exit $$status
# MkDocs Material site over docs/ into site/. The docs group is deliberately outside
# `make setup`: it is a publishing toolchain, no gate imports it, and a contributor who
# never builds the site never installs the tree. Install it on demand with
# `uv pip install --python $(PY) --group docs`, which also brings that tree under the
# pre-commit licence audit — the audit scans the environment, not the diff.
# --strict is the whole value of building locally: it fails on a link to a page that
# does not exist and on a page the nav never lists, and a register nobody can navigate
# to is one nobody reads.
docs:
	$(PY) -m mkdocs build --strict

# Live-reload preview on http://127.0.0.1:8000 for editing prose; not a gate.
docs-serve:
	$(PY) -m mkdocs serve

# `tests/fixtures/_generated` is the synthetic micro-dataset cache. It is gitignored, so
# CI is always cold and a contributor's machine is always warm; leaving it out of `clean`
# left "start from nothing" meaning two different things on the two (M-45). The cache is
# self-invalidating on a fingerprint mismatch, so removing it here is a belt to that
# brace, not the mechanism.
clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage build dist site src/*.egg-info tests/fixtures/_generated
