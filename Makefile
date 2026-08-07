# lucid-yolo developer entry points (blueprint section 14.2).
# `make gate` is the single command that decides whether a commit may land.

VENV      := .venv
PY        := $(VENV)/bin/python
UV        := uv
MINOR     ?=
TASK      ?= det
DATA_ROOT ?=

.PHONY: setup lint test precommit gate golden freeze-goldens overfit shapes check-data build clean

setup:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --python $(PY) -e . --group dev
	$(VENV)/bin/pre-commit install

# All linters (ruff check/format, mypy, hygiene hooks) run through pre-commit.
lint: precommit

# Offline unit suite: no network, no dataset, no GPU (gpu/data marks excluded).
# `--doctest-modules src scripts` runs the Examples sections as tests: every public
# function is required to carry one, so leaving them uncollected let four rot in src
# and three more in scripts. Coverage stays scoped to the lucid_yolo package: the
# scripts are entry points driving it, not the surface under measurement.
test:
	@$(PY) -m pytest -m "not gpu and not data" --doctest-modules --cov=lucid_yolo --cov-report=term src scripts tests; \
	status=$$?; if [ $$status -eq 5 ]; then echo "no tests collected yet — passing (pre WP-002)"; exit 0; else exit $$status; fi

precommit:
	$(VENV)/bin/pre-commit run --all-files

# Golden gate suite + frozen-golden regression (harness lands in WP-005).
golden:
	@if [ -f scripts/check_goldens.py ]; then $(PY) scripts/check_goldens.py; else echo "golden harness not yet installed (WP-005) — skipping"; fi

gate: precommit test golden

# Release-time snapshot of the current goldens (release WPs only, D10).
freeze-goldens:
	@test -n "$(MINOR)" || { echo "usage: make freeze-goldens MINOR=0.N"; exit 1; }
	@test -d goldens || { echo "no goldens/ directory to freeze"; exit 1; }
	mkdir -p goldens/frozen/$(MINOR)
	cp goldens/*.json goldens/frozen/$(MINOR)/

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

# Real-dataset layout validation (WP-014; synthetic stand-in per docs/ASSUMPTIONS.md A26).
check-data:
	@test -n "$(DATA_ROOT)" || { echo "usage: make check-data DATA_ROOT=/path/to/coco"; exit 1; }
	$(PY) scripts/check_data.py --data-root $(DATA_ROOT)

# Standard PEP 517 build (setuptools backend): sdist + wheel into dist/.
build:
	rm -rf dist
	$(PY) -m build

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage build dist src/*.egg-info
