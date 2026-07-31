# lit-yolo developer entry points (blueprint section 14.2).
# `make gate` is the single command that decides whether a commit may land.

VENV      := .venv
PY        := $(VENV)/bin/python
UV        := uv
MINOR     ?=
TASK      ?= det

.PHONY: setup lint test precommit gate golden freeze-goldens overfit check-data clean

setup:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --python $(PY) -e ".[dev]"
	$(VENV)/bin/pre-commit install

# All linters (ruff check/format, mypy, hygiene hooks) run through pre-commit.
lint: precommit

# Offline unit suite: no network, no dataset, no GPU (gpu/data marks excluded).
test:
	@$(PY) -m pytest -m "not gpu and not data" --cov=lit_yolo --cov-report=term; \
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
overfit:
	@echo "overfit-$(TASK) integration entry lands with WP-040"; exit 1

# Real-dataset layout validation (WP-014; synthetic stand-in per docs/ASSUMPTIONS.md A26).
check-data:
	@echo "check-data lands with WP-014"; exit 1

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage build dist src/*.egg-info
