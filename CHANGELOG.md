# Changelog

All notable changes to lucid-yolo are documented here, following the Keep a
Changelog convention; versioning is a perpetual 0.x release train — no 1.0 is
ever planned, promised, or tagged — per ADR-002 (docs/DECISIONS.md).

## [Unreleased]

### Added

- Repository scaffold: `src/` package layout, pinned `pyproject.toml`, Makefile
  gate targets, and pre-commit configuration (WP-001).
- Legal and policy baseline: Apache-2.0 LICENSE with Redmon NOTICE attribution,
  a non-affiliation README, and the PROVENANCE / ASSUMPTIONS / DECISIONS /
  AGENTS / ROADMAP governance set (WP-002, WP-003).
- Continuous integration: a pull-request workflow running ruff, strict mypy, the
  offline pytest suite with coverage, a copyleft-dependency license audit, and a
  provenance-carrying commit-trailer validator (WP-004).
- Golden gate: a recompute-and-compare harness with per-metric tolerances and a
  `goldens/frozen/` regression path wired into `make gate` (WP-005).
- Data foundation: seeded synthetic micro-dataset fixtures (boxes, polygons, and
  rotated scenes via fuse-augmentations) and the `Targets` container with its
  type-generic transform API (WP-007, WP-008).
- Model and optimizer primitives: Conv / DWConv / Bottleneck blocks, the CIoU
  regression loss, and Newton-Schulz orthogonalization for the MuSGD optimizer
  (WP-016, WP-024, WP-031).
