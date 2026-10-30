# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the dataset-reader tests that compare cached and uncached samples.

``tests/conftest.py`` puts this directory on ``sys.path``, so tests import it by bare name:
``from reader_sample_helpers import assert_same_sample``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from lucid_yolo.data import _image as image_data
from lucid_yolo.data.targets import Targets


def assert_same_sample(left: tuple[torch.Tensor, Targets], right: tuple[torch.Tensor, Targets]) -> None:
    """Assert two reader samples agree on the image and on every ``Targets`` field.

    Examples:
        >>> sample = (torch.zeros(3, 2, 2), Targets.empty())
        >>> assert_same_sample(sample, sample)
    """
    assert torch.equal(left[0], right[0]), "the images differ"
    for field in dataclasses.fields(left[1]):
        got, want = getattr(left[1], field.name), getattr(right[1], field.name)
        if isinstance(got, list):
            assert len(got) == len(want), f"{field.name} has a different length"
            assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True)), f"{field.name} differs"
        else:
            assert torch.equal(got, want), f"{field.name} differs"


def count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route every image decode through a counter and return the list it appends each path to.

    The counter wraps the real decoder, so cached or uncached reads still return real pixels.

    Args:
        monkeypatch: The test's patcher, restored at teardown.

    Examples:
        >>> callable(count_decodes)  # needs the test's live monkeypatch fixture
        True
    """
    calls: list[str] = []
    incumbent = image_data.read_image

    def counted_read_image(path: str, mode: object) -> torch.Tensor:
        calls.append(path)
        return incumbent(path, mode)  # type: ignore[arg-type]

    monkeypatch.setattr(image_data, "read_image", counted_read_image)
    return calls
