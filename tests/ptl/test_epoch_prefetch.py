# SPDX-License-Identifier: Apache-2.0
"""Pin finite epochs, worker RNG and cleanup around early input preparation."""

from __future__ import annotations

import multiprocessing
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from lucid_yolo.ptl._epoch_loader import _EpochPrefetchLoader
from lucid_yolo.ptl.datamodule import DetectionDataModule


class _TrainerStub(SimpleNamespace):
    """Keep selector metadata weakly referenceable without pretending to execute CUDA."""


class _Samples(Dataset[torch.Tensor]):
    """Expose sample order together with the worker's actual random stream."""

    def __len__(self) -> int:
        return 11

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index), float(torch.rand(())), float(torch.randint(100, ()))])


@contextmanager
def _pool(prefetch: bool, **kwargs: Any) -> Iterator[DataLoader[torch.Tensor]]:
    """Always close real workers, including on an assertion failure.

    Examples:
        >>> with _pool(False, batch_size=3, num_workers=0) as loader:
        ...     len(loader)
        4
    """
    loader = (
        _EpochPrefetchLoader(_Samples(), **kwargs)
        if prefetch
        else DataLoader(_Samples(), **{k: v for k, v in kwargs.items() if k != "prepare_next"})
    )
    try:
        yield loader
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
            loader._iterator = None


@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
def test_epoch_order_and_worker_randomness_match_native(epochs: int, workers: int, drop_last: bool) -> None:
    """Full and partial final batches preserve both random owners and finite exhaustion."""
    all_rows: list[list[torch.Tensor]] = []
    states: list[torch.Tensor] = []
    epoch_state = SimpleNamespace(current=0)
    for prefetch in (False, True):
        torch.manual_seed(17)
        global_state = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(19)
        with _pool(
            prefetch,
            batch_size=3,
            num_workers=workers,
            persistent_workers=workers > 0,
            generator=generator,
            shuffle=True,
            drop_last=drop_last,
            prepare_next=lambda: epoch_state.current < epochs - 1,
        ) as loader:
            rows = []
            for epoch in range(epochs):
                epoch_state.current = epoch
                iterator = iter(loader)
                rows.append(torch.cat(list(iterator)))
                assert list(iterator) == []
            states.append(generator.get_state().clone())
            all_rows.append(rows)
        if workers:
            assert torch.equal(global_state, torch.get_rng_state())
    assert len(all_rows[0]) == len(all_rows[1]) == epochs
    assert all(torch.equal(a, b) for a, b in zip(*all_rows, strict=True))
    assert torch.equal(*states)


def test_shutdown_discards_prepared_work_without_skipping_a_shuffle() -> None:
    """An unused early epoch must not advance the next fresh worker pool's RNG."""
    outputs = []
    for prefetch in (False, True):
        generator = torch.Generator().manual_seed(23)
        with _pool(
            prefetch,
            batch_size=3,
            num_workers=2,
            persistent_workers=True,
            generator=generator,
            shuffle=True,
            prepare_next=lambda: True,
        ) as loader:
            first = torch.cat(list(loader))
            assert loader._iterator is not None
            loader._iterator._shutdown_workers()
            loader._iterator = None
            second = torch.cat(list(loader))
            outputs.append((first, second))
    assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))


def test_partial_epoch_keeps_native_restart_order() -> None:
    """Stopping before the final batch leaves the ordinary DataLoader reset in charge."""
    outputs = []
    for prefetch in (False, True):
        with _pool(
            prefetch,
            batch_size=3,
            num_workers=2,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(29),
            shuffle=True,
            prepare_next=lambda: True,
        ) as loader:
            iterator = iter(loader)
            first = next(iterator)
            second = torch.cat(list(loader))
            outputs.append((first, second))
    assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))


@pytest.mark.parametrize("owned", [False, True])
def test_other_sampler_or_generator_keeps_ordinary_epochs(owned: bool) -> None:
    """Only the module's own random sampler may be prepared early."""
    dataset = _Samples()
    generator = torch.Generator().manual_seed(31)
    sampler = SequentialSampler(dataset) if owned else RandomSampler(dataset, generator=generator)

    def refuse() -> bool:
        raise AssertionError("this sampler has a different epoch owner")

    with _pool(
        True,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(37),
        sampler=sampler,
        prepare_next=refuse,
    ) as loader:
        assert len(list(loader)) == 4


class _CountingSamples(_Samples):
    """Signal actual worker reads beyond the first complete epoch."""

    def __init__(self, count: Any, prepared: Any) -> None:
        self.count = count
        self.prepared = prepared

    def __getitem__(self, index: int) -> torch.Tensor:
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value > len(self):
                self.prepared.set()
        return super().__getitem__(index)


def test_workers_prepare_before_the_next_epoch_is_requested() -> None:
    """An IPC signal from real reads pins overlap, rather than just wrapper metadata."""
    context = multiprocessing.get_context()
    count = context.Value("i", 0)
    prepared = context.Event()
    loader = _EpochPrefetchLoader(
        _CountingSamples(count, prepared),
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(41),
        prepare_next=lambda: True,
    )
    try:
        iterator = iter(loader)
        for _ in range(len(loader)):
            next(iterator)
        assert prepared.wait(5), "the next epoch still waits for its public iterator"
        assert list(iterator) == []
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
            assert all(not worker.is_alive() for worker in loader._iterator._workers)
            loader._iterator = None


@pytest.mark.parametrize(
    ("trainer_options", "module_options", "enabled"),
    [
        ({}, {}, True),
        ({"world_size": 2}, {}, False),
        ({"strategy": object()}, {}, False),
        ({"limit_train_batches": 1}, {}, False),
        ({"limit_train_batches": 0.5}, {}, False),
        ({"max_steps": 1}, {}, False),
        ({"overfit_batches": 1}, {}, False),
        ({"reload_dataloaders_every_n_epochs": 1}, {}, False),
        ({"max_epochs": -1}, {}, False),
        ({}, {"num_workers": 0}, False),
        ({}, {"persistent_workers": False}, False),
        ({}, {"epoch_prefetch": False}, False),
    ],
)
def test_datamodule_uses_only_the_supported_epoch_owner(
    tmp_path: Any, trainer_options: dict[str, Any], module_options: dict[str, Any], enabled: bool
) -> None:
    """Selector probes establish dispatch only; real CUDA fits establish device execution."""
    settings = {"num_workers": 2, "persistent_workers": True, "epoch_prefetch": True} | module_options
    module = DetectionDataModule(tmp_path, batch_size=3, variant="n", **settings)
    module._train = _Samples()
    module.trainer = _TrainerStub(
        **(
            {
                "accelerator": CUDAAccelerator(),
                "strategy": SingleDeviceStrategy(device="cpu"),
                "world_size": 1,
                "limit_train_batches": 1.0,
                "max_steps": -1,
                "overfit_batches": 0,
                "reload_dataloaders_every_n_epochs": 0,
                "max_epochs": 3,
                "current_epoch": 0,
                "should_stop": False,
            }
            | trainer_options
        )
    )
    assert isinstance(module.train_dataloader(), _EpochPrefetchLoader) is enabled


def test_future_reset_failure_preserves_last_batch_and_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed speculative reset belongs to the next epoch and must restore its RNG."""
    with _pool(
        False,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(43),
    ) as baseline:
        expected = [torch.cat(list(baseline)) for _ in range(2)]
    original = DataLoader.__iter__
    calls = 0

    def fail_second(self: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            torch.rand((), generator=self.generator)
            raise RuntimeError("future reset failed")
        return original(self)

    monkeypatch.setattr(DataLoader, "__iter__", fail_second)
    with _pool(
        True,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(43),
        prepare_next=lambda: True,
    ) as loader:
        assert torch.equal(torch.cat(list(loader)), expected[0])
        with pytest.raises(RuntimeError, match="future reset failed"):
            iter(loader)
        assert torch.equal(torch.cat(list(loader)), expected[1])


def test_loader_does_not_retain_the_trainer(tmp_path: Any) -> None:
    """Keeping the loader must not keep its training model alive through the predicate."""
    module = DetectionDataModule(tmp_path, 3, 2, "n", persistent_workers=True, epoch_prefetch=True)
    module._train = _Samples()
    module.trainer = _TrainerStub(
        accelerator=CUDAAccelerator(),
        strategy=SingleDeviceStrategy(device="cpu"),
        world_size=1,
        limit_train_batches=1.0,
        max_steps=-1,
        overfit_batches=0,
        reload_dataloaders_every_n_epochs=0,
        max_epochs=3,
        current_epoch=0,
        should_stop=False,
    )
    owner = weakref.ref(module.trainer)
    loader = module.train_dataloader()
    module.trainer = None
    assert owner() is None
    assert isinstance(loader, _EpochPrefetchLoader)
    assert not loader._prepare_next()


def test_lightning_reset_releases_unused_prepared_iterator() -> None:
    """An early stop must not retain the old iterator after Lightning closes its pool."""
    with _pool(
        True,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(47),
        prepare_next=lambda: True,
    ) as loader:
        combined = CombinedLoader(loader, "max_size_cycle")
        for batch in combined:
            del batch
        owner = weakref.ref(loader._iterator)
        assert owner() is not None
        combined.reset()
        assert loader._iterator is None
        assert owner() is None
