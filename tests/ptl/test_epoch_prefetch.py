# SPDX-License-Identifier: Apache-2.0
"""Pin finite epochs, worker RNG and cleanup around early input preparation."""

from __future__ import annotations

import multiprocessing
import warnings
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pytorch_lightning.accelerators import CPUAccelerator, CUDAAccelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from lucid_yolo.ptl._epoch_loader import _EpochPrefetchLoader
from lucid_yolo.ptl.datamodule import DetectionDataModule


class _TrainerStub(SimpleNamespace):
    """Keep selector metadata weakly referenceable without pretending to execute CUDA."""


class _Samples(Dataset[torch.Tensor]):
    """Expose sample order together with the worker's actual random stream."""

    def __init__(self, length: int = 11) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index), float(torch.rand(())), float(torch.randint(100, ()))])


@contextmanager
def _pool(prefetch: bool, *, length: int = 11, **kwargs: Any) -> Iterator[DataLoader[torch.Tensor]]:
    """Always close real workers, including on an assertion failure.

    Examples:
        >>> with _pool(False, batch_size=3, num_workers=0) as loader:
        ...     len(loader)
        4
    """
    loader = (
        _EpochPrefetchLoader(_Samples(length), **kwargs)
        if prefetch
        else DataLoader(_Samples(length), **{k: v for k, v in kwargs.items() if k != "prepare_next"})
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
@pytest.mark.parametrize(
    ("batch_size", "length"),
    [
        pytest.param(3, 11, id="several-batches"),
        pytest.param(11, 11, id="single-batch"),
        pytest.param(3, 4, id="tiny-dataset"),
    ],
)
def test_epoch_order_and_worker_randomness_match_native(
    epochs: int, workers: int, drop_last: bool, batch_size: int, length: int
) -> None:
    """Full and partial final batches preserve both random owners and finite exhaustion.

    Only three epochs over two workers ever reach the early reset; the other cells pin the
    unchanged native path. Two datasets stress that reset: one batch per epoch, where the
    successor is prepared on the very first batch, and four samples, fewer batches than
    the workers' queue is deep.
    """
    all_rows: list[list[torch.Tensor]] = []
    states: list[torch.Tensor] = []
    epoch_state = SimpleNamespace(current=0)
    for prefetch in (False, True):
        torch.manual_seed(17)
        global_state = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(19)
        with _pool(
            prefetch,
            length=length,
            batch_size=batch_size,
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


def test_shutdown_discards_prepared_work_without_skipping_shuffle() -> None:
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


@pytest.mark.parametrize("sequential", [False, True])
def test_other_sampler_or_generator_keeps_ordinary_epochs(sequential: bool) -> None:
    """Only the module's own random sampler may be prepared early.

    A sequential sampler and a random one drawing from a generator the loader does not
    hold are both foreign: neither epoch order is the loader's to advance ahead of time.
    """
    dataset = _Samples()
    generator = torch.Generator().manual_seed(31)
    sampler = SequentialSampler(dataset) if sequential else RandomSampler(dataset, generator=generator)

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


def _resize(loader: DataLoader[torch.Tensor], length: int) -> None:
    """Change the main-process dataset's length, as a caller rebuilding its data between epochs would.

    Examples:
        >>> loader = DataLoader(_Samples(), batch_size=3)
        >>> _resize(loader, 14)
        >>> len(loader)
        5
    """
    assert isinstance(loader.dataset, _Samples)
    loader.dataset.length = length


def test_prepared_epoch_of_other_length_raises_then_resets_natively() -> None:
    """A dataset that grows after the early reset is refused, never served a truncated epoch.

    The prepared iterator queued its index lists for 11 samples; replaying it under the
    consumer's longer count would end the epoch on a bare ``StopIteration``. The refusal
    names both lengths and rewinds the shuffle, so a retried ``iter()`` draws the same
    permutation the native loader draws for the grown dataset.
    """
    settings: dict[str, Any] = {
        "batch_size": 3,
        "num_workers": 2,
        "persistent_workers": True,
        "shuffle": True,
    }
    with _pool(False, generator=torch.Generator().manual_seed(59), **settings) as baseline:
        list(baseline)
        _resize(baseline, 14)
        expected = torch.cat(list(baseline))
    with _pool(True, generator=torch.Generator().manual_seed(59), prepare_next=lambda: True, **settings) as loader:
        list(loader)
        _resize(loader, 14)
        with pytest.raises(RuntimeError, match=r"prepared for 4 batches, but the loader now has 5"):
            iter(loader)
        retried = torch.cat(list(loader))
    assert len(retried) == 14
    assert torch.equal(retried[:, 0], expected[:, 0])


class _CountingSamples(_Samples):
    """Signal actual worker reads beyond the first complete epoch."""

    def __init__(self, count: Any, prepared: Any) -> None:
        super().__init__()
        self.count = count
        self.prepared = prepared

    def __getitem__(self, index: int) -> torch.Tensor:
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value > len(self):
                self.prepared.set()
        return super().__getitem__(index)


def test_workers_prepare_before_next_epoch_is_requested() -> None:
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
        assert prepared.wait(30), "the next epoch still waits for its public iterator"
        assert list(iterator) == []
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
            assert all(not worker.is_alive() for worker in loader._iterator._workers)
            loader._iterator = None


@pytest.fixture
def attached(tmp_path: Any) -> Callable[..., DetectionDataModule]:
    """Build a prefetch-requesting module on supported selector metadata, overriding any setting by name."""

    def build(trainer_options: dict[str, Any], **module_options: Any) -> DetectionDataModule:
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
        return module

    return build


def _ignored(gate: str) -> str:
    """Spell the one-time warning an epoch-prefetch request carries when ``gate`` keeps the ordinary loader.

    Examples:
        >>> _ignored("num_workers=0")
        'epoch_prefetch=True is ignored with num_workers=0; training epochs use the ordinary loader'
    """
    return f"epoch_prefetch=True is ignored with {gate}; training epochs use the ordinary loader"


@pytest.mark.parametrize(
    ("trainer_options", "module_options", "enabled", "expected"),
    [
        pytest.param({}, {}, True, [], id="supported"),
        pytest.param({"world_size": 2}, {}, False, [_ignored("world_size=2")], id="distributed"),
        pytest.param(
            {"accelerator": CPUAccelerator()}, {}, False, [_ignored("accelerator=CPUAccelerator")], id="cpu-accelerator"
        ),
        pytest.param({"strategy": object()}, {}, False, [_ignored("strategy=object")], id="other-strategy"),
        pytest.param({"limit_train_batches": 1}, {}, False, [_ignored("limit_train_batches=1")], id="batch-limit"),
        pytest.param(
            {"limit_train_batches": 0.5}, {}, False, [_ignored("limit_train_batches=0.5")], id="fraction-limit"
        ),
        pytest.param({"max_steps": 1}, {}, False, [_ignored("max_steps=1")], id="step-bound"),
        pytest.param({"overfit_batches": 1}, {}, False, [_ignored("overfit_batches=1")], id="overfit"),
        pytest.param(
            {"reload_dataloaders_every_n_epochs": 1},
            {},
            False,
            [_ignored("reload_dataloaders_every_n_epochs=1")],
            id="reload",
        ),
        pytest.param({"max_epochs": -1}, {}, False, [_ignored("max_epochs=-1")], id="unbounded-epochs"),
        pytest.param({}, {"num_workers": 0}, False, [_ignored("num_workers=0")], id="in-process"),
        pytest.param({}, {"persistent_workers": False}, False, [_ignored("persistent_workers=False")], id="fresh-pool"),
        pytest.param({}, {"epoch_prefetch": False}, False, [], id="not-requested"),
    ],
)
def test_datamodule_uses_only_supported_epoch_owner(
    attached: Callable[..., DetectionDataModule],
    trainer_options: dict[str, Any],
    module_options: dict[str, Any],
    enabled: bool,
    expected: list[str],
) -> None:
    """Selector probes establish dispatch and name the first unmet gate; real CUDA fits establish device execution.

    A request the run cannot honour falls back to the ordinary loader, which is
    byte-identical by design, so the warning is the only sign the opt-in did nothing.
    """
    module = attached(trainer_options, **module_options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = module.train_dataloader()
    assert isinstance(loader, _EpochPrefetchLoader) is enabled
    assert [str(warning.message) for warning in caught] == expected


def test_module_built_loader_meets_early_reset_preconditions(attached: Callable[..., DetectionDataModule]) -> None:
    """The supported build hands its loader everything ``__iter__`` checks before it may reset early.

    The prefetching path is byte-identical to the ordinary one by design, so a wiring
    regression -- a dropped persistent pool, a separate sampler, a copied generator --
    would silently turn the opt-in into a no-op that no equivalence test can see.
    """
    loader = attached({}).train_dataloader()
    assert isinstance(loader, _EpochPrefetchLoader)
    assert loader.persistent_workers
    assert loader.num_workers > 0
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.generator is not None
    assert loader.sampler.generator is loader.generator


@pytest.mark.parametrize(
    ("current_epoch", "max_epochs", "should_stop", "prepares"),
    [
        pytest.param(0, 3, False, True, id="first-of-three"),
        pytest.param(1, 3, False, True, id="penultimate"),
        pytest.param(2, 3, False, False, id="final-epoch"),
        pytest.param(0, 1, False, False, id="single-epoch"),
        pytest.param(0, 3, True, False, id="stop-requested"),
    ],
)
def test_prepare_next_stops_at_final_epoch_or_requested_stop(
    attached: Callable[..., DetectionDataModule],
    current_epoch: int,
    max_epochs: int,
    should_stop: bool,
    prepares: bool,
) -> None:
    """The module's predicate prepares a successor only while another configured epoch will run.

    The trainer's epoch and stop flag are read live, after the loader was built, as
    Lightning advances them; preparing past the final epoch or an early stop would
    queue a whole epoch of inputs nobody consumes.
    """
    module = attached({"max_epochs": max_epochs})
    loader = module.train_dataloader()
    module.trainer.current_epoch = current_epoch
    module.trainer.should_stop = should_stop
    assert isinstance(loader, _EpochPrefetchLoader)
    assert loader._prepare_next() is prepares


def test_ignored_prefetch_warns_once_per_module(attached: Callable[..., DetectionDataModule]) -> None:
    """Rebuilding the loader, as a reload or a second fit does, does not repeat the first build's warning."""
    module = attached({}, persistent_workers=False)
    with pytest.warns(UserWarning, match="persistent_workers=False") as record:
        module.train_dataloader()
        module.train_dataloader()
    assert len(record) == 1


def test_unattached_module_keeps_prefetch_warning_unspent(attached: Callable[..., DetectionDataModule]) -> None:
    """A loader built before any trainer is attached stays silent, so the attached build still names its gate.

    Without a trainer the selector cannot judge the run, so the provisional build must
    not spend the once-per-module warning a genuine fallback needs.
    """
    module = attached({}, persistent_workers=False)
    trainer = module.trainer
    module.trainer = None
    with warnings.catch_warnings(record=True) as unattached:
        warnings.simplefilter("always")
        loader = module.train_dataloader()
    module.trainer = trainer
    with pytest.warns(UserWarning, match="persistent_workers=False"):
        module.train_dataloader()
    assert not unattached
    assert not isinstance(loader, _EpochPrefetchLoader)


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


def test_reset_failure_after_worker_resume_keeps_origin_and_shuffle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A speculative reset failing after the persistent iterator restarted still restores the shuffle.

    Here the original reset has already resumed the workers and drawn the next
    permutation before the failure, the half-reset state a dying worker leaves. The
    deferred error must still name the frame it came from, and a later iter() must
    rebuild a complete next epoch in the native order from the restored generator.
    Only the order is pinned: the resumed workers already spent their own random
    streams on the discarded queue, and those live in other processes.
    """
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
    receipts: list[torch.Tensor] = []

    def fail_after_resume(self: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            receipts.append(self.generator.get_state().clone())
            original(self)
            raise RuntimeError("worker died during resume")
        return original(self)

    monkeypatch.setattr(DataLoader, "__iter__", fail_after_resume)
    with _pool(
        True,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(43),
        prepare_next=lambda: True,
    ) as loader:
        first = torch.cat(list(loader))
        restored = loader.generator.get_state().clone()
        with pytest.raises(RuntimeError, match="worker died during resume") as excinfo:
            iter(loader)
        second = torch.cat(list(loader))
    assert torch.equal(first, expected[0])
    assert torch.equal(restored, receipts[0])
    assert "fail_after_resume" in "".join(excinfo.value.__notes__)
    assert torch.equal(second[:, 0], expected[1][:, 0])


def test_interrupted_reset_restores_shuffle_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt during the speculative reset propagates at once, with the shuffle restored.

    An interrupt is not deferred to the next epoch like an ordinary error, but the
    generator must still be back at its pre-attempt state, so a resumed run rebuilds
    the same next epoch the native loader would.
    """
    with _pool(
        False,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(53),
    ) as baseline:
        expected = [torch.cat(list(baseline)) for _ in range(2)]
    original = DataLoader.__iter__
    calls = 0
    receipts: list[torch.Tensor] = []

    def interrupt_second(self: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            receipts.append(self.generator.get_state().clone())
            torch.rand((), generator=self.generator)
            raise KeyboardInterrupt
        return original(self)

    monkeypatch.setattr(DataLoader, "__iter__", interrupt_second)
    with _pool(
        True,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(53),
        prepare_next=lambda: True,
    ) as loader:
        with pytest.raises(KeyboardInterrupt):
            list(loader)
        restored = loader.generator.get_state().clone()
        second = torch.cat(list(loader))
    assert torch.equal(restored, receipts[0])
    assert torch.equal(second, expected[1])


def test_loader_does_not_retain_trainer(tmp_path: Any) -> None:
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


def test_mosaic_change_under_prefetch_warns_once(tmp_path: Any) -> None:
    """A close-mosaic flip under persistent prefetch workers warns once, naming the callback it breaks.

    The workers keep the pipeline copy they started with, so the new probability lands
    only in the main process; the warning is the one signal a CloseMosaicCallback run
    gets that its final epochs still train on mosaics.
    """
    module = DetectionDataModule(tmp_path, 3, 2, "n", persistent_workers=True, epoch_prefetch=True)
    module._train = SimpleNamespace(mosaic_p=1.0)
    with pytest.warns(UserWarning, match="CloseMosaicCallback") as record:
        module.set_mosaic_prob(0.0)
        module.set_mosaic_prob(0.5)
    assert len(record) == 1
    assert module.mosaic_prob == 0.5


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("module_options", "prob"),
    [
        pytest.param({"epoch_prefetch": False}, 0.0, id="no-prefetch"),
        pytest.param({"persistent_workers": False}, 0.0, id="fresh-workers"),
        pytest.param({"num_workers": 0}, 0.0, id="in-process"),
        pytest.param({}, 1.0, id="unchanged"),
    ],
)
def test_mosaic_change_reaching_workers_stays_silent(
    tmp_path: Any, module_options: dict[str, Any], prob: float
) -> None:
    """Only a changed value under the persistent prefetch pair warns; every other route reaches the samples.

    Fresh worker pools copy the pipeline after the change and in-process loading reads
    it directly, so a warning there would be noise; an unchanged value moves nothing.
    """
    settings = {"num_workers": 2, "persistent_workers": True, "epoch_prefetch": True} | module_options
    module = DetectionDataModule(tmp_path, batch_size=3, variant="n", **settings)
    module._train = SimpleNamespace(mosaic_p=1.0)
    module.set_mosaic_prob(prob)
    assert module.mosaic_prob == prob
