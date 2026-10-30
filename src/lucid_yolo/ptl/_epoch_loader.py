# SPDX-License-Identifier: Apache-2.0
"""Queue up to num_workers x prefetch_factor batches of the next epoch while the consumer finishes this one."""

import traceback
from collections.abc import Callable, Iterator
from typing import Any, NoReturn, TypeVar, cast
from weakref import ReferenceType, ref

from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

_T = TypeVar("_T")


class _EpochPrefetchLoader(DataLoader[_T]):
    """Reuse the ordinary persistent-worker iterator after an early epoch reset.

    The caller owns a dedicated shuffle generator and chooses when another full epoch is actually expected. Each
    returned iterator still stops at its own epoch boundary: the early reset queues work, without exposing the next
    epoch through the exhausted iterator. No executor or second worker pool is created. The prepared marker is weak so
    Lightning teardown can release an unused iterator and its queued batches. The loader length is recorded with the
    prepared epoch, and a later ``iter()`` that finds it changed raises instead of replaying a stale epoch.
    """

    def __init__(self, *args: Any, prepare_next: Callable[[], bool], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prepare_next = prepare_next
        self._prepared: ReferenceType[Any] | None = None
        self._generator_receipt: Tensor | None = None
        self._prepared_length: int | None = None
        self._prepare_error: Exception | None = None

    def __iter__(self) -> Any:
        """Yield one complete epoch, preparing its successor before the last yield.

        PyTorch annotates its return with its private iterator class. The wrapper
        is an ordinary iterator, so the private subclass uses Any at that seam.
        """
        if self._prepare_error is not None:
            error = self._prepare_error
            self._prepare_error = None
            raise error
        if not (
            self.persistent_workers
            and self.num_workers > 0
            and self.generator is not None
            and isinstance(self.sampler, RandomSampler)
            and self.sampler.generator is self.generator
        ):
            return super().__iter__()
        iterator: Iterator[_T]
        prepared = self._prepared() if self._prepared is not None else None
        if prepared is not None and prepared is self._iterator:
            if self._prepared_length != len(self):
                self._refuse_stale_epoch()
            iterator = cast(Iterator[_T], prepared)
            self._forget_prepared()
        else:
            if self._generator_receipt is not None and self.generator is not None:
                self.generator.set_state(self._generator_receipt)
            self._forget_prepared()
            iterator = super().__iter__()
        return self._one_epoch(iterator)

    def _forget_prepared(self) -> None:
        """Drop the prepared-epoch marker together with the state recorded beside it."""
        self._prepared = None
        self._generator_receipt = None
        self._prepared_length = None

    def _refuse_stale_epoch(self) -> NoReturn:
        """Rewind the shuffle, then refuse a prepared epoch that was drawn for another length.

        The prepared iterator queued its index lists for the length it saw. Serving it under a longer count would end
        the epoch on a bare ``StopIteration``, and under a shorter one it would cut the epoch silently. The persistent
        workers also keep the dataset they started with, so a changed length cannot be honoured by a reset either. The
        rewind leaves a retried ``iter()`` on the ordinary reset.
        """
        if self._generator_receipt is not None and self.generator is not None:
            self.generator.set_state(self._generator_receipt)
        prepared_length = self._prepared_length
        self._forget_prepared()
        raise RuntimeError(
            f"The next epoch was prepared for {prepared_length} batches, but the loader now has {len(self)}. "
            "Persistent workers keep the dataset they started with; rebuild the loader after resizing it."
        )

    def _one_epoch(self, iterator: Iterator[_T]) -> Iterator[_T]:
        """Keep the old epoch finite after its underlying iterator prepares the next."""
        count = len(self)
        for index in range(count):
            batch = next(iterator)
            if (
                index + 1 == count
                and self.persistent_workers
                and self.num_workers > 0
                and self.generator is not None
                and isinstance(self.sampler, RandomSampler)
                and self.sampler.generator is self.generator
                and self._prepare_next()
            ):
                assert self.generator is not None
                receipt = self.generator.get_state()
                length = len(self)
                try:
                    prepared = super().__iter__()
                except Exception as error:
                    self.generator.set_state(receipt)
                    # The error waits an epoch for the next iter(). Its live traceback would pin
                    # this frame, and the batch it holds, until then, so the frames travel as text.
                    frames = "".join(traceback.format_tb(error.__traceback__))
                    error.add_note(f"Raised while preparing the next epoch, at:\n{frames}")
                    self._prepare_error = error.with_traceback(None)
                except BaseException:
                    self.generator.set_state(receipt)
                    raise
                else:
                    self._prepared = ref(prepared)
                    self._generator_receipt = receipt
                    self._prepared_length = length
            yield batch
