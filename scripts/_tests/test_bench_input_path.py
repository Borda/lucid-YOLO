# SPDX-License-Identifier: Apache-2.0
"""The input-path benchmark executes its real CPU work and emits one JSON record."""

import json

import pytest
from scripts.bench_input_path import main


def test_tiny_cpu_benchmark_emits_timings(capsys: pytest.CaptureFixture[str]) -> None:
    """A real mixed empty/nonempty batch and 300 masks yield finite timings."""
    assert (
        main(
            [
                "--device",
                "cpu",
                "--iterations",
                "2",
                "--decode_iterations",
                "1",
                "--warmup",
                "0",
                "--batch_size",
                "2",
                "--instances",
                "1",
                "--img_size",
                "16",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert set(record) == {"device", "transfer_ms_median", "decode_ms_median", "decode_peak_mb"}
    assert record["device"] == "cpu"
    assert record["transfer_ms_median"] > 0
    assert record["decode_ms_median"] > 0
    assert record["decode_peak_mb"] is None
