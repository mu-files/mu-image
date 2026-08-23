# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Unit tests for ProcessingPipeline error, cancel, and chaining semantics."""

import logging
import time

import pytest

from muimg.processing import ProcessingPipeline

# Failure-path tests raise on purpose; the pipeline logs those at ERROR.
# Keep them out of log_cli output (see pyproject.toml log_cli_level).
logging.getLogger("muimg.processing").setLevel(logging.CRITICAL)


def test_clean_run_is_not_cancelled():
    seen = []

    def producer():
        yield from range(10)

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=seen.append,
        num_workers=2,
        queue_size=8,
    )
    pipeline.run()

    assert sorted(seen) == list(range(10))
    assert pipeline.cancelled is False


def test_clean_run_can_be_reused():
    seen = []
    progress = []

    def producer():
        yield from (1, 2, 3)

    def on_task_done(completed, total):
        progress.append((completed, total))
        return False

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=seen.append,
        num_workers=1,
        on_task_done=on_task_done,
        total_items=3,
    )
    pipeline.run()
    pipeline.run()

    assert sorted(seen) == [1, 1, 2, 2, 3, 3]
    assert pipeline.cancelled is False
    assert progress == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]


def test_writer_failure_raises_and_does_not_hang():
    def producer():
        yield from range(20)

    def writer(_item):
        raise RuntimeError("writer boom")

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=writer,
        num_workers=2,
        queue_size=4,
        writer_queue_size=2,
    )

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="writer boom"):
        pipeline.run()
    assert time.perf_counter() - t0 < 5.0
    assert pipeline.cancelled is True


def test_consumer_failure_raises_and_does_not_hang():
    def producer():
        yield from range(20)

    def consumer(item):
        if item == 3:
            raise ValueError("consumer boom")
        return item

    pipeline = ProcessingPipeline(
        producer,
        consumer=consumer,
        writer=lambda _x: None,
        num_workers=2,
        queue_size=4,
    )

    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="consumer boom"):
        pipeline.run()
    assert time.perf_counter() - t0 < 5.0
    assert pipeline.cancelled is True


def test_producer_failure_raises_and_does_not_hang():
    def producer():
        yield 0
        yield 1
        raise RuntimeError("producer boom")

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=lambda _x: None,
        num_workers=2,
        queue_size=4,
    )

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="producer boom"):
        pipeline.run()
    assert time.perf_counter() - t0 < 5.0
    assert pipeline.cancelled is True


def test_progress_counts_are_complete_and_ordered():
    total = 200
    progress = []

    def producer():
        yield from range(total)

    def on_task_done(completed, _total):
        progress.append(completed)
        return False

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=lambda _x: None,
        num_workers=4,
        on_task_done=on_task_done,
        total_items=total,
    )
    pipeline.run()

    # Initial 0 call, then every count exactly once, in order.
    assert progress == list(range(total + 1))


def test_failed_run_does_not_poison_next_run():
    seen = []
    fail_first_run = [True]

    def producer():
        yield from range(10)

    def writer(item):
        if fail_first_run[0]:
            fail_first_run[0] = False
            raise RuntimeError("first run boom")
        seen.append(item)

    pipeline = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=writer,
        num_workers=2,
        queue_size=2,
        writer_queue_size=2,
    )

    with pytest.raises(RuntimeError, match="first run boom"):
        pipeline.run()
    assert pipeline.cancelled is True

    # Second run must see exactly the fresh items: no stale tasks
    # (duplicates) and no stale sentinels (truncation) from the first run.
    seen.clear()
    pipeline.run()
    assert sorted(seen) == list(range(10))
    assert pipeline.cancelled is False


def test_none_producer_with_workers_raises_instead_of_hanging():
    pipeline = ProcessingPipeline(
        None,
        consumer=lambda x: x,
        num_workers=2,
    )

    with pytest.raises(ValueError, match="producer is required"):
        pipeline.run()


def test_none_producer_sync_mode_is_a_noop():
    pipeline = ProcessingPipeline(
        None,
        consumer=lambda x: x,
        num_workers=0,
    )
    pipeline.run()

    assert pipeline.cancelled is False


def test_chained_pipeline_clean_run():
    seen = []
    upstream_seen = []
    upstream_writer = upstream_seen.append

    def producer():
        yield from range(5)

    upstream = ProcessingPipeline(
        producer,
        consumer=lambda x: x * 10,
        writer=upstream_writer,
        num_workers=1,
        queue_size=2,
    )
    downstream = ProcessingPipeline(
        upstream,
        consumer=lambda x: x + 1,
        writer=seen.append,
        num_workers=2,
        queue_size=2,
    )
    downstream.run()

    assert sorted(upstream_seen) == [0, 10, 20, 30, 40]
    assert sorted(seen) == [1, 11, 21, 31, 41]
    assert downstream.cancelled is False
    # Upstream writer must be restored so the chain is reusable.
    assert upstream.writer is upstream_writer


def test_chained_upstream_failure_raises_and_does_not_hang():
    def producer():
        yield from range(20)

    def consumer(item):
        if item == 3:
            raise RuntimeError("upstream boom")
        return item

    upstream = ProcessingPipeline(
        producer,
        consumer=consumer,
        writer=lambda _x: None,
        num_workers=1,
        queue_size=2,
    )
    downstream = ProcessingPipeline(
        upstream,
        consumer=lambda x: x,
        writer=lambda _x: None,
        num_workers=2,
        queue_size=2,
    )

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="upstream boom"):
        downstream.run()
    assert time.perf_counter() - t0 < 5.0
    assert downstream.cancelled is True


def test_chained_downstream_failure_does_not_hang():
    def producer():
        yield from range(50)

    def writer(item):
        if item >= 2:
            raise RuntimeError("downstream boom")

    upstream = ProcessingPipeline(
        producer,
        consumer=lambda x: x,
        writer=lambda _x: None,
        num_workers=1,
        queue_size=2,
    )
    downstream = ProcessingPipeline(
        upstream,
        consumer=lambda x: x,
        writer=writer,
        num_workers=2,
        queue_size=1,
        writer_queue_size=1,
    )

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="downstream boom"):
        downstream.run()
    assert time.perf_counter() - t0 < 5.0
    assert downstream.cancelled is True
