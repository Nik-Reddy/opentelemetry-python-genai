# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=abstract-class-instantiated

import asyncio
import inspect
from unittest.mock import patch

import pytest

from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)


def test_stream_wrapper_abstract_method_signatures_match():
    method_names = (
        "_process_chunk",
        "_on_stream_end",
        "_on_stream_error",
        "_handle_process_chunk_error",
    )

    for method_name in method_names:
        assert inspect.signature(
            getattr(SyncStreamWrapper, method_name)
        ) == inspect.signature(getattr(AsyncStreamWrapper, method_name))


class _FakeSyncStream:
    def __init__(self, chunks=None, error=None, close_error=None):
        self._chunks = list(chunks or [])
        self._error = error
        self._close_error = close_error
        self.close_count = 0
        self.extra_attribute = "passthrough"

    def __iter__(self):
        return self

    def __next__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopIteration

    def close(self):
        self.close_count += 1
        if self._close_error:
            raise self._close_error

    def __len__(self):
        return 42


class _FakeSyncIterable:
    def __init__(self, chunks=None):
        self.iterator = iter(chunks or [])
        self.close_count = 0

    def __iter__(self):
        return self.iterator

    def close(self):
        self.close_count += 1


class _TestSyncStreamWrapper(SyncStreamWrapper):
    def __init__(self, stream):
        super().__init__(stream)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1

    def _on_stream_error(self, error):
        self._self_failures.append(error)


class _FailingSyncProcessStreamWrapper(_TestSyncStreamWrapper):
    def _process_chunk(self, chunk):
        raise ValueError("instrumentation failed")


class _FailingSyncStopStreamWrapper(_TestSyncStreamWrapper):
    def _on_stream_end(self):
        self._self_stop_count += 1
        raise ValueError("instrumentation failed")


class _FailingSyncFailStreamWrapper(_TestSyncStreamWrapper):
    def _on_stream_error(self, error):
        self._self_failures.append(error)
        raise ValueError("instrumentation failed")


def test_sync_stream_wrapper_processes_chunks_and_stops():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    assert next(wrapper) == "chunk"
    assert wrapper._self_processed == ["chunk"]

    try:
        next(wrapper)
    except StopIteration:
        pass

    assert wrapper._self_stop_count == 1


def test_sync_stream_wrapper_processes_iterables():
    stream = _FakeSyncIterable(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    assert next(wrapper) == "chunk"
    assert wrapper._self_processed == ["chunk"]

    with pytest.raises(StopIteration):
        next(wrapper)

    assert wrapper._self_stop_count == 1


def test_sync_stream_wrapper_fails_stream_errors():
    error = ValueError("boom")
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream(error=error))

    try:
        next(wrapper)
    except ValueError:
        pass

    assert wrapper._self_failures == [error]


def test_sync_stream_wrapper_close_stops_once():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    wrapper.close()
    wrapper.close()

    assert stream.close_count == 2
    assert wrapper._self_stop_count == 1
    assert not wrapper._self_failures


def test_sync_stream_wrapper_close_fails_with_close_error():
    error = RuntimeError("close failure")
    wrapper = _TestSyncStreamWrapper(
        _FakeSyncStream(chunks=["chunk"], close_error=error)
    )

    with pytest.raises(RuntimeError, match="close failure"):
        wrapper.close()

    assert wrapper._self_failures == [error]
    assert wrapper._self_stop_count == 0


def test_sync_stream_wrapper_exit_closes_and_propagates_user_errors():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)
    error = RuntimeError("user failure")

    assert wrapper.__exit__(RuntimeError, error, None) is False

    assert stream.close_count == 1
    assert wrapper._self_stop_count == 0
    assert wrapper._self_failures == [error]


def test_sync_stream_wrapper_exit_keeps_user_error_when_close_fails():
    close_error = RuntimeError("close failure")
    stream = _FakeSyncStream(chunks=["chunk"], close_error=close_error)
    wrapper = _TestSyncStreamWrapper(stream)
    error = RuntimeError("user failure")

    assert wrapper.__exit__(RuntimeError, error, None) is False

    assert stream.close_count == 1
    assert wrapper._self_failures == [error]
    assert wrapper._self_stop_count == 0


def test_sync_stream_wrapper_swallows_finalize_errors():
    wrapper = _FailingSyncStopStreamWrapper(_FakeSyncStream())

    wrapper.close()
    wrapper.close()

    assert wrapper._self_stop_count == 1


def test_sync_stream_wrapper_swallows_failure_finalize_errors():
    close_error = RuntimeError("close failure")
    stream = _FakeSyncStream(close_error=close_error)
    wrapper = _FailingSyncFailStreamWrapper(stream)

    with pytest.raises(RuntimeError, match="close failure"):
        wrapper.close()
    stream._close_error = None
    wrapper.close()

    assert wrapper._self_failures == [close_error]


def test_sync_stream_wrapper_swallows_stop_iteration_finalize_errors():
    wrapper = _FailingSyncStopStreamWrapper(_FakeSyncStream())

    with pytest.raises(StopIteration):
        next(wrapper)


def test_sync_stream_wrapper_preserves_stream_error_when_finalize_fails():
    error = RuntimeError("stream failure")
    wrapper = _FailingSyncFailStreamWrapper(_FakeSyncStream(error=error))

    with pytest.raises(RuntimeError, match="stream failure"):
        next(wrapper)


def test_sync_stream_wrapper_getattr_passthrough():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    assert wrapper.extra_attribute == "passthrough"


def test_sync_stream_wrapper_exposes_wrapped_stream():
    stream = _FakeSyncStream()
    wrapper = _TestSyncStreamWrapper(stream)

    assert getattr(wrapper, "__wrapped__") is stream


def test_sync_stream_wrapper_magic_method_passthrough():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    assert len(wrapper) == 42


def test_sync_stream_wrapper_stop_iteration_does_not_double_finalize():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    with pytest.raises(StopIteration):
        next(wrapper)
    wrapper.close()

    assert wrapper._self_stop_count == 1
    assert not wrapper._self_failures


def test_sync_stream_wrapper_swallows_process_chunk_errors():
    wrapper = _FailingSyncProcessStreamWrapper(
        _FakeSyncStream(chunks=["chunk"])
    )

    assert next(wrapper) == "chunk"
    assert not wrapper._self_failures


class _FakeAsyncStream:
    def __init__(self, chunks=None, error=None, close_error=None):
        self._chunks = list(chunks or [])
        self._error = error
        self._close_error = close_error
        self.close_count = 0
        self.extra_attribute = "passthrough"

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopAsyncIteration

    async def close(self):
        self.close_count += 1
        if self._close_error:
            raise self._close_error

    def __len__(self):
        return 42


class _FakeAsyncIterable:
    def __init__(self, chunks=None):
        self.iterator = _FakeAsyncStream(chunks=chunks)
        self.close_count = 0

    def __aiter__(self):
        return self.iterator

    async def close(self):
        self.close_count += 1


class _TestAsyncStreamWrapper(AsyncStreamWrapper):
    def __init__(self, stream):
        super().__init__(stream)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1

    def _on_stream_error(self, error):
        self._self_failures.append(error)


class _FailingAsyncProcessStreamWrapper(_TestAsyncStreamWrapper):
    def _process_chunk(self, chunk):
        raise ValueError("instrumentation failed")


class _FailingAsyncStopStreamWrapper(_TestAsyncStreamWrapper):
    def _on_stream_end(self):
        self._self_stop_count += 1
        raise ValueError("instrumentation failed")


class _FailingAsyncFailStreamWrapper(_TestAsyncStreamWrapper):
    def _on_stream_error(self, error):
        self._self_failures.append(error)
        raise ValueError("instrumentation failed")


def test_async_stream_wrapper_processes_chunks_and_stops():
    async def exercise():
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream(chunks=["chunk"]))

        assert await anext(wrapper) == "chunk"
        assert wrapper._self_processed == ["chunk"]

        try:
            await anext(wrapper)
        except StopAsyncIteration:
            pass

        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_processes_async_iterables():
    async def exercise():
        stream = _FakeAsyncIterable(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)

        assert await anext(wrapper) == "chunk"
        assert wrapper._self_processed == ["chunk"]

        with pytest.raises(StopAsyncIteration):
            await anext(wrapper)

        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_fails_stream_errors():
    async def exercise():
        error = ValueError("boom")
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream(error=error))

        with pytest.raises(ValueError):
            await anext(wrapper)

        assert wrapper._self_failures == [error]

    asyncio.run(exercise())


def test_async_stream_wrapper_close_stops_once():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)

        await wrapper.close()
        await wrapper.close()

        assert stream.close_count == 2
        assert wrapper._self_stop_count == 1
        assert not wrapper._self_failures

    asyncio.run(exercise())


def test_async_stream_wrapper_close_fails_with_close_error():
    async def exercise():
        error = RuntimeError("close failure")
        wrapper = _TestAsyncStreamWrapper(
            _FakeAsyncStream(chunks=["chunk"], close_error=error)
        )

        with pytest.raises(RuntimeError, match="close failure"):
            await wrapper.close()

        assert wrapper._self_failures == [error]
        assert wrapper._self_stop_count == 0

    asyncio.run(exercise())


def test_async_stream_wrapper_exit_closes_and_propagates_user_errors():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)
        error = RuntimeError("user failure")

        assert await wrapper.__aexit__(RuntimeError, error, None) is False

        assert stream.close_count == 1
        assert wrapper._self_stop_count == 0
        assert wrapper._self_failures == [error]

    asyncio.run(exercise())


def test_async_stream_wrapper_exit_keeps_user_error_when_close_fails():
    async def exercise():
        close_error = RuntimeError("close failure")
        stream = _FakeAsyncStream(chunks=["chunk"], close_error=close_error)
        wrapper = _TestAsyncStreamWrapper(stream)
        error = RuntimeError("user failure")

        assert await wrapper.__aexit__(RuntimeError, error, None) is False

        assert stream.close_count == 1
        assert wrapper._self_failures == [error]
        assert wrapper._self_stop_count == 0

    asyncio.run(exercise())


def test_async_stream_wrapper_swallows_finalize_errors():
    async def exercise():
        wrapper = _FailingAsyncStopStreamWrapper(_FakeAsyncStream())

        await wrapper.close()
        await wrapper.close()

        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_swallows_failure_finalize_errors():
    async def exercise():
        close_error = RuntimeError("close failure")
        stream = _FakeAsyncStream(close_error=close_error)
        wrapper = _FailingAsyncFailStreamWrapper(stream)

        with pytest.raises(RuntimeError, match="close failure"):
            await wrapper.close()
        stream._close_error = None
        await wrapper.close()

        assert wrapper._self_failures == [close_error]

    asyncio.run(exercise())


def test_async_stream_wrapper_swallows_stop_iteration_finalize_errors():
    async def exercise():
        wrapper = _FailingAsyncStopStreamWrapper(_FakeAsyncStream())

        with pytest.raises(StopAsyncIteration):
            await anext(wrapper)

    asyncio.run(exercise())


def test_async_stream_wrapper_preserves_stream_error_when_finalize_fails():
    async def exercise():
        error = RuntimeError("stream failure")
        wrapper = _FailingAsyncFailStreamWrapper(_FakeAsyncStream(error=error))

        with pytest.raises(RuntimeError, match="stream failure"):
            await anext(wrapper)

    asyncio.run(exercise())


def test_async_stream_wrapper_getattr_passthrough():
    wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

    assert wrapper.extra_attribute == "passthrough"


def test_async_stream_wrapper_exposes_wrapped_stream():
    stream = _FakeAsyncStream()
    wrapper = _TestAsyncStreamWrapper(stream)

    assert getattr(wrapper, "__wrapped__") is stream


def test_async_stream_wrapper_magic_method_passthrough():
    wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

    assert len(wrapper) == 42


def test_async_stream_wrapper_stop_iteration_does_not_double_finalize():
    async def exercise():
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

        with pytest.raises(StopAsyncIteration):
            await anext(wrapper)
        await wrapper.close()

        assert wrapper._self_stop_count == 1
        assert not wrapper._self_failures

    asyncio.run(exercise())


def test_async_stream_wrapper_swallows_process_chunk_errors():
    async def exercise():
        wrapper = _FailingAsyncProcessStreamWrapper(
            _FakeAsyncStream(chunks=["chunk"])
        )

        assert await anext(wrapper) == "chunk"
        assert not wrapper._self_failures

    asyncio.run(exercise())


# --- Timing measurement tests ---


def test_sync_stream_wrapper_records_ttfc():
    """TTFC is computed from start_time_s to first chunk arrival."""

    stream = _FakeSyncStream(chunks=["a", "b", "c"])
    wrapper = _TestSyncStreamWrapper.__new__(_TestSyncStreamWrapper)
    SyncStreamWrapper.__init__(wrapper, stream, start_time_s=99.0)
    wrapper._self_processed = []
    wrapper._self_stop_count = 0
    wrapper._self_failures = []

    # Iterate with controlled time. Only after-read timestamps are
    # consumed now that the wrapper no longer measures blocking-read time.
    read_times = iter([101.2, 101.8, 102.1])
    with patch("timeit.default_timer", side_effect=read_times):
        chunks = list(wrapper)

    assert chunks == ["a", "b", "c"]
    # TTFC = first chunk after_read (101.2) - start_time_s (99.0) = 2.2
    assert wrapper._self_ttfc_seconds == pytest.approx(2.2)
    # Inter-chunk gaps: 101.8 - 101.2 = 0.6, 102.1 - 101.8 = 0.3
    assert wrapper._self_last_chunk_at == pytest.approx(102.1)


def test_sync_stream_wrapper_no_ttfc_without_start_time():
    """Without start_time_s, TTFC stays None."""
    stream = _FakeSyncStream(chunks=["a", "b"])
    wrapper = _TestSyncStreamWrapper(stream)

    list(wrapper)

    assert wrapper._self_ttfc_seconds is None
    # With timing disabled (no start_time_s, no timing_target), all
    # timing state stays None; the chunk-arrival timer is not invoked.
    assert wrapper._self_last_chunk_at is None


def test_sync_stream_wrapper_single_chunk_no_gaps():
    """Single-chunk stream has TTFC but no gaps."""

    stream = _FakeSyncStream(chunks=["only"])
    wrapper = _TestSyncStreamWrapper.__new__(_TestSyncStreamWrapper)
    SyncStreamWrapper.__init__(wrapper, stream, start_time_s=50.0)
    wrapper._self_processed = []
    wrapper._self_stop_count = 0
    wrapper._self_failures = []

    read_times = iter([60.5])
    with patch("timeit.default_timer", side_effect=read_times):
        chunks = list(wrapper)

    assert chunks == ["only"]
    assert wrapper._self_ttfc_seconds == pytest.approx(10.5)  # 60.5 - 50.0
    # First chunk records arrival time but emits no gap.
    assert wrapper._self_last_chunk_at == pytest.approx(60.5)


def test_sync_stream_wrapper_error_before_first_chunk_no_ttfc():
    """If stream errors before first chunk, no TTFC is recorded."""
    stream = _FakeSyncStream(error=RuntimeError("network"))
    wrapper = _TestSyncStreamWrapper.__new__(_TestSyncStreamWrapper)
    SyncStreamWrapper.__init__(wrapper, stream, start_time_s=10.0)
    wrapper._self_processed = []
    wrapper._self_stop_count = 0
    wrapper._self_failures = []

    with pytest.raises(RuntimeError, match="network"):
        next(wrapper)

    assert wrapper._self_ttfc_seconds is None
    assert wrapper._self_last_chunk_at is None


def test_async_stream_wrapper_records_ttfc():
    """Async wrapper records TTFC and chunk gaps."""

    async def exercise():
        stream = _FakeAsyncStream(chunks=["x", "y", "z"])
        wrapper = _TestAsyncStreamWrapper.__new__(_TestAsyncStreamWrapper)
        AsyncStreamWrapper.__init__(wrapper, stream, start_time_s=200.0)
        wrapper._self_processed = []
        wrapper._self_stop_count = 0
        wrapper._self_failures = []

        read_times = iter([201.3, 202.0, 202.2])
        with patch("timeit.default_timer", side_effect=read_times):
            chunks = []
            async for chunk in wrapper:
                chunks.append(chunk)

        assert chunks == ["x", "y", "z"]
        assert wrapper._self_ttfc_seconds == pytest.approx(
            1.3
        )  # 201.3 - 200.0
        # Inter-chunk gaps: 202.0 - 201.3 = 0.7, 202.2 - 202.0 = 0.2
        assert wrapper._self_last_chunk_at == pytest.approx(202.2)

    asyncio.run(exercise())


# --- timing_target sync tests ---


class _FakeTimingTarget:
    def __init__(self):
        self.ttfc_seconds = None
        self.chunk_gap_seconds = []

    def _record_chunk_gap(self, gap):
        self.chunk_gap_seconds.append(gap)


class _TimingTargetSyncWrapper(SyncStreamWrapper):
    def __init__(self, stream, **kwargs):
        super().__init__(stream, **kwargs)
        self._self_processed = []
        self._self_end_target_ttfc = None

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        if self._self_timing_target is not None:
            self._self_end_target_ttfc = self._self_timing_target.ttfc_seconds

    def _on_stream_error(self, error):
        if self._self_timing_target is not None:
            self._self_end_target_ttfc = self._self_timing_target.ttfc_seconds


def test_timing_target_receives_values_on_success():
    mock_patch = patch

    target = _FakeTimingTarget()
    stream = _FakeSyncStream(chunks=["a", "b", "c"])
    wrapper = _TimingTargetSyncWrapper(
        stream, start_time_s=100.0, timing_target=target
    )

    times = iter([100.5, 101.0, 101.3])
    with mock_patch("timeit.default_timer", side_effect=times):
        for _ in wrapper:
            pass

    assert target.ttfc_seconds == pytest.approx(0.5)
    # Inter-chunk gaps: 101.0 - 100.5 = 0.5, 101.3 - 101.0 = 0.3
    assert target.chunk_gap_seconds == pytest.approx([0.5, 0.3])


def test_timing_target_without_start_time_raises():
    """timing_target without start_time_s is a misconfiguration: TTFC has no
    meaning without a start, so the constructor surfaces it loudly rather than
    silently emitting no TTFC."""
    target = _FakeTimingTarget()
    stream = _FakeSyncStream(chunks=["a"])
    with pytest.raises(ValueError, match="start_time_s is required"):
        _TimingTargetSyncWrapper(stream, timing_target=target)


def test_async_timing_target_without_start_time_raises():
    target = _FakeTimingTarget()
    stream = _FakeAsyncStream(chunks=["a"])
    wrapper = _TestAsyncStreamWrapper.__new__(_TestAsyncStreamWrapper)
    with pytest.raises(ValueError, match="start_time_s is required"):
        AsyncStreamWrapper.__init__(wrapper, stream, timing_target=target)


def test_timing_target_populated_before_on_stream_end():
    mock_patch = patch

    target = _FakeTimingTarget()
    stream = _FakeSyncStream(chunks=["x"])
    wrapper = _TimingTargetSyncWrapper(
        stream, start_time_s=50.0, timing_target=target
    )

    times = iter([50.2])
    with mock_patch("timeit.default_timer", side_effect=times):
        for _ in wrapper:
            pass

    # Hook captured target.ttfc_seconds when _on_stream_end ran
    assert wrapper._self_end_target_ttfc == pytest.approx(0.2)


def test_timing_target_populated_before_on_stream_error():
    mock_patch = patch

    target = _FakeTimingTarget()
    error = RuntimeError("fail")
    stream = _FakeSyncStream(chunks=["x"], error=error)
    wrapper = _TimingTargetSyncWrapper(
        stream, start_time_s=50.0, timing_target=target
    )

    times = iter([50.2])
    with mock_patch("timeit.default_timer", side_effect=times):
        with pytest.raises(RuntimeError):
            for _ in wrapper:
                pass

    assert wrapper._self_end_target_ttfc == pytest.approx(0.2)


def test_no_timing_target_still_works():
    stream = _FakeSyncStream(chunks=["a", "b"])
    wrapper = _TimingTargetSyncWrapper(stream, start_time_s=10.0)

    for _ in wrapper:
        pass

    assert wrapper._self_ttfc_seconds is not None
