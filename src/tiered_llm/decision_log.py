"""Append-only JSONL decision log: thread-safe, buffered, crash-tolerant.

One JSON object per line. Every model call, every skipped provider, every
breaker transition and every cascade verdict ends up here, which is what turns
"the system did something odd at 03:12" into a trace you can read.

Durability model
----------------
* Events are buffered and written when ``max_buffer`` events are pending,
  every ``max_delay`` seconds (a small daemon thread), immediately for event
  types in ``flush_on``, on :meth:`flush`/:meth:`close`, and at interpreter exit.
  A hard crash loses at most ``max_delay`` seconds of events; use
  ``max_buffer=1`` for strict per-event durability.
* Each flush is one unbuffered append followed (by default) by ``fsync``. If the
  append fails halfway (disk full), the file is truncated back to where it was
  and the whole batch is retried later, so events are neither lost nor doubled.
* One writer process per file; threads within the process are fine.
* If a previous process died mid-line, the unterminated tail is closed with a
  newline before appending, so the next event cannot be glued onto a fragment
  and corrupt both. :func:`iter_decisions` skips such fragments.
* ``log()`` never raises into business code. File errors keep events queued
  (bounded by ``max_pending``); an event that cannot be serialised is dropped
  instead of being retried forever.
* ``NaN``/``inf`` are written as ``null``: bare ``NaN`` tokens parse in Python
  but break other JSON readers and poison averages downstream.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import math
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

__all__ = ["DecisionLog", "iter_decisions"]

logger = logging.getLogger("tiered_llm.decision_log")

DEFAULT_FLUSH_ON = frozenset({"breaker_transition"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(v) for v in value]
    return value


def _serialize(event: dict[str, Any]) -> str:
    try:
        return json.dumps(event, default=str, ensure_ascii=False, allow_nan=False)
    except ValueError:
        return json.dumps(_sanitize(event), default=str, ensure_ascii=False, allow_nan=False)


def _heal_unterminated_tail(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
        with open(path, "rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except OSError:
        pass  # best effort; never block logging on it


def _flush_at_exit(ref: weakref.ReferenceType[DecisionLog]) -> None:
    log = ref()
    if log is not None:
        log.close()


def _flush_periodically(
    ref: weakref.ReferenceType[DecisionLog], stop: threading.Event, interval: float
) -> None:
    # Holds only a weak reference, so an abandoned log can still be collected.
    while not stop.wait(interval):
        log = ref()
        if log is None:
            return
        try:
            log.flush()
        except Exception:  # pragma: no cover - flush already guards itself
            logger.exception("decision log: periodic flush failed")
        del log


class DecisionLog:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        rotate_daily: bool = False,
        max_buffer: int = 50,
        max_delay: float = 5.0,
        max_pending: int = 100_000,
        fsync: bool = True,
        flush_on: Iterable[str] = DEFAULT_FLUSH_ON,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """
        :param path: a ``.jsonl`` file, or a directory when ``rotate_daily`` is set
            (files are then named ``YYYY-MM-DD.jsonl`` by UTC date).
        """
        self.path = Path(path)
        self.rotate_daily = rotate_daily
        self.max_buffer = max(1, max_buffer)
        self.max_delay = max_delay
        self.max_pending = max_pending
        self.fsync = fsync
        self.flush_on = frozenset(flush_on)
        self._clock = clock
        self._now = now
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._handle: BinaryIO | None = None
        self._handle_path: Path | None = None
        self._last_flush = clock()
        self._closed = False
        self._written = 0
        self._dropped = 0
        self._write_failures = 0
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None
        atexit.register(_flush_at_exit, weakref.ref(self))

    def _start_flusher(self) -> None:
        # Called under the lock on first use, so no thread exists for unused logs.
        if self._flusher is None and self.max_delay > 0:
            self._flusher = threading.Thread(
                target=_flush_periodically,
                args=(weakref.ref(self), self._stop, self.max_delay),
                name="decision-log-flusher",
                daemon=True,
            )
            self._flusher.start()

    # -- writing ------------------------------------------------------------

    def log(self, event: str, **fields: Any) -> None:
        """Record one event. Never raises."""
        try:
            record: dict[str, Any] = {"ts": self._now().isoformat(timespec="milliseconds"), "event": event}
            record.update(fields)
            with self._lock:
                if self._closed:
                    return
                self._start_flusher()
                self._buffer.append(record)
                if len(self._buffer) > self.max_pending:
                    overflow = len(self._buffer) - self.max_pending
                    del self._buffer[:overflow]
                    self._dropped += overflow
                due = (
                    event in self.flush_on
                    or len(self._buffer) >= self.max_buffer
                    or self._clock() - self._last_flush >= self.max_delay
                )
            if due:
                self.flush()
        except Exception:  # pragma: no cover - defensive, logging must not break callers
            logger.exception("decision log: failed to record event %r", event)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._flush_locked()
            self._closed = True
            if self._handle is not None:
                with contextlib.suppress(OSError):
                    self._handle.close()
                self._handle = None

    def __enter__(self) -> DecisionLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _target(self) -> Path:
        if self.rotate_daily:
            return self.path / f"{self._now().date().isoformat()}.jsonl"
        return self.path

    def _ensure_handle(self) -> BinaryIO:
        target = self._target()
        if self._handle is not None and self._handle_path == target:
            return self._handle
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None
        target.parent.mkdir(parents=True, exist_ok=True)
        _heal_unterminated_tail(target)
        # Unbuffered binary append: a failed write can be rolled back exactly.
        self._handle = open(target, "ab", buffering=0)  # noqa: SIM115
        self._handle_path = target
        return self._handle

    def _flush_locked(self) -> None:
        self._last_flush = self._clock()
        if not self._buffer:
            return
        events, self._buffer = self._buffer, []
        try:
            handle = self._ensure_handle()
        except OSError as exc:
            self._requeue(events, exc)
            return

        lines: list[str] = []
        for event in events:
            try:
                lines.append(_serialize(event))
            except Exception as exc:  # poison event: drop it, never retry forever
                self._dropped += 1
                self._record_error(exc)
        if not lines:
            return
        data = ("\n".join(lines) + "\n").encode("utf-8", errors="replace")
        start = 0
        try:
            start = os.fstat(handle.fileno()).st_size
            view = memoryview(data)
            while view:
                written = handle.write(view)
                if not written:
                    raise OSError("short write")
                view = view[written:]
        except OSError as exc:
            # Roll back whatever part of the batch reached the file, then retry
            # the whole batch later: no fragments, no duplicates.
            with contextlib.suppress(OSError):
                os.ftruncate(handle.fileno(), start)
            with contextlib.suppress(OSError):
                handle.close()
            self._handle = None
            self._requeue(events, exc)
            return
        self._written += len(lines)
        if self.fsync:
            try:
                os.fsync(handle.fileno())
            except OSError as exc:
                # The data is in the OS cache already; re-queueing would duplicate it.
                self._record_error(exc)
                return
        self._last_error = None

    def _requeue(self, events: list[dict[str, Any]], exc: BaseException) -> None:
        self._buffer = events + self._buffer
        if len(self._buffer) > self.max_pending:
            overflow = len(self._buffer) - self.max_pending
            del self._buffer[:overflow]
            self._dropped += overflow
        self._record_error(exc)

    def _record_error(self, exc: BaseException) -> None:
        self._write_failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.error("decision log write failed: %s", self._last_error)

    # -- introspection ------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Numbers for a health endpoint: is the audit trail actually being written?"""
        with self._lock:
            return {
                "path": str(self._target()),
                "buffered": len(self._buffer),
                "written": self._written,
                "dropped": self._dropped,
                "write_failures": self._write_failures,
                "last_error": self._last_error,
                "seconds_since_flush": round(self._clock() - self._last_flush, 3),
            }


def iter_decisions(path: str | os.PathLike[str], *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """Yield the events of one log file, skipping blank or crash-truncated lines."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                if strict:
                    raise ValueError(f"{path}:{number}: invalid JSON line") from None
                continue
            if isinstance(record, dict):
                yield record
