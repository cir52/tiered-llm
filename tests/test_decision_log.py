from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tiered_llm import DecisionLog, iter_decisions


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_buffers_until_threshold_then_writes_jsonl(tmp_path, clock):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=3, max_delay=60, clock=clock, fsync=False)
    log.log("a", n=1)
    log.log("b", n=2)
    assert not path.exists()
    log.log("c", n=3)
    records = [json.loads(line) for line in lines(path)]
    assert [(r["event"], r["n"]) for r in records] == [("a", 1), ("b", 2), ("c", 3)]
    assert all(r["ts"].endswith("+00:00") for r in records)


def test_flushes_after_max_delay_and_on_priority_events(tmp_path, clock):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=100, max_delay=5, clock=clock, fsync=False)
    log.log("a")
    clock.advance(5)
    log.log("b")
    assert len(lines(path)) == 2
    log.log("breaker_transition", provider="x")
    assert len(lines(path)) == 3


def test_close_flushes_and_stops_accepting(tmp_path):
    path = tmp_path / "d.jsonl"
    with DecisionLog(path, max_buffer=100) as log:
        log.log("a")
    log.log("after-close")
    assert [r["event"] for r in iter_decisions(path)] == ["a"]


def test_heals_a_crash_truncated_tail(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"event": "complete"}\n{"event": "trunc', encoding="utf-8")
    with DecisionLog(path, max_buffer=1, fsync=False) as log:
        log.log("next")
    assert [r["event"] for r in iter_decisions(path)] == ["complete", "next"]
    with pytest.raises(ValueError, match=":2:"):
        list(iter_decisions(path, strict=True))


def test_non_finite_numbers_become_null(tmp_path):
    path = tmp_path / "d.jsonl"
    with DecisionLog(path, fsync=False) as log:
        log.log("m", score=float("nan"), nested={"x": [float("inf"), 1.5]})
    (record,) = iter_decisions(path)
    assert record["score"] is None and record["nested"] == {"x": [None, 1.5]}
    assert "NaN" not in path.read_text()


def test_unserializable_values_fall_back_to_str(tmp_path):
    path = tmp_path / "d.jsonl"
    with DecisionLog(path, fsync=False) as log:
        log.log("m", when=datetime(2026, 1, 2, tzinfo=timezone.utc), obj=object())
    (record,) = iter_decisions(path)
    assert record["when"].startswith("2026-01-02")


def test_daily_rotation(tmp_path):
    day = [datetime(2026, 3, 1, 23, 59, tzinfo=timezone.utc)]
    log = DecisionLog(tmp_path / "logs", rotate_daily=True, max_buffer=1, fsync=False, now=lambda: day[0])
    log.log("late")
    day[0] += timedelta(minutes=2)
    log.log("early")
    log.close()
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == ["2026-03-01.jsonl", "2026-03-02.jsonl"]


def test_write_failures_keep_events_queued(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=1, fsync=False)
    original = log._ensure_handle
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return original()

    monkeypatch.setattr(log, "_ensure_handle", flaky)
    log.log("first")
    health = log.health()
    assert health["buffered"] == 1 and health["write_failures"] == 1 and "disk full" in health["last_error"]
    log.log("second")
    log.close()
    assert [r["event"] for r in iter_decisions(path)] == ["first", "second"]
    assert log.health()["last_error"] is None


def test_pending_queue_is_bounded(tmp_path, monkeypatch):
    log = DecisionLog(tmp_path / "d.jsonl", max_buffer=1, max_pending=3, fsync=False)
    monkeypatch.setattr(log, "_ensure_handle", lambda: (_ for _ in ()).throw(OSError("gone")))
    for i in range(10):
        log.log("e", i=i)
    health = log.health()
    assert health["buffered"] == 3 and health["dropped"] == 7


def test_concurrent_writers_produce_only_whole_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=7, fsync=False)

    def writer(worker: int) -> None:
        for i in range(500):
            log.log("tick", worker=worker, i=i, payload="x" * 50)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()
    records = list(iter_decisions(path, strict=True))
    assert len(records) == 4000
    assert {(r["worker"], r["i"]) for r in records} == {(w, i) for w in range(8) for i in range(500)}


def test_idle_events_are_flushed_by_the_background_timer(tmp_path):
    import time

    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=100, max_delay=0.05, fsync=False)
    log.log("lonely")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not path.exists():
        time.sleep(0.01)
    assert [r["event"] for r in iter_decisions(path)] == ["lonely"]
    log.close()


def test_a_half_written_batch_is_rolled_back_not_duplicated(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, max_buffer=10, max_delay=60, fsync=False)
    original = log._ensure_handle
    state = {"fail": True}

    class DiskFullAfterHalf:
        def __init__(self, inner):
            self.inner = inner

        def fileno(self):
            return self.inner.fileno()

        def write(self, data):
            if state["fail"]:
                state["fail"] = False
                self.inner.write(bytes(data[: len(data) // 2]))
                raise OSError(28, "No space left on device")
            return self.inner.write(data)

        def close(self):
            self.inner.close()

    monkeypatch.setattr(log, "_ensure_handle", lambda: DiskFullAfterHalf(original()))
    for i in range(10):
        log.log("e", i=i)  # the 10th triggers a flush that fails halfway
    assert log.health()["write_failures"] == 1
    log.close()
    records = list(iter_decisions(path, strict=True))
    assert [r["i"] for r in records] == list(range(10))
