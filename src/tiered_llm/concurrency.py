"""Event-loop-aware concurrency limit.

``asyncio.Semaphore`` binds to the event loop it is first used on. A provider
object created at import time and then used from several loops (``asyncio.run``
per job, ``asyncio.to_thread`` workers running their own loop, test suites)
fails with "is bound to a different event loop" - or, worse, silently stops
limiting. Trade52 hit exactly this with its local-GPU semaphore. This limiter
keeps one semaphore per running loop, created under a thread lock.

The limit therefore applies per event loop; in the usual single-loop service
that is the same as a global limit.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from types import TracebackType

__all__ = ["ConcurrencyLimiter"]


class ConcurrencyLimiter:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = limit
        self._lock = threading.Lock()
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            semaphore = self._semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.limit)
                self._semaphores[loop] = semaphore
            return semaphore

    async def __aenter__(self) -> None:
        await self._semaphore().acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._semaphore().release()
