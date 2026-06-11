"""In-process tracer: captures call records into per-session ring buffers."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable
from threading import Lock

from token_sentinel.events import CallRecord


class Tracer:
    """Bounded per-session ring buffer of call records.

    Two memory bounds:
      - ``max_records_per_session``: each session's deque caps at this length
        (oldest call evicted on overflow). Default 200.
      - ``max_sessions``: total number of distinct sessions retained. When the
        cap is hit, the *least recently used* session is evicted. Default 1000.

    The session cap exists because the wrappers default ``session_id`` to a
    fresh UUID per call when the user doesn't pass ``_sentinel_session_id``.
    Without a cap, a long-running multi-tenant agent leaks one dict entry per
    LLM call indefinitely. Set ``max_sessions=None`` to disable the cap if
    you're explicitly threading session IDs through every call.
    """

    def __init__(
        self,
        *,
        max_records_per_session: int = 200,
        max_sessions: int | None = 1000,
    ):
        self._max_records = max_records_per_session
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[CallRecord]] = OrderedDict()
        self._lock = Lock()

    def record(self, call: CallRecord) -> None:
        with self._lock:
            buf = self._sessions.get(call.session_id)
            if buf is None:
                buf = deque(maxlen=self._max_records)
                self._sessions[call.session_id] = buf
                if self._max_sessions is not None and len(self._sessions) > self._max_sessions:
                    # Evict the least recently used session (insertion order
                    # for new sessions, refresh order for accessed ones).
                    self._sessions.popitem(last=False)
            else:
                # Accessing an existing session refreshes its LRU position.
                self._sessions.move_to_end(call.session_id)
            buf.append(call)

    def session(self, session_id: str) -> list[CallRecord]:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(buf)

    def all_sessions(self) -> Iterable[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)
