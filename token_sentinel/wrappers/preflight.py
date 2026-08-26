"""Shared wrap-time pre-call guard.

Stops the next provider HTTP request when the cloud policy says so
(kill-switch, org/session budget, already-recognised historical waste).
Never raises anything except :class:`LeakDetected` subclasses.
"""

from __future__ import annotations

from typing import Any

from token_sentinel.events import LeakDetected


def guard_before_call(sentinel: Any, session_id: str, **hint: Any) -> None:
    fn = getattr(sentinel, "preflight", None)
    if fn is None:
        return
    try:
        fn(session_id, **hint)
    except LeakDetected:
        raise
    except Exception:
        return
