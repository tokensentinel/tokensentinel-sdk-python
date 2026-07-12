"""Timezone-safe time helpers for rules.

Rules compare ``CallRecord.timestamp`` values for sliding windows. Mixing
naive and aware datetimes raises ``TypeError``; that exception is swallowed
by ``Sentinel.record_call``'s per-rule guard and *silently disables* the
rule for the call. Treat naive timestamps as UTC so evaluation always
returns a number.
"""

from __future__ import annotations

from datetime import datetime, timezone


def as_utc(dt: datetime) -> datetime:
    """Return ``dt`` as timezone-aware UTC.

    Naive datetimes are assumed UTC (the wrappers always emit aware UTC;
    hand-built records in tests or custom enrichers sometimes omit tz).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def elapsed_seconds(later: datetime, earlier: datetime) -> float:
    """Seconds from ``earlier`` to ``later``, timezone-safe.

    Positive when ``later`` is after ``earlier``. Never raises on naive/aware
    mismatch.
    """
    return (as_utc(later) - as_utc(earlier)).total_seconds()
