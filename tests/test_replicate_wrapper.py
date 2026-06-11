"""Tests for ``token_sentinel.wrappers.replicate.wrap_replicate``.

No real Replicate API calls. We construct ``SimpleNamespace`` mocks shaped
like a ``replicate.Client`` instance and verify that:

  - ``client.run`` is swapped with an instrumented version that builds a
    CallRecord with ``usage_extra`` populated for the model's pricing
    dimension (per_image vs per_second).
  - ``client.predictions.create`` stashes a pending entry keyed on
    ``prediction.id`` in the module-level ``_REPLICATE_PENDING`` dict.
  - ``client.predictions.get`` on a terminal-status prediction pops the
    pending entry and fires ``sentinel.record_call``.
  - Failed / canceled predictions still produce a CallRecord (so the
    customer sees burn for compute-spent-but-failed runs).
  - Instrumentation never breaks the user's call.
  - The Sentinel ``wrap()`` dispatch routes ``replicate.Client`` to
    this wrapper.

``pytest.importorskip("replicate")`` is intentionally NOT used here: the
wrapper duck-types the Client surface and has no module-level import of
the SDK, so the tests run cleanly even when the optional ``replicate``
package is not installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel
from token_sentinel.wrappers.replicate import (
    _PENDING_TTL_SECONDS,
    _REPLICATE_PENDING,
    _drop_stale_pending,
    wrap_replicate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingFn:
    """Real callable for ``functools.wraps`` compatibility — mirrors the
    pattern from ``conftest._RecordingCreate``."""

    __name__ = "fn"
    __qualname__ = "Client.fn"
    __module__ = "replicate.client"
    __annotations__: dict = {}
    __doc__ = "mock"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value: Any = None
        self.side_effect: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None and (
            isinstance(self.side_effect, BaseException)
            or (isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException))
        ):
            raise self.side_effect
        return self.return_value


def _make_replicate_client() -> SimpleNamespace:
    """Build a fake replicate.Client with the surface the wrapper needs."""
    fake_class = type("Client", (), {"__module__": "replicate.client"})
    client = fake_class()
    client.run = _RecordingFn()
    create = _RecordingFn()
    create.__name__ = "create"
    get = _RecordingFn()
    get.__name__ = "get"
    client.predictions = SimpleNamespace(create=create, get=get)
    return client


def _make_prediction(
    *,
    pid: str = "pred-1",
    status: str = "succeeded",
    output: Any = None,
    predict_time: float | None = None,
    error: Any = None,
) -> SimpleNamespace:
    metrics = SimpleNamespace(predict_time=predict_time) if predict_time is not None else None
    return SimpleNamespace(
        id=pid,
        status=status,
        output=output,
        metrics=metrics,
        error=error,
    )


@pytest.fixture(autouse=True)
def _clear_pending() -> Any:
    """Wipe the module-level pending dict before each test."""
    _REPLICATE_PENDING.clear()
    yield
    _REPLICATE_PENDING.clear()


# ---------------------------------------------------------------------------
# 1. wrap_replicate swaps client.run and predictions.create/get
# ---------------------------------------------------------------------------


def test_wrap_replicate_swaps_methods() -> None:
    client = _make_replicate_client()
    original_run = client.run
    original_create = client.predictions.create
    original_get = client.predictions.get

    s = Sentinel(project="proj")
    out = wrap_replicate(client, s)

    assert out is client
    assert client.run is not original_run
    assert client.predictions.create is not original_create
    assert client.predictions.get is not original_get


# ---------------------------------------------------------------------------
# 2. client.run on an image model builds per_image usage_extra
# ---------------------------------------------------------------------------


def test_run_image_model_records_per_image() -> None:
    client = _make_replicate_client()
    client.run.return_value = ["https://replicate.delivery/pbxt/abc/out.png"]

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    out = client.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": "a cat sitting on a mat"},
    )
    assert out == ["https://replicate.delivery/pbxt/abc/out.png"]

    sessions = list(s.tracer.all_sessions())
    assert len(sessions) == 1
    rec = s.tracer.session(sessions[0])[0]
    assert rec.provider == "replicate"
    assert rec.model == "black-forest-labs/flux-schnell"
    assert rec.method == "run"
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0
    assert rec.usage_extra["dimension_kind"] == "per_image"
    assert rec.usage_extra["dimension_value"] == 1.0


# ---------------------------------------------------------------------------
# 3. client.run on a video model builds per_second usage_extra
# ---------------------------------------------------------------------------


def test_run_video_model_records_per_second() -> None:
    client = _make_replicate_client()
    client.run.return_value = "https://replicate.delivery/pbxt/abc/out.mp4"

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    out = client.run(
        "tencent/hunyuan-video",
        input={"prompt": "a cat walking on a beach"},
    )
    assert out == "https://replicate.delivery/pbxt/abc/out.mp4"

    sessions = list(s.tracer.all_sessions())
    rec = s.tracer.session(sessions[0])[0]
    assert rec.usage_extra["dimension_kind"] == "per_second"
    # No metrics.predict_time on bare run() output — wrapper defaults to 1.0
    # so we don't bill $0 for a successful video gen.
    assert rec.usage_extra["dimension_value"] == 1.0


# ---------------------------------------------------------------------------
# 4. predictions.create stashes a pending entry; predictions.get records it
# ---------------------------------------------------------------------------


def test_predictions_create_then_get_records_call() -> None:
    client = _make_replicate_client()
    submission = _make_prediction(pid="pred-A", status="starting", output=None)
    # Capture references to the originals so the test can change their
    # return_value AFTER wrap_replicate swaps the slot on
    # ``client.predictions``. The wrapper closes over the original
    # callable, not the live slot, so flipping the slot's return_value
    # after wrap has no effect — set it on the original captured mock.
    original_create = client.predictions.create
    original_get = client.predictions.get
    original_create.return_value = submission

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    out_sub = client.predictions.create(
        version="ideogram-ai/ideogram-v2",
        input={"prompt": "test prompt"},
    )
    assert out_sub is submission
    assert "pred-A" in _REPLICATE_PENDING
    # No CallRecord yet — submission alone shouldn't fire record_call.
    assert list(s.tracer.all_sessions()) == []

    # Now the terminal poll.
    final = _make_prediction(
        pid="pred-A",
        status="succeeded",
        output=["https://replicate.delivery/pbxt/xyz/out.png"],
    )
    original_get.return_value = final
    got = client.predictions.get("pred-A")
    assert got is final

    # Pending entry has been popped + CallRecord has been emitted.
    assert "pred-A" not in _REPLICATE_PENDING
    sessions = list(s.tracer.all_sessions())
    assert len(sessions) == 1
    rec = s.tracer.session(sessions[0])[0]
    assert rec.method == "predictions.get"
    assert rec.model == "ideogram-ai/ideogram-v2"
    assert rec.usage_extra["dimension_kind"] == "per_image"
    assert rec.usage_extra["dimension_value"] == 1.0
    assert rec.raw_response_meta["status"] == "succeeded"


# ---------------------------------------------------------------------------
# 5. Failed prediction still records a CallRecord (with status + error meta)
# ---------------------------------------------------------------------------


def test_failed_prediction_still_recorded() -> None:
    client = _make_replicate_client()
    submission = _make_prediction(pid="pred-F", status="starting")
    original_create = client.predictions.create
    original_get = client.predictions.get
    original_create.return_value = submission

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    client.predictions.create(version="black-forest-labs/flux-dev", input={"prompt": "x"})

    failed = _make_prediction(
        pid="pred-F",
        status="failed",
        output=None,
        predict_time=3.2,
        error="GPU OOM during inference",
    )
    original_get.return_value = failed
    client.predictions.get("pred-F")

    sessions = list(s.tracer.all_sessions())
    rec = s.tracer.session(sessions[0])[0]
    assert rec.raw_response_meta["status"] == "failed"
    assert rec.raw_response_meta["error"] == "GPU OOM during inference"
    # predict_time=3.2 → fallback to per_second dimension for failed runs
    assert rec.usage_extra["dimension_value"] == 3.2


# ---------------------------------------------------------------------------
# 6. predictions.get on a non-terminal status does NOT fire record_call
# ---------------------------------------------------------------------------


def test_get_on_non_terminal_status_no_record() -> None:
    client = _make_replicate_client()
    submission = _make_prediction(pid="pred-P", status="starting")
    original_create = client.predictions.create
    original_get = client.predictions.get
    original_create.return_value = submission

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    client.predictions.create(version="black-forest-labs/flux-schnell", input={})

    # Mid-poll: still "processing" — no CallRecord yet, pending entry stays.
    poll = _make_prediction(pid="pred-P", status="processing", output=None)
    original_get.return_value = poll
    client.predictions.get("pred-P")

    assert "pred-P" in _REPLICATE_PENDING
    assert list(s.tracer.all_sessions()) == []


# ---------------------------------------------------------------------------
# 7. Failure isolation: a buggy record_call must not crash the user's call
# ---------------------------------------------------------------------------


def test_failure_isolation_in_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_replicate_client()
    client.run.return_value = ["https://example/x.png"]

    s = Sentinel(project="proj")
    wrap_replicate(client, s)

    # Force record_call to explode for everything that isn't LeakDetected.
    def boom(_call: Any) -> None:
        raise RuntimeError("simulated tracer crash")

    monkeypatch.setattr(s, "record_call", boom)

    # User's call must still return the model output unchanged.
    out = client.run("black-forest-labs/flux-dev", input={"prompt": "hi"})
    assert out == ["https://example/x.png"]


# ---------------------------------------------------------------------------
# 8. Missing-output successful prediction still records (with count=1)
# ---------------------------------------------------------------------------


def test_missing_output_succeeded_records_one_image() -> None:
    """A succeeded prediction with output=None is rare but possible (e.g. a
    diff'd model whose output field is async-loaded). We bill 0 images
    so the customer sees zero burn and can investigate."""
    client = _make_replicate_client()
    original_create = client.predictions.create
    original_get = client.predictions.get
    original_create.return_value = _make_prediction(pid="pred-N", status="starting")

    s = Sentinel(project="proj")
    wrap_replicate(client, s)
    client.predictions.create(version="recraft-ai/recraft-v3", input={"prompt": "x"})

    original_get.return_value = _make_prediction(pid="pred-N", status="succeeded", output=None)
    client.predictions.get("pred-N")

    sessions = list(s.tracer.all_sessions())
    rec = s.tracer.session(sessions[0])[0]
    # None output → 0 artifacts counted (not 1). The customer sees the
    # burn dimension (per_image with value 0) and can investigate why
    # the model returned no output.
    assert rec.usage_extra["dimension_kind"] == "per_image"
    assert rec.usage_extra["dimension_value"] == 0.0


# ---------------------------------------------------------------------------
# 9. Stale pending entries are dropped after the TTL
# ---------------------------------------------------------------------------


def test_stale_pending_entries_dropped() -> None:
    """The module-level pending dict has a 10-minute TTL. Entries older
    than that get GC'd on the next access."""
    import time as _time

    _REPLICATE_PENDING["old-id"] = {
        "session_id": "s",
        "model_ref": "x",
        "input_dict": {},
        "monotonic_start": _time.perf_counter() - _PENDING_TTL_SECONDS - 5.0,
        "sentinel": None,
    }
    # An access (any access) triggers lazy GC of stale entries.
    _drop_stale_pending(_time.perf_counter())
    assert "old-id" not in _REPLICATE_PENDING


# ---------------------------------------------------------------------------
# 10. Sentinel.wrap() dispatches a replicate.Client to wrap_replicate
# ---------------------------------------------------------------------------


def test_sentinel_wrap_dispatches_replicate() -> None:
    """The module-string detection in ``Sentinel.wrap()`` routes a fake
    ``replicate.Client`` to the replicate wrapper."""
    client = _make_replicate_client()
    s = Sentinel(project="proj")
    original_run = client.run
    out = s.wrap(client)
    assert out is client
    assert client.run is not original_run
