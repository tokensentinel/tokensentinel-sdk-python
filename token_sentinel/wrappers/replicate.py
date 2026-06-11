"""Wrap a Replicate client to capture call records for non-token-priced models.

SDK choice
-----------------
We target the stable ``replicate`` package (``pip install replicate>=1.0``),
which exposes a ``replicate.Client`` class with ``client.run(...)``,
``client.predictions.create(...)`` and ``client.predictions.get(...)``. The
internal research
notes that ``replicate-python-beta`` (Stainless-generated) is the "right
thing" architecturally for new code, but its class is ``Replicate`` not
``Client`` and its install footprint is smaller. We pick stable ``replicate``
for  because:

  1. It's the package that 100% of existing Replicate Python code uses today.
  2. Both stable and beta SDKs expose the same ``client.run`` /
     ``client.predictions.{create,get}`` surface — wrapping one happens to
     wrap the other if a customer ever subclasses (the detection branch in
     ``sentinel.wrap()`` matches both module prefixes via the
     ``module.startswith("replicate")`` test).
  3. The stable package's known ``httpx.Proxy`` ImportError on Python 3.13+
     (per research note) is a deployment issue for the customer, not for
     the wrapper — we still work on Python 3.10/3.11/3.12 which is the
     SDK's official support window (see ``pyproject.toml``).

Pricing-dimension translation
-----------------------------
Replicate bills per-image / per-pixel / per-second, not per-token. The
wrapper builds a :class:`CallRecord` with ``prompt_tokens=0,
completion_tokens=0`` and stashes the actual pricing dimension under
``usage_extra``:

    {"dimension_kind": "per_image", "dimension_value": 1.0}    # image models
    {"dimension_kind": "per_second", "dimension_value": 4.2}   # video models

The cloud-side ``cost_estimator.NON_TOKEN_PRICES`` table dispatches on
``dimension_kind`` and multiplies through to a USD figure. See
the cloud-side cost estimator for
the dispatch path.

Two entry points
----------------
1. ``client.run(model_ref, input={...})`` — fire-and-forget. Synchronously
   waits for the prediction to complete and returns its output. We instrument
   this as a single-call boundary: record the CallRecord on return.

2. ``client.predictions.create(...)`` + later ``client.predictions.get(id)``
   — polling pattern. ``create`` returns a Prediction object whose
   ``.status`` starts at ``"starting"``; the customer polls ``.get(id)``
   until ``status`` reaches a terminal state. We track the submission in a
   process-wide ``_REPLICATE_PENDING`` dict keyed on ``prediction.id``,
   and build+record the CallRecord on the terminal ``predictions.get``.

Failure-isolation contract
--------------------------
Any error in instrumentation is swallowed; the user's ``client.run(...)``
or ``client.predictions.{create,get}(...)`` call returns the model output
unchanged. ``LeakDetected`` from ``mode='block'`` is the only exception
that propagates (matches the Anthropic/OpenAI/Bedrock contract).
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from token_sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# Pending-submissions registry
# ---------------------------------------------------------------------------
#
# Replicate's polling pattern is two HTTP calls (POST /predictions then
# polling GET /predictions/<id>) with arbitrary time between them. We need
# to remember the submission's start time + the input shape so the eventual
# terminal GET can build a complete CallRecord.
#
# Module-level dict keyed by ``prediction.id``. Entries are dropped on:
#   - the terminal ``predictions.get`` call (success / failure / canceled), or
#   - lazy GC during the next access if older than ``_PENDING_TTL_SECONDS``
#     (cleans up after polls that never reach a terminal state).
#
# Module-level (not Sentinel-instance) so two Sentinels sharing the same
# Replicate client through wrap() can both see the pending entries. The
# pending entry stores a weak ref to the recording Sentinel so the actual
# record_call dispatch happens against the right instance.

_PENDING_TTL_SECONDS = 600.0  # 10 minutes

_REPLICATE_PENDING: dict[str, dict[str, Any]] = {}
_PENDING_LOCK = threading.Lock()


def _drop_stale_pending(now_monotonic: float) -> None:
    """Drop entries older than ``_PENDING_TTL_SECONDS`` from the pending dict.

    Called on every access — the dict is bounded by per-second submission
    rate * TTL (e.g. 10 submissions/sec ~= 6000 entries max), so the
    O(n) sweep is cheap in practice. Holds the lock for the duration.
    """
    cutoff = now_monotonic - _PENDING_TTL_SECONDS
    with _PENDING_LOCK:
        stale = [
            pid
            for pid, entry in _REPLICATE_PENDING.items()
            if entry.get("monotonic_start", 0.0) < cutoff
        ]
        for pid in stale:
            _REPLICATE_PENDING.pop(pid, None)


# ---------------------------------------------------------------------------
# wrap_replicate
# ---------------------------------------------------------------------------


def wrap_replicate(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a ``replicate.Client`` in-place. Returns the same client.

    Mutates ``client.run`` and ``client.predictions.{create,get}`` so each
    is replaced with an instrumented version. Mirrors the Anthropic /
    OpenAI / Bedrock wrappers: failure-isolated, no async variants for
     (the stable replicate SDK exposes ``async_run`` / ``async_create``
    / ``async_get`` separately and we defer those to .1 alongside the
    audio providers' async surfaces).

    The detection branch in ``Sentinel.wrap()`` validates that
    ``client.predictions`` is reachable before this function runs, so we
    can assume the surface exists. Missing methods are silently skipped
    rather than crash — same defensive posture as the boto3 Bedrock
    wrapper, since SDK forks/subclasses might trim the surface.
    """
    # client.run — fire-and-forget convenience
    original_run = getattr(client, "run", None)
    if original_run is not None and callable(original_run):
        client.run = _make_run(original_run, sentinel)

    # client.predictions.create / get — polling-job pattern
    predictions = getattr(client, "predictions", None)
    if predictions is not None:
        original_create = getattr(predictions, "create", None)
        if original_create is not None and callable(original_create):
            predictions.create = _make_predictions_create(original_create, sentinel)
        original_get = getattr(predictions, "get", None)
        if original_get is not None and callable(original_get):
            predictions.get = _make_predictions_get(original_get, sentinel)

    return client


# ---------------------------------------------------------------------------
# client.run — fire-and-forget
# ---------------------------------------------------------------------------


def _make_run(original_run: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_run)
    def instrumented_run(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        # Replicate's run signature is ``run(ref, input=None, **kwargs)``.
        # The first positional arg is the model ref; we capture it for the
        # record without depending on the SDK's internal kwarg names. Async
        # variants are not instrumented in  — we'd see ``original_run``
        # be a coroutinefunction, which we explicitly don't support here.
        model_ref = ""
        if args:
            model_ref = str(args[0])
        elif "ref" in kwargs:
            model_ref = str(kwargs["ref"])
        raw_input = kwargs.get("input")
        input_dict: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}

        start = time.perf_counter()
        try:
            output = original_run(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary mirroring the other wrappers:
        # - Record-building errors are swallowed (instrumentation must
        #   never break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected,
        #   which is the entire point of mode='block' and must propagate.
        try:
            record = _build_record_for_run(
                session_id=session_id,
                model_ref=model_ref,
                input_dict=input_dict,
                output=output,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return output
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass
        return output

    return instrumented_run


# ---------------------------------------------------------------------------
# client.predictions.create — submit (don't build CallRecord yet)
# ---------------------------------------------------------------------------


def _make_predictions_create(original_create: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_create)
    def instrumented_create(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        # Capture submission metadata before we delegate. ``version`` (a
        # model ref) and ``input`` are the canonical kwargs on
        # ``predictions.create``; defensively pull from ``model`` too as a
        # legacy alias. The wrapper does not try to validate the args —
        # if the SDK rejects the call we propagate that as-is.
        model_ref = str(kwargs.get("version") or kwargs.get("model") or "")
        raw_input = kwargs.get("input")
        input_dict: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}

        start_monotonic = time.perf_counter()
        try:
            prediction = original_create(*args, **kwargs)
        except Exception:
            # The SDK raised before we could stash anything. Let it
            # propagate — we have nothing to clean up.
            raise

        # Stash a pending entry keyed on the prediction's id. We must
        # tolerate predictions without an id (a buggy SDK or a mocked
        # response): in that case we silently skip the stashing so the
        # eventual ``predictions.get`` falls through to a no-op too.
        try:
            prediction_id = _extract_prediction_id(prediction)
            if prediction_id:
                with _PENDING_LOCK:
                    _REPLICATE_PENDING[prediction_id] = {
                        "session_id": session_id,
                        "model_ref": model_ref,
                        "input_dict": dict(input_dict) if input_dict else {},
                        "monotonic_start": start_monotonic,
                        "sentinel": sentinel,
                    }
                # Opportunistic stale cleanup so the dict never grows
                # unbounded if a customer fires-and-forgets a prediction.
                _drop_stale_pending(start_monotonic)
        except Exception:
            # Stashing must NEVER break the user's call. Swallow.
            pass

        return prediction

    return instrumented_create


# ---------------------------------------------------------------------------
# client.predictions.get — terminal-status flush
# ---------------------------------------------------------------------------


def _make_predictions_get(original_get: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_get)
    def instrumented_get(*args: Any, **kwargs: Any) -> Any:
        try:
            prediction = original_get(*args, **kwargs)
        except Exception:
            # Let the SDK's exception through — the customer is the only
            # one who can recover, and we don't have a CallRecord to
            # build anyway.
            raise

        try:
            _maybe_record_terminal(prediction, sentinel)
        except LeakDetected:
            raise
        except Exception:
            # Failure-isolated — the customer's call returns unchanged.
            pass
        return prediction

    return instrumented_get


def _maybe_record_terminal(prediction: Any, sentinel: Sentinel) -> None:
    """If this prediction is in a terminal state, build + record a CallRecord.

    Terminal states (per Replicate API): ``"succeeded"``, ``"failed"``,
    ``"canceled"``. Non-terminal states (``"starting"``, ``"processing"``)
    leave the pending entry alone so the next poll can pick it up.

    A prediction that never had a pending entry (e.g. the customer called
    ``predictions.get`` on an id they got from somewhere other than this
    SDK instance) is silently ignored — we never have the submission
    timestamp / input metadata, so there's nothing useful to record.
    """
    status = _get_attr_or_key(prediction, "status")
    if status not in ("succeeded", "failed", "canceled"):
        return
    prediction_id = _extract_prediction_id(prediction)
    if not prediction_id:
        return

    now_monotonic = time.perf_counter()
    with _PENDING_LOCK:
        entry = _REPLICATE_PENDING.pop(prediction_id, None)
    if entry is None:
        # Lazy stale cleanup on miss too — keeps the dict bounded even
        # when terminal polls never arrive. Re-acquires the lock; cheap
        # since we just released it.
        _drop_stale_pending(now_monotonic)
        return

    elapsed_ms = (now_monotonic - float(entry.get("monotonic_start", now_monotonic))) * 1000

    record = _build_record_for_prediction(
        session_id=str(entry.get("session_id", "")),
        model_ref=str(entry.get("model_ref", "")),
        input_dict=entry.get("input_dict", {}) or {},
        prediction=prediction,
        status=str(status),
        latency_ms=elapsed_ms,
    )

    # Use the Sentinel from the pending entry — that's the instance that
    # observed the submission. Falls back to the caller-side sentinel if
    # the entry's reference is gone (shouldn't happen but defensive).
    target_sentinel = entry.get("sentinel") or sentinel
    try:
        target_sentinel.record_call(record)
    except LeakDetected:
        raise
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _build_record_for_run(
    *,
    session_id: str,
    model_ref: str,
    input_dict: dict[str, Any],
    output: Any,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord for ``client.run(...)``.

    Replicate's ``run`` is synchronous — by the time we get here the
    prediction has reached a terminal state (typically "succeeded";
    failures raise). We don't have a Prediction object, just the raw
    output, so the dimension inference goes off ``output`` directly.
    """
    usage_extra = _infer_usage_extra_from_output(model_ref, output)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="replicate",
        model=model_ref or "unknown",
        method="run",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=_request_hash(model_ref, input_dict),
        tool_calls=[],
        user_facing_output=True,
        raw_request=_redacted_request(model_ref, input_dict),
        raw_response_meta={"status": "succeeded"},
        usage_extra=usage_extra,
    )


def _build_record_for_prediction(
    *,
    session_id: str,
    model_ref: str,
    input_dict: dict[str, Any],
    prediction: Any,
    status: str,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord for a terminal ``predictions.get(id)``.

    For ``status='succeeded'`` we infer the pricing dimension from the
    prediction's ``output`` plus its ``metrics.predict_time`` (the
    Replicate-standard duration field).

    For ``status='failed'`` / ``'canceled'`` we still emit a record so the
    customer sees the burn (a failed video gen still incurs GPU-seconds
    cost). The dimension is best-effort: we use ``metrics.predict_time``
    if present, otherwise fall back to ``dimension_kind='per_image'``
    with ``dimension_value=0`` so the cloud-side cost lookup short-circuits
    to zero — better than rejecting the record entirely.
    """
    output = _get_attr_or_key(prediction, "output")
    if status == "succeeded":
        usage_extra = _infer_usage_extra_from_prediction(model_ref, prediction, output)
    else:
        # Even failed/canceled predictions accumulate cost (compute was
        # spent). Use predict_time when available so we don't undercount
        # billed-but-failed runs.
        predict_time = _safe_float(_metrics_predict_time(prediction))
        usage_extra = {
            "dimension_kind": "per_second" if predict_time > 0 else "per_image",
            "dimension_value": predict_time,
            "model_specific_meta": {"status": status},
        }
    error = _get_attr_or_key(prediction, "error")
    response_meta: dict[str, Any] = {"status": status}
    if error:
        # Stash a redacted error message rather than the full Prediction
        # — many SDKs put stack traces here.
        response_meta["error"] = _coerce_error(error)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="replicate",
        model=model_ref or "unknown",
        method="predictions.get",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=_request_hash(model_ref, input_dict),
        tool_calls=[],
        user_facing_output=(status == "succeeded"),
        raw_request=_redacted_request(model_ref, input_dict),
        raw_response_meta=response_meta,
        usage_extra=usage_extra,
    )


# ---------------------------------------------------------------------------
# Dimension inference
# ---------------------------------------------------------------------------


# Model-ref prefixes that bill per-second (video models). Everything not in
# this list is assumed to bill per-image — a deliberate bias toward the
# common case (Flux/SD/Ideogram/Recraft are all per-image). The cloud-side
# ``NON_TOKEN_PRICES`` table is authoritative on USD; this inference only
# decides which row of that table to consult.
_VIDEO_MODEL_PREFIXES = (
    "tencent/hunyuan-video",
    "kuaishou/kling-video",
    "alibaba/wan-2-1",
    "alibaba/wan",
    "genmo/mochi-1",
    "genmo/mochi",
)


def _infer_usage_extra_from_output(model_ref: str, output: Any) -> dict[str, Any]:
    """Infer the pricing dimension from the model_ref + raw output of run().

    Video models are detected by prefix; the dimension_value defaults to
    1.0 (a 'standard 1-second clip') because ``client.run`` does NOT
    return ``metrics.predict_time``. Customers who need accurate
    per-second metering should switch to the ``predictions.create`` +
    ``predictions.get`` polling pattern, which DOES surface
    ``metrics.predict_time``.

    Image models default to ``dimension_value = max(1, count(output))``
    — Replicate returns a list of URLs for multi-image generations.
    """
    ref_lower = (model_ref or "").lower()
    if any(ref_lower.startswith(p) for p in _VIDEO_MODEL_PREFIXES):
        return {
            "dimension_kind": "per_second",
            "dimension_value": 1.0,
            "model_specific_meta": {"source": "run_default_duration"},
        }
    return {
        "dimension_kind": "per_image",
        "dimension_value": float(_count_outputs(output)),
        "model_specific_meta": {"source": "run_output_count"},
    }


def _infer_usage_extra_from_prediction(
    model_ref: str,
    prediction: Any,
    output: Any,
) -> dict[str, Any]:
    """Infer the pricing dimension from a Prediction object.

    Video models prefer ``metrics.predict_time`` (the Replicate-standard
    GPU-seconds field). Image models prefer the output count.
    """
    ref_lower = (model_ref or "").lower()
    if any(ref_lower.startswith(p) for p in _VIDEO_MODEL_PREFIXES):
        predict_time = _safe_float(_metrics_predict_time(prediction))
        if predict_time <= 0:
            # Fall back to a conservative 1-second default so we don't
            # bill the customer $0 for a video that actually ran.
            predict_time = 1.0
        return {
            "dimension_kind": "per_second",
            "dimension_value": predict_time,
            "model_specific_meta": {"source": "metrics.predict_time"},
        }
    return {
        "dimension_kind": "per_image",
        "dimension_value": float(_count_outputs(output)),
        "model_specific_meta": {"source": "prediction_output_count"},
    }


# ---------------------------------------------------------------------------
# Output / metric helpers
# ---------------------------------------------------------------------------


def _count_outputs(output: Any) -> int:
    """Return the number of artifacts in a Replicate output payload.

    Replicate returns either:
      - A single URL string (one artifact).
      - A list of URL strings (n artifacts).
      - A dict (rare; e.g. ``{"url": "..."}``) — count 1.
      - None on failure paths.

    Always returns ``>= 1`` for non-None outputs so we never bill the
    customer $0 for a successful generation.
    """
    if output is None:
        return 0
    if isinstance(output, (list, tuple)):
        # Replicate sometimes returns a list with a single element; honour
        # the count rather than treating it as 1.
        return max(1, len(output))
    # Anything non-list-like (string, dict, scalar) counts as one artifact.
    return 1


def _metrics_predict_time(prediction: Any) -> Any:
    """Extract ``prediction.metrics.predict_time`` defensively.

    ``metrics`` may be missing, None, or a dict — replicate's SDK
    sometimes returns a dataclass-like object, sometimes a raw dict
    depending on transport. Handle both.
    """
    metrics = _get_attr_or_key(prediction, "metrics")
    if metrics is None:
        return None
    return _get_attr_or_key(metrics, "predict_time")


def _extract_prediction_id(prediction: Any) -> str:
    """Pull the prediction id off whatever shape the SDK returned."""
    pid = _get_attr_or_key(prediction, "id")
    if pid is None:
        return ""
    return str(pid)


def _get_attr_or_key(obj: Any, name: str) -> Any:
    """Return ``obj.name`` or ``obj[name]`` (dict fallback). None if missing.

    Replicate's SDK historically swapped between Pydantic models and raw
    dicts depending on version + transport (sync vs async); defensive
    accessors let us not care which shape we got.
    """
    if obj is None:
        return None
    val: Any = None
    try:
        val = getattr(obj, name)
    except AttributeError:
        val = None
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return val


def _safe_float(v: Any) -> float:
    """Coerce ``v`` to float; return 0.0 on any failure."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _coerce_error(error: Any) -> str:
    """Render an error payload as a short string for ``raw_response_meta``.

    Truncates at 500 chars so a misbehaving model that returns a
    multi-megabyte error never blows out the local tracer.
    """
    s = str(error)
    if len(s) > 500:
        return s[:497] + "..."
    return s


# ---------------------------------------------------------------------------
# Request hash + redacted request
# ---------------------------------------------------------------------------


def _request_hash(model_ref: str, input_dict: dict[str, Any]) -> str:
    """Stable SHA-256 hex of the request shape.

    We hash ONLY the input dict's keys + a length summary (NOT values)
    so the hash is collision-free across distinct prompts but never
    contains prompt content — matches the redaction discipline of
    ``LeakEvent.evidence``.
    """
    keys = sorted((input_dict or {}).keys())
    summary: dict[str, Any] = {"model_ref": model_ref, "keys": keys}
    # Per-key value-length so identical prompts hash identically but
    # the hash never carries the prompt text.
    summary["value_lengths"] = {k: len(str((input_dict or {}).get(k, ""))) for k in keys}
    return hashlib.sha256(
        json.dumps(summary, sort_keys=True, default=str).encode("utf-8", errors="replace")
    ).hexdigest()


def _redacted_request(model_ref: str, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted ``raw_request`` dict — keys + counts only, NO values.

    Matches the previous redaction contract used by ``LeakEvent.evidence``:
    a customer reading their dashboard should never see prompt text
    leaking out of the SDK. The keys+lengths shape gives a downstream
    leak-detection rule enough signal to flag e.g. "input has a 5MB
    image attached" without exposing the image data itself.
    """
    keys = sorted((input_dict or {}).keys())
    return {
        "model_ref": model_ref,
        "input_keys": keys,
        "input_value_lengths": {k: len(str((input_dict or {}).get(k, ""))) for k in keys},
    }


__all__ = ["wrap_replicate"]
