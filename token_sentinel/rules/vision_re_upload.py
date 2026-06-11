"""Rule: same image uploaded across multiple consecutive vision calls.

Vision-modality analog of :mod:`embedding_waste`. The leak: customer's
agent ships the same screenshot/document/photo on every turn of a
conversation instead of caching the OCR/description result locally or
reusing the provider's attachment-id surface (Anthropic file uploads,
OpenAI image-id, Gemini file URI). At Anthropic's ~1,600-tokens-per-1568px
image rate this is materially expensive: a 10-turn agent that re-uploads
the same 5-image screenshot bundle each turn burns ~80,000 tokens per
session that should have been ~8,000.

Why client-side hashing
=======================

Providers don't return a "this is the same image you sent last time"
signal. Anthropic / OpenAI / Gemini all charge per-call for the image
tokens regardless of payload similarity. The only way to detect the
pattern is to hash the bytes we see on the way out (or as base64 in the
request), then look for repeats within a session window.

Hash strategy: **two-tier**. The primary signal is exact-byte SHA-256
(zero deps, matches ``embedding_waste`` precedent). Per internal research
§ 8 → "Recommend starting with exact-byte hash; upgrade to pHash
judge-augmented path."  adds a **perceptual-hash fallback** (pHash
via the ``imagehash`` library) so the rule keeps firing when a minor
recompression / resize between turns made the bytes different but the
image is visibly the same screenshot. The perceptual path is an optional
dependency (``token-sentinel[vision-perceptual]``); when ``imagehash`` /
``Pillow`` are not installed the rule degrades to default behaviour
(exact-byte only).

Two-tier ranking semantics
==========================

When both signals would fire on the same session, the **exact** match
wins: it's a higher-precision signal (the bytes are truly identical, no
ambiguity), so we keep its base confidence rather than discounting it.
The perceptual signal only fires when the exact path turns up nothing —
the rule walks the latest call's images, checks SHA-256 against the
recent window first, and only falls through to phash comparison if no
exact-match consecutive run reaches ``min_calls``. Perceptual matches
take a fixed ``-0.05`` confidence penalty relative to the exact base to
reflect the lower-precision nature of the signal (Hamming distance ≤ 6
is "visually similar"; a low-content image like a blank screenshot can
collide perceptually with another blank screenshot that is genuinely a
different upload).

Provider request shapes
=======================

The wrapper-captured ``raw_request`` differs by provider. The extractor
walks each ``messages[*]`` (Anthropic / OpenAI) or top-level ``contents``
(Gemini) entry and pulls image bytes from the standard locations:

- **OpenAI** (``wrappers/openai.py`` builds ``raw_request={"messages":
  ..., "tools": ..., "max_tokens": ...}``):

      messages[i].content[j] = {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBOR..."}
      }

  Base64-data URLs are the supported in-band shape; ``http://`` URLs
  are out of band and we cannot hash them client-side without
  pulling them — out of scope for the rule.

- **Anthropic** (``wrappers/anthropic.py`` builds the same
  ``raw_request={"messages": ..., "tools": ..., "max_tokens": ...}``):

      messages[i].content[j] = {
          "type": "image",
          "source": {"type": "base64", "media_type": "image/png",
                     "data": "iVBOR..."}
      }

- **Gemini** (``wrappers/gemini.py`` builds
  ``raw_request={"model": ..., "contents": ..., "tools": ...}``):

      contents[i].parts[j].inline_data.data = "iVBOR..."
      (or)
      contents[i] = {"inline_data": {"data": "iVBOR...", ...}}

  Gemini's ``Part.from_uri()`` (file_uri) is out of band and not
  hashed here.

We hash the **raw base64 string** rather than decoding to bytes first
for the SHA-256 primary signal: this avoids paying the base64 decode
cost on hot paths AND keeps the hash invariant to base64 padding/case
(which would change if a producer re-encoded the same image bytes
through a different base64 library). The result is functionally
equivalent to "SHA-256 of the bytes the customer asked us to ship" —
exactly what we want to detect repetition of. For the **perceptual**
fallback we must decode the base64 to actual image bytes (PIL needs to
parse the pixels), but the SHA-256 result is computed first and
short-circuits the phash path on a hit, so we don't pay the decode cost
on the common path.

Confidence
==========

- **Exact match** (SHA-256): Base 0.7 (matches ``model_misroute``). Each
  additional duplicate beyond the threshold adds +0.1, capped at 0.99.
- **Perceptual match** (phash, when exact misses): Base 0.65 (exact
  base - 0.05). Same +0.1-per-extra-duplicate scaling, capped at 0.99.

DoS hardening
=============

Per-image bytes are capped at ``vision_re_upload.max_image_bytes``
(default 5 MB) before hashing. A 100MB base64 blob would otherwise eat
both CPU and memory on what is meant to be a hot-path observation. The
cap is generous enough that legitimate frontier-model image uploads
(~1568px JPEG ~ 800 KB base64 ~ 1.1 MB) fit comfortably; pathological
inputs are still hashed but only over the leading 5MB, which still
yields a stable repeat signal. The same cap applies to the decoded
bytes fed into PIL for the perceptual path.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Optional perceptual-hash dependency. The rule degrades to exact-byte
# matching when these aren't installed — see ``_perceptual_hash`` for
# the graceful-fallback contract.
try:
    import imagehash
    from PIL import Image  # type: ignore[import-untyped]

    _PERCEPTUAL_AVAILABLE = True
except ImportError:
    _PERCEPTUAL_AVAILABLE = False

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per image, see module docstring

# Hamming-distance threshold below which two phashes count as "visually
# similar" for the perceptual fallback. The phash output is a 64-bit
# hash, so the distance space is [0, 64]. 6 is the value the imagehash
# README recommends for "near-identical" detection (resize / mild
# recompression); higher values start picking up genuinely different
# images that share a colour palette.
_PERCEPTUAL_MAX_DISTANCE = 6


class VisionReUploadRule(Rule):
    name = "vision_re_upload"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window = self.get("window_seconds", 60)
        min_calls = self.get("min_calls", 3)
        max_image_bytes = self.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)

        now = session[-1].timestamp
        recent = [c for c in session if (now - c.timestamp).total_seconds() <= window]
        if len(recent) < min_calls:
            return None

        # Walk each call and collect the set of (sha256, phash_or_None)
        # records for each image found. A call may contain multiple
        # images; we count a call as "re-uploading hash H" iff H appears
        # in the call's set. Tracking by-call (rather than by-image-
        # instance across the whole window) keeps the firing semantics
        # aligned with the docstring: "the same image was uploaded in N
        # consecutive calls."
        per_call_images: list[list[_ImageRecord]] = []
        for c in recent:
            try:
                records = _extract_image_records(c, max_bytes=max_image_bytes)
            except Exception:
                records = []
            per_call_images.append(records)

        latest_records = per_call_images[-1]
        if not latest_records:
            return None

        # ---- Primary signal: exact-byte SHA-256 match ----
        exact_event = _try_exact_match(
            per_call_images=per_call_images,
            latest_records=latest_records,
            min_calls=min_calls,
            window=window,
            recent=recent,
            project=project,
        )
        if exact_event is not None:
            return exact_event

        # ---- Fallback: perceptual hash (phash) match ----
        # Only attempt if imagehash is installed AND at least one image
        # in the latest call produced a phash (decode succeeded).
        if not _PERCEPTUAL_AVAILABLE:
            return None
        return _try_perceptual_match(
            per_call_images=per_call_images,
            latest_records=latest_records,
            min_calls=min_calls,
            window=window,
            recent=recent,
            project=project,
        )


# ---------------------------------------------------------------------------
# Match strategies
# ---------------------------------------------------------------------------


def _try_exact_match(
    *,
    per_call_images: list[list[_ImageRecord]],
    latest_records: list[_ImageRecord],
    min_calls: int,
    window: int,
    recent: list[CallRecord],
    project: str,
) -> LeakEvent | None:
    """Look for an SHA-256 image hash that repeats in ``min_calls``
    consecutive calls anchored on the latest. First hit wins."""
    per_call_sha = [{r.sha256 for r in rs} for rs in per_call_images]
    latest_hashes = per_call_sha[-1]
    if not latest_hashes:
        return None
    for image_hash in latest_hashes:
        consecutive = 0
        for call_hashes in reversed(per_call_sha):
            if image_hash in call_hashes:
                consecutive += 1
            else:
                break
        if consecutive < min_calls:
            continue
        # Confidence: 0.7 base + 0.1 per additional duplicate beyond
        # threshold, capped at 0.99. min_calls=3 with 3 dupes is 0.7;
        # 4 dupes is 0.8; 6 dupes is 0.99 (clamped).
        extra = consecutive - min_calls
        confidence = min(0.7 + extra * 0.1, 0.99)
        return LeakEvent(
            type="vision_re_upload",
            confidence=confidence,
            project=project,
            session_id=recent[-1].session_id,
            rule="v0.vision_re_upload",
            evidence={
                # 16-hex-char prefix mirrors ``embedding_waste`` —
                # enough to dedupe in dashboards, not enough to
                # carry the full bytes back to the customer's
                # handler / log sink.
                "image_hash": image_hash[:16],
                "duplicate_count": consecutive,
                "window_seconds": window,
                "provider": recent[-1].provider,
                "match_type": "exact",
                "perceptual_distance": 0,
            },
            estimated_burn=round((consecutive - 1) * 5e-3, 4),
            suggested_action="cache_image_locally_or_reuse_attachment_id",
        )
    return None


def _try_perceptual_match(
    *,
    per_call_images: list[list[_ImageRecord]],
    latest_records: list[_ImageRecord],
    min_calls: int,
    window: int,
    recent: list[CallRecord],
    project: str,
) -> LeakEvent | None:
    """Look for a phash in the latest call that has a near-neighbour
    (Hamming distance <= ``_PERCEPTUAL_MAX_DISTANCE``) in each of
    ``min_calls`` consecutive prior calls. First hit wins; the *smallest*
    cross-call distance among the matched run is recorded so dashboards
    can see how tight the signal was."""
    latest_phashes = [r.phash for r in latest_records if r.phash is not None]
    if not latest_phashes:
        return None
    for phash in latest_phashes:
        consecutive = 0
        best_distance = 0  # 0 for the self-match against the latest call
        for call_records in reversed(per_call_images):
            # In each earlier call, find the closest phash to ``phash``
            # and accept the call iff any image is within threshold.
            distances = [
                _phash_distance(phash, r.phash) for r in call_records if r.phash is not None
            ]
            if not distances:
                break
            min_distance = min(distances)
            if min_distance > _PERCEPTUAL_MAX_DISTANCE:
                break
            consecutive += 1
            # Track the worst (largest) distance across the matched run —
            # this is what dashboards see as "how tight was the visual
            # match across the consecutive uploads". A run with distances
            # [0, 4, 5] reports 5.
            if min_distance > best_distance:
                best_distance = min_distance
        if consecutive < min_calls:
            continue
        # Perceptual base = exact base (0.7) - 0.05 = 0.65. Same +0.1
        # scaling so a 4-dupe perceptual hits 0.75, 5-dupe 0.85, 6-dupe
        # 0.95, 7+ caps at 0.99.
        extra = consecutive - min_calls
        confidence = min(0.65 + extra * 0.1, 0.99)
        return LeakEvent(
            type="vision_re_upload",
            confidence=confidence,
            project=project,
            session_id=recent[-1].session_id,
            rule="v0.vision_re_upload",
            evidence={
                # The phash is itself only 16 hex chars (64-bit), so we
                # ship the whole thing — there's no privacy benefit to
                # truncating and dashboards may want to render the full
                # value for debugging visually-similar pairs.
                "image_hash": phash,
                "duplicate_count": consecutive,
                "window_seconds": window,
                "provider": recent[-1].provider,
                "match_type": "perceptual",
                "perceptual_distance": best_distance,
            },
            estimated_burn=round((consecutive - 1) * 5e-3, 4),
            suggested_action="cache_image_locally_or_reuse_attachment_id",
        )
    return None


# ---------------------------------------------------------------------------
# Image-byte extraction
# ---------------------------------------------------------------------------


class _ImageRecord:
    """Per-image telemetry pulled out of a call's raw_request.

    Carries both the exact SHA-256 (the primary signal, hashed over
    the base64 string) and the perceptual hash ( fallback, hashed
    over the decoded pixels via PIL+imagehash). ``phash`` is ``None``
    when the perceptual dependency is missing, when the bytes were not
    valid base64, or when PIL could not parse the image (malformed
    payload, unsupported format, etc.) — the rule treats this as
    "perceptual signal unavailable for this image" and continues with
    the SHA-256 signal alone.
    """

    __slots__ = ("sha256", "phash")

    def __init__(self, sha256: str, phash: str | None) -> None:
        self.sha256 = sha256
        self.phash = phash


def _extract_image_records(call: CallRecord, *, max_bytes: int) -> list[_ImageRecord]:
    """Walk ``call.raw_request`` and return one :class:`_ImageRecord`
    per image-byte payload found, across all three supported providers.

    The walk is provider-agnostic: we look at the standard locations
    used by Anthropic / OpenAI / Gemini wrapper outputs and pick up
    every shape we recognise. An unrecognised shape is skipped silently
    (no logging) so the rule degrades gracefully on future wrapper
    additions or hand-built ``raw_request`` payloads.

    The list may contain duplicate ``sha256`` entries (e.g., two copies
    of the same image attached to one call); the rule de-duplicates
    those at the per-call set level before counting consecutive runs.
    """
    raw = getattr(call, "raw_request", None)
    if not isinstance(raw, dict):
        return []

    records: list[_ImageRecord] = []

    # Anthropic / OpenAI path — ``raw_request['messages']`` is a list of
    # dicts; each dict's ``content`` can be a string or a list of blocks.
    for message in raw.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                _collect_block_records(block, records, max_bytes=max_bytes)

    # Gemini path — ``raw_request['contents']`` is either a string (text-
    # only call), a list of strings/Parts, or a list of dicts with
    # ``parts``. We walk all the shapes the google-genai SDK can emit.
    contents = raw.get("contents")
    _walk_gemini_contents(contents, records, max_bytes=max_bytes)

    return records


def _extract_image_hashes(call: CallRecord, *, max_bytes: int) -> set[str]:
    """Back-compat wrapper used by the unit tests: return only the
    SHA-256 hex digests, ignoring the perceptual records. New code
    should call :func:`_extract_image_records` directly.
    """
    return {r.sha256 for r in _extract_image_records(call, max_bytes=max_bytes)}


def _collect_block_records(block: Any, out: list[_ImageRecord], *, max_bytes: int) -> None:
    """Pull image-byte hashes out of a single OpenAI/Anthropic content block."""
    if not isinstance(block, dict):
        return
    block_type = block.get("type")

    # OpenAI shape: {"type": "image_url", "image_url": {"url": "data:..."}}
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                payload = _extract_data_url_payload(url)
                if payload is not None:
                    out.append(_record_from_b64(payload, max_bytes=max_bytes))

    # Anthropic shape: {"type": "image", "source": {"type": "base64",
    #                   "media_type": "image/png", "data": "iVBOR..."}}
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict):
            data = source.get("data")
            if isinstance(data, str) and data:
                out.append(_record_from_b64(data, max_bytes=max_bytes))

    # Gemini "inline_data" shape inside an OpenAI/Anthropic-looking
    # blocks list (rare but seen when customers hand-build a unified
    # multimodal shape and pass it to multiple providers).
    inline = block.get("inline_data") or block.get("inlineData")
    if isinstance(inline, dict):
        data = inline.get("data")
        if isinstance(data, str) and data:
            out.append(_record_from_b64(data, max_bytes=max_bytes))


def _walk_gemini_contents(contents: Any, out: list[_ImageRecord], *, max_bytes: int) -> None:
    """Walk the (possibly nested) ``contents`` shape and pull inline_data.

    google-genai accepts ``contents`` as:

      - a bare string (text-only — no images to find)
      - a list of strings + ``Part`` objects + dicts
      - a dict with ``role`` + ``parts: [...]`` keys
      - a list of those dicts

    We walk shallowly: one level of list, one level of ``parts``. Deeper
    nesting isn't part of the documented shape and would only appear in
    pathological customer-built payloads.
    """
    if contents is None:
        return
    if isinstance(contents, str):
        return
    if isinstance(contents, dict):
        _walk_gemini_content_entry(contents, out, max_bytes=max_bytes)
        return
    if isinstance(contents, list):
        for entry in contents:
            _walk_gemini_content_entry(entry, out, max_bytes=max_bytes)


def _walk_gemini_content_entry(entry: Any, out: list[_ImageRecord], *, max_bytes: int) -> None:
    """Process a single entry in a Gemini ``contents`` list/dict."""
    if entry is None or isinstance(entry, str):
        return
    if isinstance(entry, dict):
        # Direct {"inline_data": {"data": "..."}} shape — some customers
        # bypass the parts wrapper.
        inline = entry.get("inline_data") or entry.get("inlineData")
        if isinstance(inline, dict):
            data = inline.get("data")
            if isinstance(data, str) and data:
                out.append(_record_from_b64(data, max_bytes=max_bytes))
        # Standard {"role": ..., "parts": [...]} shape.
        parts = entry.get("parts")
        if isinstance(parts, list):
            for part in parts:
                _walk_gemini_part(part, out, max_bytes=max_bytes)
        return
    # ``Part`` objects expose ``inline_data`` as an attribute. Pull via
    # ``getattr`` so we work without importing google-genai.
    _walk_gemini_part(entry, out, max_bytes=max_bytes)


def _walk_gemini_part(part: Any, out: list[_ImageRecord], *, max_bytes: int) -> None:
    """Pull inline_data off a single Gemini Part (dict or SDK object)."""
    if part is None or isinstance(part, str):
        return
    inline = None
    if isinstance(part, dict):
        inline = part.get("inline_data") or part.get("inlineData")
    else:
        # google-genai Part exposes ``.inline_data`` as an attribute.
        inline = getattr(part, "inline_data", None)
        if inline is None:
            inline = getattr(part, "inlineData", None)
    if inline is None:
        return
    # ``inline`` may itself be a dict or an SDK Blob object.
    data = inline.get("data") if isinstance(inline, dict) else getattr(inline, "data", None)
    if isinstance(data, str) and data:
        out.append(_record_from_b64(data, max_bytes=max_bytes))
    elif isinstance(data, (bytes, bytearray)):
        # Some SDK paths surface the raw bytes; hash directly.
        out.append(_record_from_bytes(bytes(data), max_bytes=max_bytes))


def _extract_data_url_payload(url: str) -> str | None:
    """Return the base64 payload of a ``data:image/...;base64,XXX`` URL.

    Returns ``None`` for non-data URLs (``http://`` / ``https://``); we
    don't fetch remote images. Also returns ``None`` for malformed
    ``data:`` URLs that lack the ``;base64,`` separator.
    """
    if not url.startswith("data:"):
        return None
    # Standard shape: ``data:<mediatype>;base64,<payload>``. Some
    # producers omit ``;base64`` and inline percent-encoded bytes; we
    # don't try to handle those — they're rare and the rule's job is
    # to detect repetition, not to be a general data-URL parser.
    marker = ";base64,"
    idx = url.find(marker)
    if idx < 0:
        return None
    return url[idx + len(marker) :]


def _record_from_b64(data: str, *, max_bytes: int) -> _ImageRecord:
    """Build an :class:`_ImageRecord` from a base64 image string.

    The SHA-256 leg is computed against the (possibly truncated) base64
    string — the hash contract. The perceptual leg decodes to bytes
    and runs ``imagehash.phash`` against the PIL image; failures (bad
    base64, malformed image, optional dep missing) silently produce
    ``phash=None``.
    """
    return _ImageRecord(
        sha256=_hash_b64(data, max_bytes=max_bytes),
        phash=_perceptual_hash_from_b64(data, max_bytes=max_bytes),
    )


def _record_from_bytes(data: bytes, *, max_bytes: int) -> _ImageRecord:
    """Build an :class:`_ImageRecord` from raw image bytes."""
    return _ImageRecord(
        sha256=_hash_bytes(data, max_bytes=max_bytes),
        phash=_perceptual_hash(data, max_bytes=max_bytes),
    )


def _hash_b64(data: str, *, max_bytes: int) -> str:
    """SHA-256 of a base64 image string, truncated at ``max_bytes`` chars.

    See module docstring on DoS hardening.
    """
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return hashlib.sha256(data.encode("ascii", errors="replace")).hexdigest()


def _hash_bytes(data: bytes, *, max_bytes: int) -> str:
    """SHA-256 of raw image bytes, truncated at ``max_bytes`` bytes."""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------


def _perceptual_hash_from_b64(data: str, *, max_bytes: int) -> str | None:
    """Decode a base64 image string and run ``imagehash.phash``.

    Returns ``None`` if:

    - the perceptual dependency isn't installed (``_PERCEPTUAL_AVAILABLE``
      is False),
    - the base64 payload fails to decode (corruption, missing padding),
    - the resulting bytes aren't a parseable image (PIL raises),
    - any unexpected exception fires inside imagehash.

    The contract is "best effort; never propagate exceptions" — failures
    here only mean "we lose the phash signal for this one image", which
    is strictly less bad than the entire rule loop crashing.
    """
    if not _PERCEPTUAL_AVAILABLE:
        return None
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        return None
    return _perceptual_hash(raw, max_bytes=max_bytes)


def _perceptual_hash(image_bytes: bytes, *, max_bytes: int) -> str | None:
    """Return ``imagehash.phash(image)`` as a hex string, or ``None``.

    Mirrors :func:`_perceptual_hash_from_b64` for the case where the
    upstream extractor surfaced raw bytes (some Gemini Blob shapes).
    Caps the input at ``max_bytes`` for the same DoS reasons as
    SHA-256, then hands the bytes to PIL via ``BytesIO``.
    """
    if not _PERCEPTUAL_AVAILABLE:
        return None
    if len(image_bytes) > max_bytes:
        image_bytes = image_bytes[:max_bytes]
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # imagehash returns an ``ImageHash`` instance whose ``__str__``
        # is a 16-char hex string (64-bit hash). We pin the str form
        # because it's stable across imagehash versions and gives us a
        # clean comparand for the Hamming-distance helper.
        return str(imagehash.phash(img))
    except Exception:
        # Malformed images, unsupported formats, OOM under PIL —
        # everything degrades silently to "no perceptual signal".
        return None


def _phash_distance(a: str, b: str) -> int:
    """Hamming distance between two phash hex strings.

    imagehash produces a 64-bit hash rendered as 16 hex chars. We XOR
    the two integer values and count the set bits — equivalent to the
    bit-difference count between the two underlying hashes. A return of
    0 means "identical phash" (visually indistinguishable); imagehash's
    own README treats ``<= 6`` as "almost certainly the same image" and
    ``> 10`` as "different image".

    Returns ``64`` (the maximum possible distance) on any parse error
    so a malformed phash never accidentally satisfies the threshold.
    """
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (ValueError, TypeError):
        return 64
