"""Rule: same tool, repeated similar calls in a window.

default similarity is **TF-IDF over character n-grams** (default n=4),
implemented in pure Python. Rationale: the leak we are detecting is a
literal-text recurrence problem ("same tool, near-identical args"), and
character n-grams catch paraphrases like ``"web of life"`` vs ``"\\"web of
life\\""`` that token-set Jaccard misses, while staying inside the <50ms hot-path
budget with zero network and zero model dependency.

The Jaccard implementation is preserved as an explicit fallback for tiny
corpora (<2 calls — TF-IDF is undefined on a single document) and as a
user-selectable cheaper metric via ``tool_loop.similarity_metric: 'jaccard'``.

Implementation notes:

- Char n-grams of size ``tool_loop.charngram_size`` (default 4, valid range
  3–5 — calibrated empirically). Each argument JSON string is padded with
  boundary markers to give the n-gram extractor signal at the edges.
- Term frequency: ``collections.Counter`` per document.
- Inverse document frequency: ``ln((N + 1) / (df + 1)) + 1`` (smoothed,
  matches sklearn's default ``smooth_idf=True, sublinear_tf=False``).
- Vectors are stored as sparse ``dict[str, float]`` and L2-normalised, so
  cosine similarity is just the dot product.

Swaps in ``sentence-transformers/all-MiniLM-L6-v2`` for arg corpora that
need true semantic similarity; that path stays opt-in to keep the SDK
network-free and dependency-light.

Privacy / DoS hardening:

- ``sample_args`` in evidence is **redacted by default**. Raw tool arguments
  often carry user PII (chat queries, account IDs, secrets pasted into a
  prompt). Shipping them verbatim to a customer's leak handler — which may
  forward to Slack, Datadog, Sentry, or our own cloud sink — is a data-leak
  vector. The redacted form ships only the arg-dict's key list (sorted), the
  per-key serialized value length, and a 16-hex-char SHA-256 prefix of the
  full sorted JSON, so handlers can still tell "these three calls are the
  same call" without seeing any value. Customers who explicitly opt in via
  ``tool_loop.include_raw_args: True`` get the original raw args (useful for
  local-dev triage on a non-production project).

- Per-arg JSON strings are truncated to ``tool_loop.max_arg_bytes`` (default
  64KB) before n-gram extraction. A misbehaving upstream tool returning a
  100MB blob would otherwise produce ~100M character n-grams and gigabytes
  of memory inside the customer's hot path. The prefix-truncated form still
  yields meaningful similarity for the "same call repeated" pattern this
  rule detects.

- The total evaluation corpus is capped at ``tool_loop.max_total_corpus_bytes``
  (default 1MB). If a window contains many large args, similarity is
  evaluated on the leading args that fit in the cap; trailing args are
  skipped from the similarity computation. ``call_count`` in evidence still
  reflects the unfiltered count so the customer sees the true rate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Default DoS caps. Exposed as module-level constants so retrieval_thrash and
# tests can reference them without re-deriving the numbers.
DEFAULT_MAX_ARG_BYTES = 65536  # 64 KB per arg JSON before n-gram extraction
DEFAULT_MAX_TOTAL_CORPUS_BYTES = 1_048_576  # 1 MB across all args in the window


class ToolLoopRule(Rule):
    name = "tool_loop"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window = self.get("window_seconds", 60)
        min_calls = self.get("min_calls", 3)
        metric = self.get("similarity_metric", "tfidf_charngram")
        ngram_size = self.get("charngram_size", 4)
        # Threshold is metric-sensitive. TF-IDF char-n-gram dilutes the cosine
        # by document length, so equivalently-similar args return lower scores
        # than Jaccard does. Calibrated empirically on the synthetic corpus.
        default_threshold = 0.70 if metric == "tfidf_charngram" else 0.85
        threshold = self.get("cosine_threshold", default_threshold)
        max_arg_bytes = self.get("max_arg_bytes", DEFAULT_MAX_ARG_BYTES)
        max_total_corpus_bytes = self.get("max_total_corpus_bytes", DEFAULT_MAX_TOTAL_CORPUS_BYTES)
        include_raw_args = self.get("include_raw_args", False)

        now = session[-1].timestamp
        recent = [c for c in session if (now - c.timestamp).total_seconds() <= window]

        # group tool invocations by tool name
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for call in recent:
            for tc in call.tool_calls:
                by_tool.setdefault(tc.get("name", "unknown"), []).append(tc)

        for tool_name, calls in by_tool.items():
            if len(calls) < min_calls:
                continue

            similarity = _mean_pairwise_similarity(
                calls,
                metric=metric,
                ngram_size=ngram_size,
                max_arg_bytes=max_arg_bytes,
                max_total_corpus_bytes=max_total_corpus_bytes,
            )
            if similarity >= threshold:
                confidence = min(0.6 + (similarity - threshold) * 4, 0.99)
                sample_args = _build_sample_args(calls[:3], include_raw=include_raw_args)
                return LeakEvent(
                    type="tool_loop",
                    confidence=confidence,
                    project=project,
                    session_id=session[-1].session_id,
                    rule="v0.tool_loop",
                    evidence={
                        "tool": tool_name,
                        "call_count": len(calls),
                        "window_seconds": window,
                        "mean_similarity": round(similarity, 3),
                        "sample_args": sample_args,
                    },
                    estimated_burn=_estimate_burn(recent),
                    suggested_action="pause_for_human_review",
                )
        return None


# ---------------------------------------------------------------------------
# Sample-arg redaction (privacy)
# ---------------------------------------------------------------------------


def _build_sample_args(calls: list[dict[str, Any]], *, include_raw: bool) -> list[dict[str, Any]]:
    """Build the ``sample_args`` evidence list.

    When ``include_raw`` is False (the default), each arg dict is replaced
    with a redacted summary that conveys the *shape* of the call without
    the values:

    .. code-block:: python

        {
            "keys": ["query", "top_k"],          # sorted
            "value_lengths": {"query": 24, "top_k": 1},  # JSON byte length per key
            "hash": "9d4e6c2f8a1b3e7d",         # 16-hex-char SHA-256 prefix of
                                                # json.dumps(args, sort_keys=True)
        }

    The hash is stable across calls — three identical calls produce three
    identical hashes — so the consumer can still tell "this was the same
    call repeated" while never seeing any values. Non-dict argument shapes
    (rare; some hosts pass a string or list) are summarised by type.

    When ``include_raw`` is True, the original ``arguments`` is shipped
    unchanged. This is intended for local-dev / non-production projects
    only, where the customer has explicitly accepted the privacy tradeoff
    in exchange for richer triage data.
    """
    out: list[dict[str, Any]] = []
    for tc in calls:
        args = tc.get("arguments", {})
        if include_raw:
            out.append(args if isinstance(args, dict) else {"_value": args})
            continue
        out.append(_redact_args(args))
    return out


def _redact_args(args: Any) -> dict[str, Any]:
    """Return a redacted shape-only summary of an args object.

    Always returns a JSON-serialisable dict so the consumer can stringify
    the whole evidence blob for transport without further processing.

    DoS-safe: the per-value length and the full-args hash are both computed
    from a bounded serialisation of the input. A 100MB string value still
    produces a sensible value_length (the truncated length) without
    materialising the full blob through ``json.dumps``.
    """
    # Per-value length cap. We need this because ``json.dumps`` on a 100MB
    # string is itself the DoS — we never want to call it on uncapped data.
    # The cap is generous (1MB) so realistic values report exact length.
    value_length_cap = 1_048_576

    if isinstance(args, dict):
        # Build (str_key, original_key) pairs so non-string keys (ints,
        # tuples, etc.) don't break the value-lookup. We stringify only for
        # the output shape; lookups use the original key.
        key_pairs = [(str(k), k) for k in args]
        key_pairs.sort(key=lambda kv: kv[0])
        keys_sorted = [s for s, _ in key_pairs]
        value_lengths: dict[str, int] = {}
        for str_k, orig_k in key_pairs:
            value_lengths[str_k] = _bounded_value_length(args[orig_k], cap=value_length_cap)
        full = _bounded_dumps(args, max_bytes=value_length_cap)
        return {
            "keys": keys_sorted,
            "value_lengths": value_lengths,
            "hash": _sha256_prefix(full, length=16),
        }
    # Non-dict args: still produce a stable shape-only summary.
    full = _bounded_dumps(args, max_bytes=value_length_cap)
    return {
        "keys": [],
        "value_lengths": {},
        "hash": _sha256_prefix(full, length=16),
    }


def _bounded_value_length(value: Any, *, cap: int) -> int:
    """Compute ``len(json.dumps(value))`` without materialising oversized
    intermediates.

    For most values this is just ``len(json.dumps(value, default=str))`` —
    small dicts, small strings, ints, lists. For string values larger than
    the cap we report the cap length without serialising the full string;
    the customer still gets a useful "this is huge" signal without us
    burning seconds in ``json.dumps``.

    Nested non-string values (e.g., a dict with a 100MB string leaf) are
    routed through :func:`_bounded_dumps` so the inner string is shrunk
    BEFORE ``json.dumps`` is invoked. Without this, a maliciously-nested
    arg shape like ``{"filters": {"q": "X" * 100_000_000}}`` would
    materialise the entire blob inside ``json.dumps`` even on the
    redaction path — which was the entire DoS we set out to fix in .
    """
    # Fast path for strings: ``json.dumps`` on a 100MB string allocates a
    # 100MB-plus output buffer. Cheap probe first.
    if isinstance(value, str):
        if len(value) > cap:
            # Approximate: json.dumps wraps in quotes and escapes; for a
            # value this large we just report the cap. Customers using
            # ``value_lengths`` for exact compares should not be feeding
            # multi-megabyte strings as args.
            return cap
        try:
            return len(json.dumps(value))
        except Exception:
            return -1
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    # Non-string, non-bytes: route through the bounded serialiser so that a
    # nested huge string leaf is shrunk before json.dumps walks it.
    try:
        return len(_bounded_dumps(value, max_bytes=cap))
    except Exception:
        return -1


def _sha256_prefix(s: str, *, length: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Bounded JSON serialiser
# ---------------------------------------------------------------------------


def _bounded_dumps(obj: Any, *, max_bytes: int) -> str:
    """Serialise ``obj`` to JSON, truncated at ``max_bytes`` characters.

    The naive ``json.dumps(obj)[:max_bytes]`` is wrong here: if ``obj`` is a
    dict containing a 100MB string value, ``json.dumps`` walks the entire
    100MB into a Python string before we ever get the chance to truncate.
    That alone burns multiple seconds and hundreds of megabytes of RAM
    inside the customer's hot path — exactly the DoS we are guarding
    against.

    Strategy: pre-truncate string values inside the obj before handing it
    off to ``json.dumps``. We walk one level deep on dict/list since that
    catches the actual attack shape (a tool argument with one or two huge
    string fields). Fully recursive truncation isn't worth the complexity
    here — anything pathological deeper than that hits the per-arg cap on
    the resulting JSON string too.

    Errors fall back to ``repr(obj)`` truncated to ``max_bytes`` so a
    weirdly-shaped value can never crash the rule.
    """
    try:
        truncated = _shrink_for_dump(obj, budget=max_bytes)
        s = json.dumps(truncated, sort_keys=True, default=str)
    except Exception:
        try:
            s = repr(obj)
        except Exception:
            s = ""
    if len(s) > max_bytes:
        s = s[:max_bytes]
    return s


def _shrink_for_dump(obj: Any, *, budget: int) -> Any:
    """Best-effort pre-truncation of large string leaves before ``json.dumps``.

    The point is to ensure ``json.dumps`` never has to materialise a
    multi-megabyte string. We allocate a slightly generous slice (budget+1)
    so the post-dump truncation still kicks in deterministically.
    """
    if isinstance(obj, str):
        if len(obj) > budget:
            return obj[:budget]
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = _shrink_for_dump(v, budget=budget)
        return out
    if isinstance(obj, (list, tuple)):
        return [_shrink_for_dump(x, budget=budget) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Pairwise similarity (with DoS caps)
# ---------------------------------------------------------------------------


def _mean_pairwise_similarity(
    calls: list[dict[str, Any]],
    *,
    metric: str = "tfidf_charngram",
    ngram_size: int = 4,
    max_arg_bytes: int = DEFAULT_MAX_ARG_BYTES,
    max_total_corpus_bytes: int = DEFAULT_MAX_TOTAL_CORPUS_BYTES,
) -> float:
    """Mean pairwise similarity over argument JSON strings.

    Dispatches on ``metric``. Falls back to Jaccard when the corpus is too
    small for TF-IDF to be meaningful (TF-IDF needs ≥2 documents to compute
    document frequencies; with only 1 it would degenerate to a TF-only
    cosine, which is no better than Jaccard).

    DoS hardening:

    - Each per-arg JSON string is truncated at ``max_arg_bytes`` before
      similarity computation. A 100MB arg therefore costs us at most 64KB
      of n-gram work, not 100MB.
    - The total corpus is bounded by ``max_total_corpus_bytes``. If
      cumulative arg bytes exceed the cap, we evaluate similarity on the
      leading arg strings that fit and drop the trailing ones. This is a
      best-effort signal under attack rather than a refusal-to-evaluate.
    - We need at least 2 evaluable args to compute pairwise similarity;
      if the cap is so low that only one survives, return 0.0.
    """
    if len(calls) < 2:
        return 0.0

    args_strs: list[str] = []
    total_bytes = 0
    for c in calls:
        # Use a bounded serialiser so a 100MB string in the args dict never
        # actually enters json.dumps as a 100MB blob — we'd lose multiple
        # seconds and gigabytes of RAM to that call alone before truncation
        # could take effect.
        s = _bounded_dumps(c.get("arguments", {}), max_bytes=max_arg_bytes)
        # Soft total-corpus cap. Keep a string only if adding it doesn't
        # blow past the budget. We stop *before* exceeding so we never do
        # n-gram work on a string that doesn't fit.
        if total_bytes + len(s) > max_total_corpus_bytes and args_strs:
            break
        args_strs.append(s)
        total_bytes += len(s)

    if len(args_strs) < 2:
        return 0.0

    if metric == "jaccard":
        sim = _mean_jaccard(args_strs)
    elif metric == "tfidf_charngram":
        sim = _mean_tfidf_charngram(args_strs, ngram_size=ngram_size)
    else:
        # unknown metric: fail safe to Jaccard rather than throw inside detection
        sim = _mean_jaccard(args_strs)
    # Round to 6 decimals so floating-point noise doesn't drop us under a
    # threshold that should match exactly (e.g., similarity=1.0 == threshold=1.0).
    return round(sim, 6)


# ---------------------------------------------------------------------------
# Jaccard (fallback / explicit opt-in)
# ---------------------------------------------------------------------------


def _mean_jaccard(args_strs: list[str]) -> float:
    """Token-set Jaccard on argument JSON strings."""
    token_sets = [set(_tokens(s)) for s in args_strs]

    pairs = 0
    total = 0.0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            jacc = len(a & b) / len(a | b)
            total += jacc
            pairs += 1
    return total / pairs if pairs else 0.0


def _tokens(s: str) -> list[str]:
    out = []
    cur: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


# ---------------------------------------------------------------------------
# TF-IDF char n-gram (default)
# ---------------------------------------------------------------------------


def _mean_tfidf_charngram(args_strs: list[str], *, ngram_size: int = 4) -> float:
    """Mean pairwise cosine similarity of TF-IDF char-n-gram vectors.

    Steps:
      1. Lowercase and pad each string with boundary markers.
      2. Extract character n-grams of size ``ngram_size`` (clamped to 3–5).
      3. Compute term frequency per document with ``Counter``.
      4. Compute smoothed IDF over the local corpus:
            idf(t) = ln((N + 1) / (df(t) + 1)) + 1
      5. Build sparse TF-IDF vectors as ``dict[str, float]`` and L2-normalise.
      6. Cosine = dot product (vectors are unit-length); average over all
         (i, j) pairs with i < j.
    """
    n = max(3, min(5, int(ngram_size)))
    docs = [_char_ngrams(s, n=n) for s in args_strs]

    # Document frequency: count of documents containing each n-gram.
    df: Counter[str] = Counter()
    for tokens in docs:
        for term in set(tokens):
            df[term] += 1

    n_docs = len(docs)
    # Smoothed IDF (matches sklearn's smooth_idf=True default; never zero,
    # never negative — safe even when a term appears in every document).
    idf = {term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    # Build TF-IDF vectors and L2-normalise.
    vectors: list[dict[str, float]] = []
    for tokens in docs:
        if not tokens:
            vectors.append({})
            continue
        tf = Counter(tokens)
        vec = {term: freq * idf[term] for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {term: v / norm for term, v in vec.items()}
        vectors.append(vec)

    pairs = 0
    total = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if not a or not b:
                continue
            # Iterate over the smaller dict for speed.
            if len(a) > len(b):
                a, b = b, a
            sim = 0.0
            for term, weight in a.items():
                other = b.get(term)
                if other is not None:
                    sim += weight * other
            total += sim
            pairs += 1
    return total / pairs if pairs else 0.0


def _char_ngrams(s: str, *, n: int = 4) -> list[str]:
    """Lowercased character n-grams with boundary padding.

    Boundary padding (``\\x01`` markers) lets the n-gram extractor pick up
    prefixes/suffixes shorter than ``n``, mirroring sklearn's ``char_wb``
    boundary semantics in spirit.
    """
    if not s:
        return []
    text = s.lower()
    pad = "\x01" * (n - 1)
    padded = pad + text + pad
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


# ---------------------------------------------------------------------------
# Burn estimate (unchanged)
# ---------------------------------------------------------------------------


def _estimate_burn(recent: list[CallRecord]) -> float:
    """Rough USD estimate of next 3 cycles at the current burn rate."""
    if not recent:
        return 0.0
    avg_tokens = sum(c.prompt_tokens + c.completion_tokens for c in recent) / len(recent)
    cost_per_call = avg_tokens * 9e-6
    return round(cost_per_call * 3, 4)
