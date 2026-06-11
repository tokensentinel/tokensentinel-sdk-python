"""Rule: conversational repair loop .

Detects the **cost-aligned symptom of hallucination / grounding failures**:
the agent rewrites the same content under repeated contradicting user
corrections within a short turn window. Each rewrite re-spends prompt
tokens (and any tool / retrieval overhead the agent does before
responding) without converging toward what the user actually asked for.

This is the FinOps-wedge framing of a quality problem. The rule does NOT
attempt to detect *hallucination* per se — that would drift the product
into eval-vendor territory (Arize / Patronus / Galileo). Instead it
catches the **economic shape** of the failure: an agent that re-runs
similar generation under repeated user dissent is burning real dollars
on a path the customer would surface to an engineer if they saw it
plotted on a chart.

Signal
------

We walk the conversation history that the wrapper layer captured in
``CallRecord.raw_request["messages"]`` (Anthropic / OpenAI chat shape:
``[{"role": "user"|"assistant", "content": str | list[block]}]``). The
last call in the session carries the full history including all prior
assistant turns — there is no separate "response_text" field on
``CallRecord``, but the next call's prompt history is functionally
equivalent and is what the wrapper layer already persists.

A *correction-shaped user turn* satisfies two cheap, deterministic
predicates (no model dep, no network):

  1. The user text contains a word-boundary negation/disagreement
     keyword from :data:`_CORRECTION_KEYWORDS` ("no", "not", "wrong",
     "actually", "I meant", "instead", ...).
  2. The user turn is materially shorter than the prior agent turn
     (default ratio 0.8). Corrections in real transcripts are terse
     ("no, I meant Q3") relative to the answer they are rejecting.

A *repair loop* is fired when:

  - ``>= min_corrections`` (default 2) correction-shaped user turns
    appear within the last ``window_turns`` (default 10) turns of the
    conversation, AND
  - the agent's regenerated assistant turns around those corrections
    stay above ``similarity_threshold`` (default 0.7) on pairwise TF-IDF
    character-3-gram cosine similarity. Below 0.7 the agent is
    materially changing the output and the customer is getting value
    out of the back-and-forth; we don't fire.

Confidence
----------

Base **0.65** at exactly ``min_corrections`` corrections. **+0.05** per
additional correction, capped at **0.9**. The cap is deliberate: even
five corrections in a row could plausibly be a user iterating on a
creative spec, so the rule surfaces as "high confidence" rather than
1.0. The cap matches the voice_switching_loop cap for consistency
across the family.

Suggested action
----------------

``surface_correction_pattern_to_engineer`` — the customer's engineering
team is the right audience because the remediation is upstream:
better retrieval grounding, a tighter system prompt, or a small
verifier/reranker before the final answer. The dashboard surfaces a
correction-pattern view that operators can hand to engineering instead
of a halt-the-call intervention.

Privacy
-------

Evidence ships only **structural** details — correction count,
similarity scores, the keywords matched, and the user-turn / agent-turn
character lengths. The raw text is never copied into the evidence dict,
following the  ``sample_args`` redaction precedent: customer leak
handlers may forward events to Slack / Datadog / Sentry, and end-user
chat content must not leak through that path.

False-positive failure mode
---------------------------

The big one is **legitimate iterative refinement** — a user genuinely
clarifying a creative spec over several turns ("no, more like Q3 but
with the West region split out"). The TF-IDF similarity gate is the
primary guardrail: if the agent's output is materially different
across the corrections (cosine < 0.7), the rule does NOT fire. The
secondary guardrail is the length-ratio gate: a user who is writing
*more* in their corrections than the agent wrote in its answer is
collaborating, not correcting. Both gates are tunable per-customer via
the standard ``self.get(...)`` config knobs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Word-boundary negation / disagreement markers. The list is intentionally
# small and conservative — adding a long tail of weak markers ("um", "but",
# "wait") starts catching collaborative-refinement turns that are not
# corrections. Each entry is matched as a regex with ``\b`` boundaries so
# "not" does not fire inside "notation" and "no" does not fire inside
# "noisy". Multi-word entries ("I meant", "that's not right") are escaped
# verbatim — ``\b`` still anchors the leading / trailing word characters.
_CORRECTION_KEYWORDS: tuple[str, ...] = (
    "no",
    "not",
    "wrong",
    "incorrect",
    "that's not right",
    "thats not right",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "shouldn't",
    "shouldnt",
    "isn't",
    "isnt",
    "actually",
    "i meant",
    "instead",
    "rather",
)

_CORRECTION_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _CORRECTION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class RepairLoopRule(Rule):
    """Fires on conversational repair waste .

    See module docstring for full motivation. Operates purely on
    ``session[-1].raw_request["messages"]`` — the wrapper layer already
    captures the conversation history for us, and the most recent call
    carries the longest such history. Rules that walk every CallRecord
    pay an O(N*M) cost; this one is O(M) on the message list of one
    call.
    """

    name = "repair_loop"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window_turns = int(self.get("window_turns", 10))
        min_corrections = int(self.get("min_corrections", 2))
        similarity_threshold = float(self.get("similarity_threshold", 0.7))
        length_ratio = float(self.get("length_ratio", 0.8))

        latest = session[-1]
        messages = _extract_messages(latest)
        if not messages:
            return None

        # Restrict to the last ``window_turns`` turns. This is the
        # "within a short window" gate from the spec — a correction
        # from yesterday is not a current leak.
        recent_turns = messages[-window_turns:]
        turns = _parse_turns(recent_turns)
        if len(turns) < 2:
            return None

        # Walk the turn list: for each user turn that looks like a
        # correction relative to the immediately-prior assistant turn,
        # record a correction event. We then check whether the
        # surrounding agent turns are still similar (i.e. the agent did
        # NOT meaningfully change its output despite the user's
        # disagreement).
        corrections: list[_Correction] = []
        for i, turn in enumerate(turns):
            if turn.role != "user":
                continue
            prior_agent = _last_agent_before(turns, i)
            if prior_agent is None:
                # First user turn has no prior agent output to compare
                # against — by definition not a *correction*.
                continue
            matched = _correction_keywords(turn.text)
            if not matched:
                continue
            # Length-ratio guard: corrections are terse relative to the
            # answer they reject. If the user is writing more than the
            # agent did, they're collaborating, not correcting.
            if not _is_short_relative_to(turn.text, prior_agent.text, ratio=length_ratio):
                continue
            corrections.append(
                _Correction(
                    user_turn_index=i,
                    prior_agent_index=turns.index(prior_agent),
                    matched_keywords=matched,
                )
            )

        if len(corrections) < min_corrections:
            return None

        # Similarity gate: collect the agent turns BEFORE and AFTER each
        # correction, then check pairwise cosine similarity across the
        # set. The agent's regeneration is a repair-loop only if its
        # outputs stay similar despite the user's repeated dissent.
        regen_texts = _collect_regeneration_texts(turns, corrections)
        if len(regen_texts) < 2:
            return None
        mean_similarity = _mean_pairwise_tfidf_cosine(regen_texts, n=3)
        if mean_similarity < similarity_threshold:
            return None

        # Confidence: 0.65 base + 0.05 per correction past the minimum,
        # capped at 0.9. Matches the voice_switching_loop shape.
        extra = len(corrections) - min_corrections
        confidence = min(0.65 + 0.05 * extra, 0.9)

        # Burn estimate: each redundant agent regeneration re-spent the
        # prompt of the call that produced it. Use the latest call's
        # prompt token count as a proxy for the per-rewrite cost; sum
        # across the corrections-1 turns the customer "should have"
        # avoided after the first divergent attempt.
        rewrites_wasted = max(len(corrections) - 1, 0)
        estimated_burn = round(latest.prompt_tokens * 9e-6 * rewrites_wasted, 4)

        return LeakEvent(
            type="repair_loop",
            confidence=confidence,
            project=project,
            session_id=latest.session_id,
            rule="v0.repair_loop",
            evidence={
                "correction_count": len(corrections),
                "window_turns": window_turns,
                "mean_similarity": round(mean_similarity, 3),
                "matched_keywords": _dedup_preserve_order(
                    kw for c in corrections for kw in c.matched_keywords
                )[:5],
                # Structural-only: char lengths, never raw text.
                "user_turn_lengths": [len(turns[c.user_turn_index].text) for c in corrections],
                "agent_turn_lengths": [len(turns[c.prior_agent_index].text) for c in corrections],
                "turns_evaluated": len(turns),
            },
            estimated_burn=estimated_burn,
            suggested_action="surface_correction_pattern_to_engineer",
        )


# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------


class _Turn:
    """A single parsed conversation turn.

    Plain class (not a dataclass) so mypy strict mode doesn't trip on
    the implicit ``__eq__`` / ``__hash__`` machinery — we never compare
    turns by value, only by identity for the ``turns.index(...)``
    lookup in the rule body.
    """

    __slots__ = ("role", "text")

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


class _Correction:
    """A correction event recorded during the rule walk."""

    __slots__ = ("user_turn_index", "prior_agent_index", "matched_keywords")

    def __init__(
        self,
        *,
        user_turn_index: int,
        prior_agent_index: int,
        matched_keywords: list[str],
    ) -> None:
        self.user_turn_index = user_turn_index
        self.prior_agent_index = prior_agent_index
        self.matched_keywords = matched_keywords


def _extract_messages(call: CallRecord) -> list[Any]:
    """Pull the messages list off ``call.raw_request`` defensively.

    Returns an empty list on any malformed shape — a missing key, a
    non-list value, a None ``raw_request``. The rule's contract is to
    degrade silently rather than crash the rule loop.
    """
    raw = getattr(call, "raw_request", None)
    if not isinstance(raw, dict):
        return []
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return []
    return messages


def _parse_turns(messages: list[Any]) -> list[_Turn]:
    """Walk the chat-shape messages list into a flat sequence of turns.

    Handles both Anthropic/OpenAI shapes:

      - ``{"role": "user", "content": "text"}``
      - ``{"role": "assistant", "content": [{"type": "text", "text": "..."}]}``

    Non-dict entries, missing roles, and unknown roles (e.g. ``"system"``
    — we deliberately don't count system prompts as conversation turns,
    they're config) are dropped. Tool / image blocks are ignored: the
    rule operates on text similarity, and a tool_use block has no text
    relevance for the correction-shape signal.
    """
    out: list[_Turn] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _flatten_content(m.get("content"))
        if not text:
            continue
        out.append(_Turn(role=role, text=text))
    return out


def _flatten_content(content: Any) -> str:
    """Reduce a message ``content`` to a single text string.

    String content is returned as-is. List-of-blocks content (Anthropic
    shape, multi-modal OpenAI shape) is reduced to a space-joined
    concatenation of the ``text`` fields. Non-text blocks are skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text_value = block.get("text")
                if isinstance(text_value, str) and text_value:
                    parts.append(text_value)
        return " ".join(parts)
    return ""


def _last_agent_before(turns: list[_Turn], idx: int) -> _Turn | None:
    """Return the closest ``assistant`` turn appearing before index ``idx``.

    Returns ``None`` when no prior assistant turn exists. This is the
    "prior agent output" the correction-shape predicate is evaluated
    against.
    """
    for j in range(idx - 1, -1, -1):
        if turns[j].role == "assistant":
            return turns[j]
    return None


def _last_agent_at_or_after(turns: list[_Turn], idx: int) -> _Turn | None:
    """Return the closest ``assistant`` turn at or after index ``idx``.

    Used to grab the "regenerated" agent output following each
    correction. Returns ``None`` when the conversation history ends on
    a user turn (the agent hasn't responded to the correction yet).
    """
    for j in range(idx, len(turns)):
        if turns[j].role == "assistant":
            return turns[j]
    return None


def _correction_keywords(text: str) -> list[str]:
    """Return the distinct correction keywords matched in ``text``.

    Empty list means the turn is not correction-shaped. Order is
    first-occurrence so the evidence dict reads the way a human would
    scan the turn.
    """
    if not text:
        return []
    matches = _CORRECTION_PATTERN.findall(text)
    if not matches:
        return []
    return _dedup_preserve_order(m.lower() for m in matches)


def _is_short_relative_to(user_text: str, agent_text: str, *, ratio: float) -> bool:
    """True iff ``user_text`` is shorter than ``ratio * len(agent_text)``.

    Strips whitespace before measuring so a corrective like "no." with
    trailing newlines isn't disqualified by a few padding chars. When
    the prior agent text is empty we conservatively return ``False`` —
    we have no baseline to compare against.
    """
    a = len(user_text.strip())
    b = len(agent_text.strip())
    if b == 0:
        return False
    return a < b * ratio


def _collect_regeneration_texts(turns: list[_Turn], corrections: list[_Correction]) -> list[str]:
    """Gather the agent turns we want to compare for similarity.

    For each correction we include the prior agent turn and the
    immediately-following agent turn (if any). The resulting set is
    deduplicated by identity (a single agent turn that sits between two
    corrections appears once, not twice) so pairwise similarity isn't
    skewed by a turn comparing against itself.
    """
    selected: list[_Turn] = []
    seen: set[int] = set()
    for c in corrections:
        prior = turns[c.prior_agent_index]
        if id(prior) not in seen:
            seen.add(id(prior))
            selected.append(prior)
        following = _last_agent_at_or_after(turns, c.user_turn_index + 1)
        if following is not None and id(following) not in seen:
            seen.add(id(following))
            selected.append(following)
    return [t.text for t in selected]


def _dedup_preserve_order(items: Any) -> list[str]:
    """Return a list with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# TF-IDF char-n-gram cosine similarity (lightweight reimplementation)
# ---------------------------------------------------------------------------
#
# Mirrors the algorithm in ``tool_loop._mean_tfidf_charngram`` but kept
# local here so the two rules can evolve independently — tool_loop
# operates on JSON-serialised tool arguments and tunes ngram_size=4;
# repair_loop operates on natural-language agent text and tunes
# ngram_size=3 (shorter n-grams handle paraphrase across sentence
# boundaries better for prose).


def _mean_pairwise_tfidf_cosine(texts: list[str], *, n: int = 3) -> float:
    """Mean pairwise cosine similarity of TF-IDF char-n-gram vectors.

    Returns 0.0 when fewer than 2 non-empty texts are supplied (cosine
    is undefined). Steps:

      1. Lowercase + pad each text with boundary markers so the n-gram
         extractor picks up sub-n-length prefixes / suffixes.
      2. Extract character n-grams of size ``n``.
      3. Compute term frequencies and smoothed inverse document
         frequencies over the local corpus.
      4. Build sparse TF-IDF vectors as ``dict[str, float]`` and
         L2-normalise.
      5. Cosine = dot product (unit-length vectors); average over all
         ``(i, j)`` pairs with ``i < j``.
    """
    docs = [_char_ngrams(t, n=n) for t in texts]
    docs = [d for d in docs if d]
    if len(docs) < 2:
        return 0.0

    df: Counter[str] = Counter()
    for tokens in docs:
        for term in set(tokens):
            df[term] += 1

    n_docs = len(docs)
    idf: dict[str, float] = {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()
    }

    vectors: list[dict[str, float]] = []
    for tokens in docs:
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
            if len(a) > len(b):
                a, b = b, a
            sim = 0.0
            for term, weight in a.items():
                other = b.get(term)
                if other is not None:
                    sim += weight * other
            total += sim
            pairs += 1
    if pairs == 0:
        return 0.0
    return round(total / pairs, 6)


def _char_ngrams(s: str, *, n: int = 3) -> list[str]:
    """Lowercased character n-grams with boundary padding.

    Boundary padding (``\\x01`` markers) lets the n-gram extractor pick
    up prefixes / suffixes shorter than ``n``, mirroring the sklearn
    ``char_wb`` boundary semantics that tool_loop uses.
    """
    if not s:
        return []
    text = s.lower()
    pad = "\x01" * (n - 1)
    padded = pad + text + pad
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]
