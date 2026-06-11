"""Tests for MED-7: ``model_misroute`` keyword matching uses word boundaries.

The v0.2.0 implementation matched keywords as plain substrings of the
lowercased flattened prompt. That was too loose — ``"is this"`` (formerly in
the keyword list) matched on perfectly innocuous prompts like ``"is this code
correct"``. v0.3.2 swaps in word-boundary regex matching and tightens the
keyword list:

- Drops bare ``"is this"``.
- Adds ``"is this a"`` and ``"is this an"`` (unambiguously classification-
  shaped).
- All other keywords still match but only at word boundaries.

These tests pin the new behaviour so a future cleanup pass can't silently
re-loosen the matcher.
"""

from __future__ import annotations

import pytest

from token_sentinel.rules.model_misroute import (
    CLASSIFY_KEYWORDS,
    ModelMisrouteRule,
)

# ---------------------------------------------------------------------------
# Word boundary fires correctly on classification prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "Please classify this email.",
        "categorize the following document",
        "label this image briefly",
        "which category does this fit?",
        "yes or no: is the meeting useful",
        "true or false: Paris is in Spain",
        "rate from 1 to 10 the quality",
        "rate this on a scale of 1-5",
        "is this a positive review?",
        "is this an error message?",
    ],
)
def test_word_boundary_fires_on_classification_prompts(make_call, prompt):
    """Each canonical classification prompt should fire the rule."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": prompt}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None, f"expected fire on {prompt!r}"


# ---------------------------------------------------------------------------
# Word boundary does NOT fire on substring-only matches inside other words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        # "classify" should NOT match inside "declassified" / "reclassify".
        "Please summarise this declassified memo briefly",
        "Reclassifying old documents for the archive",
        # "label this" should NOT match "labels this is a long word" (no boundary).
        # Use a real false-positive shape: the substring is split mid-word.
        "The labelling system is broken",
        # "categorize" should NOT match "miscategorized" without word boundary.
        "The miscategorized records need cleanup",
    ],
)
def test_word_boundary_does_not_fire_inside_other_words(make_call, prompt):
    """Keywords appearing as substrings inside larger words must not fire."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": prompt}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None, (
        f"unexpected fire on {prompt!r}"
    )


# ---------------------------------------------------------------------------
# "is this code correct" no longer fires (the canonical MED-7 false positive)
# ---------------------------------------------------------------------------


def test_is_this_code_correct_no_longer_fires(make_call):
    """The canonical MED-7 false-positive prompt must NOT fire.

    Pre-fix: ``"is this"`` was a substring keyword, so this prompt fired.
    Post-fix: ``"is this"`` is removed from the keyword list and ``"is this
    a"`` / ``"is this an"`` only match before an article — so a bare ``"is
    this code correct"`` no longer trips the rule.
    """
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "is this code correct"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_is_this_what_you_want_no_longer_fires(make_call):
    """A second common false-positive shape ('is this what you want?')."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Is this what you want?"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# "is this a positive review" still fires
# ---------------------------------------------------------------------------


def test_is_this_a_positive_review_still_fires(make_call):
    """The genuine classification cue ('is this a X') must still fire."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "is this a positive review"}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert "is this a" in ev.evidence["matched_keywords"]


def test_is_this_an_error_message_still_fires(make_call):
    """The 'an' variant of the classification cue."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "is this an error message"}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert "is this an" in ev.evidence["matched_keywords"]


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected_kw",
    [
        ("CLASSIFY THIS EMAIL", "classify"),
        ("Classify This Email", "classify"),
        ("Is This A Positive Review", "is this a"),
        ("YES OR NO: was it useful", "yes or no"),
    ],
)
def test_case_insensitive_matching(make_call, prompt, expected_kw):
    """Word-boundary matcher must be case-insensitive."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": prompt}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None, f"expected fire on {prompt!r}"
    assert expected_kw in ev.evidence["matched_keywords"], (
        f"expected {expected_kw!r} in matched_keywords, got {ev.evidence['matched_keywords']}"
    )


# ---------------------------------------------------------------------------
# matched_keywords list reports the actual matched substring (lowercased)
# ---------------------------------------------------------------------------


def test_matched_keywords_reports_human_readable_substring(make_call):
    """The evidence field must ship the actual keyword (the substring that
    fired), not the regex pattern.
    """
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "Please classify this and rate from 1 to 10",
                    }
                ]
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    matched = ev.evidence["matched_keywords"]
    # Both keywords should be present in their human-readable form.
    assert "classify" in matched
    assert "rate from 1" in matched
    # And NEITHER should be a regex artefact (no \b, no escaped char).
    for kw in matched:
        assert "\\" not in kw
        assert "(" not in kw
        assert "?" not in kw


def test_matched_keywords_deduplicates_repeated_hits(make_call):
    """If the same keyword appears twice in the prompt, evidence reports it once."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "classify this. then classify it again.",
                    }
                ]
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["matched_keywords"].count("classify") == 1


# ---------------------------------------------------------------------------
# Sanity: keyword list shape
# ---------------------------------------------------------------------------


def test_classify_keywords_drops_bare_is_this():
    """The bare ``is this`` substring keyword has been removed in v0.3.2."""
    assert "is this" not in CLASSIFY_KEYWORDS


def test_classify_keywords_includes_is_this_a_and_an():
    """The tightened ``is this a`` / ``is this an`` keywords are present."""
    assert "is this a" in CLASSIFY_KEYWORDS
    assert "is this an" in CLASSIFY_KEYWORDS


def test_classify_keywords_includes_categorise_uk_spelling():
    """Both ``categorize`` and ``categorise`` are in the keyword list."""
    assert "categorize" in CLASSIFY_KEYWORDS
    assert "categorise" in CLASSIFY_KEYWORDS
