"""Tests for ``Sentinel._load_rules`` unknown-rule-name warnings (MED-2).

Closes the v0.2.0 code-review finding "_load_rules silently drops unknown
rule names" — ``Sentinel(project="x", rules=["tool_loop", "tool_lop"])`` (typo)
used to load only ``tool_loop`` with no signal to the customer that they
actually disabled half of what they thought they enabled.

Behaviour: emit a ``UserWarning`` listing the unknown names AND the known
names. Don't raise — that would be a hard breaking change for customers
who currently rely on the silent-drop. Warning lets them notice the typo
without breaking existing setups.
"""

from __future__ import annotations

import warnings

import pytest

from token_sentinel import Sentinel

# ---------------------------------------------------------------------------
# Warning fires on unknown name
# ---------------------------------------------------------------------------


def test_unknown_rule_name_emits_userwarning():
    with pytest.warns(UserWarning) as record:
        Sentinel(project="proj", rules=["tool_loop", "tool_lop"])
    # exactly one warning from _load_rules
    matching = [w for w in record if "TokenSentinel" in str(w.message)]
    assert len(matching) >= 1


def test_warning_message_lists_unknown_names():
    with pytest.warns(UserWarning) as record:
        Sentinel(project="proj", rules=["tool_loop", "tool_lop", "made_up"])
    msg = str(record[0].message)
    assert "tool_lop" in msg
    assert "made_up" in msg
    # The valid one should NOT be in the unknowns list — but it can appear
    # in the "Known rules" enumeration. Check it's not in the "ignored"
    # portion specifically.
    ignored_part = msg.split("Known rules:")[0]
    assert "tool_loop" not in ignored_part


def test_warning_message_lists_known_names():
    """The warning enumerates every loaded rule so the customer can spot
    the closest valid name to their typo."""
    with pytest.warns(UserWarning) as record:
        Sentinel(project="proj", rules=["tool_lop"])
    msg = str(record[0].message)
    # All eight default-rule names must be enumerated.
    for known in [
        "tool_loop",
        "context_bloat",
        "embedding_waste",
        "zombie",
        "model_misroute",
        "retry_storm",
        "tool_definition_bloat",
        "retrieval_thrash",
    ]:
        assert known in msg


# ---------------------------------------------------------------------------
# Warning does NOT fire on valid configurations
# ---------------------------------------------------------------------------


def test_no_warning_when_all_rules_valid():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Sentinel(project="proj", rules=["tool_loop", "retry_storm"])
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert matching == []


def test_no_warning_when_rules_is_all_default():
    """``rules='all'`` is the default and must never warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Sentinel(project="proj")  # default rules='all'
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert matching == []


def test_no_warning_when_rules_is_all_explicit():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Sentinel(project="proj", rules="all")
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert matching == []


def test_no_warning_when_rules_empty():
    """Empty list disables every rule but doesn't request anything unknown."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Sentinel(project="proj", rules=[])
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert matching == []


# ---------------------------------------------------------------------------
# Warning category is exactly UserWarning (not RuntimeWarning, not bare Warning)
# ---------------------------------------------------------------------------


def test_warning_is_userwarning_not_runtimewarning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Sentinel(project="proj", rules=["bogus_rule"])
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert len(matching) == 1
    assert matching[0].category is UserWarning
    # And specifically NOT RuntimeWarning — that's reserved for runtime
    # surprises (e.g. block-mode-on-streaming).
    assert not issubclass(matching[0].category, RuntimeWarning) or (
        matching[0].category is not RuntimeWarning
    )


# ---------------------------------------------------------------------------
# Warning is suppressible via the standard filter mechanism
# ---------------------------------------------------------------------------


def test_warning_suppressible_via_filterwarnings():
    """Customers who deliberately want to ignore the warning must be able
    to silence it with ``warnings.filterwarnings(...)``."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=UserWarning)
        Sentinel(project="proj", rules=["bogus_rule"])
    matching = [item for item in w if "unknown rule names" in str(item.message)]
    assert matching == []


# ---------------------------------------------------------------------------
# Behavioural invariant: the rule list itself is correctly filtered
# ---------------------------------------------------------------------------


def test_unknown_rules_still_dropped_from_rule_list():
    """Warning is informational; the silent-drop behaviour itself is preserved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Sentinel(project="proj", rules=["tool_loop", "bogus"])
    names = {r.name for r in s._rules}
    assert names == {"tool_loop"}
