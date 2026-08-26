"""Minimal design-partner snippet: api_key + cloud_endpoint only.

Catch KillSwitchActive / OrgBudgetExceeded / PreflightBlocked so an
operator can halt spend from Slack or the dashboard without redeploying.
"""

from __future__ import annotations

import os

from token_sentinel import (
    KillSwitchActive,
    OrgBudgetExceeded,
    PreflightBlocked,
    Sentinel,
)

sentinel = Sentinel(
    project=os.environ.get("TOKENSENTINEL_PROJECT", "acme-agent"),
    cloud_endpoint=os.environ.get("TOKENSENTINEL_CLOUD", "http://127.0.0.1:8000"),
    api_key=os.environ["TOKENSENTINEL_API_KEY"],
)

# Use a stable session id so session budget + pre-flight flags accumulate.
session = sentinel.session(session_id="demo-session", tags={"team": "design-partner"})


def guarded_call(client, **kwargs):
    try:
        sentinel.preflight(session.session_id)
        return client.messages.create(_sentinel_session_id=session.session_id, **kwargs)
    except KillSwitchActive:
        print("halted: operator kill-switch")
        raise
    except OrgBudgetExceeded as exc:
        print(f"halted: org budget ${exc.budget_usd}")
        raise
    except PreflightBlocked as exc:
        print(f"halted: historical pattern {exc.pattern}")
        raise
