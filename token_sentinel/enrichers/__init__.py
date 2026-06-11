"""Enrichers — bridges from third-party callback / observability frameworks
into the TokenSentinel session buffer.

A wrapper instruments a *client* (``sentinel.wrap(anthropic.Anthropic())``);
an enricher instruments a *framework* by hooking its event bus. Both end up
producing :class:`token_sentinel.events.CallRecord` instances and routing
them through :meth:`Sentinel.record_call`, so the rule engine sees a
uniform view regardless of which surface the customer used to make their
LLM calls.

Each enricher module is gated on its dependency being installed. The
import below uses a defensive shim: if ``langchain_core`` is missing, the
module imports successfully but ``TokenSentinelCallbackHandler`` raises
:class:`ImportError` with installation hint at construction time. This
mirrors the wrapper pattern (cold-import cost stays minimal for customers
who don't use the integration).
"""

from token_sentinel.enrichers.langchain import TokenSentinelCallbackHandler
from token_sentinel.enrichers.otel import TokenSentinelSpanProcessor

__all__ = ["TokenSentinelCallbackHandler", "TokenSentinelSpanProcessor"]
