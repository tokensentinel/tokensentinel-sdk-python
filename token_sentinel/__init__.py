"""TokenSentinel — predictive token-waste detection for AI agents.

Public framing is "token waste" (cost) — not "token leak" (security). The
internal Python API still uses ``LeakEvent`` / ``LeakDetected`` / ``on_leak``
for backward compatibility with installed customer code; the stable release
adds ``WasteEvent`` / ``WasteDetected`` / ``on_waste`` as transparent aliases
(same objects, not subclasses). New code may use either set of names.
"""

from token_sentinel.events import (
    BudgetExceeded,
    CallRecord,
    KillSwitchActive,
    LeakDetected,
    LeakEvent,
    VelocityExceeded,
    WasteDetected,
    WasteEvent,
)
from token_sentinel.sentinel import Sentinel

__version__ = "1.0.0"
__all__ = [
    "Sentinel",
    "LeakEvent",
    "WasteEvent",
    "CallRecord",
    "LeakDetected",
    "WasteDetected",
    "BudgetExceeded",
    "VelocityExceeded",
    "KillSwitchActive",
]
