"""Base class for rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent


class Rule(ABC):
    """A leak detection rule. Pure function of session buffer + config.

    Implementations must:
    - Be deterministic (same input → same output)
    - Run in <10ms p95 (total budget across all rules: 50ms)
    - Never perform I/O
    - Catch their own exceptions and degrade gracefully
    """

    name: str = "rule"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def get(self, key: str, default: Any) -> Any:
        return self.config.get(f"{self.name}.{key}", default)

    @abstractmethod
    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        """Return a LeakEvent if the rule fires, else None."""
