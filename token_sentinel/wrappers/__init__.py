"""Client wrappers — instrument official LLM SDKs in-place.

Anthropic and OpenAI wrappers are required (their deps come with the
``anthropic`` / ``openai`` extras). Other providers are optional — their
wrapper modules import their respective SDK at module import time, so they
are only loaded if the SDK is installed.
"""

from token_sentinel.wrappers.anthropic import wrap_anthropic
from token_sentinel.wrappers.openai import wrap_openai

__all__ = ["wrap_anthropic", "wrap_openai"]

# Optional providers — exported only if their SDK is installed.
try:
    from token_sentinel.wrappers.gemini import wrap_gemini  # noqa: F401

    __all__.append("wrap_gemini")
except ImportError:
    pass

try:
    from token_sentinel.wrappers.bedrock import wrap_bedrock  # noqa: F401

    __all__.append("wrap_bedrock")
except ImportError:
    pass

try:
    from token_sentinel.wrappers.voyage import wrap_voyage  # noqa: F401

    __all__.append("wrap_voyage")
except ImportError:
    pass

# Cohere wrapper duck-types the ``ClientV2`` / ``AsyncClientV2`` surface
# (no module-level ``import cohere``), so the import always succeeds even
# when the optional ``cohere`` SDK isn't installed. The customer is
# responsible for importing their own ``cohere.ClientV2`` instance.
from token_sentinel.wrappers.cohere import wrap_cohere  # noqa: E402, F401

__all__.append("wrap_cohere")

# Replicate wrapper has no module-level SDK import (it duck-types the
# Client surface), so the import always succeeds even when the optional
# ``replicate`` SDK isn't installed. Importing the user's actual client
# is the customer's responsibility.
from token_sentinel.wrappers.replicate import wrap_replicate  # noqa: E402, F401

__all__.append("wrap_replicate")

# Deepgram wrapper also duck-types the client surface (no module-level
# import of ``deepgram``), so we import it unconditionally for the same
# reason as replicate.
from token_sentinel.wrappers.deepgram import wrap_deepgram  # noqa: E402, F401

__all__.append("wrap_deepgram")
