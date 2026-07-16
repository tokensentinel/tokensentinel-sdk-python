"""Hatch metadata hook: absolute-ize relative README links for PyPI only.

GitHub keeps the on-disk README.md with relative links (branch-aware, good DX).
PyPI has no repo base URL, so the published long_description rewrites those
links against a single GitHub root before upload.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

from hatchling.metadata.plugin.interface import MetadataHookInterface

# Single fallback base (blob for files; tree used when the path ends with /).
_REPO = "https://github.com/tokensentinel/tokensentinel-sdk-python"
_BLOB = f"{_REPO}/blob/main/"
_TREE = f"{_REPO}/tree/main/"

# Markdown inline links / images: [text](target) or ![alt](target)
_LINK_RE = re.compile(r"(!?\[[^\]]*\])\(([^)]+)\)")


def _is_absolute_or_special(target: str) -> bool:
    t = target.strip()
    return (
        not t
        or t.startswith("#")
        or t.startswith("mailto:")
        or t.startswith("http://")
        or t.startswith("https://")
        or t.startswith("data:")
        or "://" in t.split("/", 1)[0]  # scheme://
    )


def absolutize_readme_links(text: str, blob_base: str = _BLOB, tree_base: str = _TREE) -> str:
    """Rewrite relative markdown link targets to absolute GitHub URLs."""

    def repl(match: re.Match[str]) -> str:
        prefix, target = match.group(1), match.group(2).strip()
        if _is_absolute_or_special(target):
            return match.group(0)

        # Optional title: (path "title") — keep title if present
        path_part, title = target, ""
        if target[:1] in "\"'" or " \"" in target or " '" in target:
            # rare; leave untouched rather than mis-parse
            parts = target.split(None, 1)
            path_part = parts[0]
            title = f" {parts[1]}" if len(parts) > 1 else ""

        path_part = path_part.lstrip("./")
        path_only, hash_sep, frag = path_part.partition("#")
        if not path_only:
            return match.group(0)

        base = tree_base if path_only.endswith("/") else blob_base
        abs_url = urljoin(base, path_only)
        if hash_sep:
            abs_url = f"{abs_url}#{frag}"
        return f"{prefix}({abs_url}{title})"

    return _LINK_RE.sub(repl, text)


class CustomMetadataHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        readme_path = Path(self.root) / "README.md"
        raw = readme_path.read_text(encoding="utf-8")
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolutize_readme_links(raw),
        }
