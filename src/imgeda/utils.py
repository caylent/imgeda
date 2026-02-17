"""Shared utilities."""

from __future__ import annotations

import html


def fmt_bytes(b: int | float) -> str:
    """Format byte count to human-readable string."""
    if b > 1_000_000_000:
        return f"{b / 1_000_000_000:.2f} GB"
    if b > 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    return f"{b / 1_000:.2f} KB"


def escape_html(s: str) -> str:
    """HTML-escape a string for safe embedding in reports."""
    return html.escape(s)
