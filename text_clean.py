#!/usr/bin/env python3
"""
text_clean.py  –  Lightweight text normalization for fake news detector.
========================================================================
Default behavior:
- Lowercase conversion
- URL and email removal
- Non-ASCII character filtering
- Whitespace normalization

This module provides preprocessing utilities for ML-based text classification.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

# Precompiled patterns (module-level so they compile once)
URL_RE       = re.compile(r"https?://\S+")
EMAIL_RE     = re.compile(r"\S+@\S+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
EXTRA_SPACE_RE = re.compile(r"\s+")

__all__ = ["clean_text", "clean_many"]


def clean_text(
    text: Optional[str],
    *,
    lowercase        : bool = True,
    remove_urls      : bool = True,
    remove_emails    : bool = True,
    remove_non_ascii : bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """
    Clean a single text string with sensible defaults.

    Parameters
    ----------
    text : str | None
        Input text. Non-string values are treated as empty string.
    lowercase : bool
        Convert to lowercase.
    remove_urls : bool
        Remove URL-like substrings (http/https).
    remove_emails : bool
        Remove email-like substrings.
    remove_non_ascii : bool
        Strip non-ASCII characters.
    collapse_whitespace : bool
        Replace whitespace runs with a single space and strip ends.

    Returns
    -------
    str
        Normalized text (possibly empty string).
    """
    if not isinstance(text, str):
        return ""

    s = text

    if lowercase:
        s = s.lower()
    if remove_urls:
        s = URL_RE.sub(" ", s)
    if remove_emails:
        s = EMAIL_RE.sub(" ", s)
    if remove_non_ascii:
        s = NON_ASCII_RE.sub(" ", s)
    if collapse_whitespace:
        s = EXTRA_SPACE_RE.sub(" ", s).strip()

    return s


def clean_many(
    texts: Sequence[Optional[str]],
    **kwargs,
) -> List[str]:
    """
    Vectorized convenience wrapper to clean a sequence of texts.

    Parameters
    ----------
    texts : sequence of str | None
    **kwargs
        Passed through to clean_text.

    Returns
    -------
    list[str]
    """
    return [clean_text(t, **kwargs) for t in texts]
