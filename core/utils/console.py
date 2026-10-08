# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared rich console + helpers for pretty terminal output across the pipeline."""

from __future__ import annotations

import os

from rich.console import Console

# Single shared console so all pipeline output uses one width/style detection.
CONSOLE = Console()


def fmt_path(path: str) -> str:
    """Shorten a path for display: cwd-relative when possible, else normalized.

    Collapses noise like ``core/../results`` and trims the long absolute
    prefix when the path lives under the working directory.
    """
    p = os.path.normpath(str(path))
    try:
        rel = os.path.relpath(p)
    except ValueError:  # different drive (Windows) — keep absolute
        return p
    return rel if not rel.startswith("..") else p


def block_header(title: str, style: str = "bold cyan") -> None:
    """Print a pipeline block header as a horizontal rule with a centered title."""
    CONSOLE.rule(f"[{style}]{title}[/{style}]")
