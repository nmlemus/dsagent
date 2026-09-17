"""The model a persona runs on when it does not name one.

Here rather than in `host/build.py` so that anything needing the name — pricing,
the runner's per-step cost — can have it without importing Deep Agents.
"""

from __future__ import annotations

import os

FALLBACK_MODEL = "anthropic:claude-sonnet-5"
"""Chosen in the ROADMAP decision of 2026-09-16: newer and cheaper than Sonnet 4.6,
and it runs adaptive thinking unconfigured. A stronger model is opted into per
persona, never made the default — the default carries every step of every run."""


def default_model() -> str:
    """`DSAGENT_MODEL`, or the fallback. Read per call, so a test can set it."""
    return os.environ.get("DSAGENT_MODEL", FALLBACK_MODEL)
