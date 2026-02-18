"""Shared request validators for the API."""

from __future__ import annotations

from typing import Annotated

from fastapi import Path

# Session ID: alphanumeric, underscore, hyphen only (matches manager._generate_session_id).
# Prevents path traversal and injection in JSON store and path-based routes.
SessionIdPath = Annotated[
    str,
    Path(
        description="Session ID (alphanumeric, underscore, hyphen only)",
        pattern=r"^[a-zA-Z0-9_-]+$",
        min_length=1,
        max_length=256,
    ),
]
