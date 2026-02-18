"""File management endpoints."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dsagent.config import get_settings
from dsagent.server.deps import (
    get_session_manager,
    verify_api_key,
)
from dsagent.server.models import ErrorResponse
from dsagent.server.validators import SessionIdPath
from dsagent.session import SessionManager

router = APIRouter(dependencies=[Depends(verify_api_key)])


# =============================================================================
# Models
# =============================================================================


class FileInfo(BaseModel):
    """Information about a file."""

    name: str = Field(..., description="File name")
    size: int = Field(..., description="File size in bytes")
    modified: datetime = Field(..., description="Last modified time")
    category: str = Field(..., description="File category (data, artifacts, notebooks)")


class FileListResponse(BaseModel):
    """Response for listing files."""

    files: List[FileInfo] = Field(default_factory=list)
    total: int = Field(0, description="Total number of files")
    category: str = Field(..., description="File category")


class FileUploadResponse(BaseModel):
    """Response for file upload."""

    filename: str = Field(..., description="Uploaded file name")
    size: int = Field(..., description="File size in bytes")
    path: str = Field(..., description="Relative path to file")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_session_path(
    session_manager: SessionManager,
    session_id: str,
    category: str = "data"
) -> Path:
    """Get the path to a session's file category directory.

    Args:
        session_manager: Session manager instance
        session_id: Session ID
        category: File category (data, artifacts, notebooks)

    Returns:
        Path to the category directory

    Raises:
        HTTPException: If session not found or invalid category
    """
    session = session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get the appropriate path based on category
    if category == "data":
        path = session.data_path
    elif category == "artifacts":
        path = session.artifacts_path
    elif category == "notebooks":
        path = session.notebooks_path
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {category}. Must be data, artifacts, or notebooks",
        )

    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} has no {category} path configured",
        )

    # Ensure path exists
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def _get_file_info(file_path: Path, category: str) -> FileInfo:
    """Get file information."""
    stat = file_path.stat()
    return FileInfo(
        name=file_path.name,
        size=stat.st_size,
        modified=datetime.fromtimestamp(stat.st_mtime),
        category=category,
    )


def _resolve_file_path(base_path: Path, filename: str) -> Path:
    """Resolve file path and ensure it stays under base_path (path traversal guard).

    Args:
        base_path: Base directory (e.g. session data/artifacts path).
        filename: Requested filename (may contain path components; only basename is used).

    Returns:
        Resolved Path under base_path.

    Raises:
        HTTPException: 400 if the resolved path would be outside base_path.
    """
    safe_name = Path(filename).name
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )
    file_path = (base_path / safe_name).resolve()
    base_resolved = base_path.resolve()
    try:
        if not file_path.is_relative_to(base_resolved):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path",
            )
    except (ValueError, AttributeError):
        base_str = str(base_resolved) + os.sep
        if file_path != base_resolved and not str(file_path).startswith(base_str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path",
            )
    return file_path


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/sessions/{session_id}/files",
    response_model=List[FileUploadResponse],
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
async def upload_files(
    session_id: SessionIdPath,
    files: List[UploadFile] = File(..., description="Files to upload"),
    category: str = Query("data", description="File category (data, artifacts)"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> List[FileUploadResponse]:
    """Upload one or more files to a session.

    Files are stored in the session's workspace directory under the
    specified category (data/, artifacts/, or notebooks/).

    Args:
        session_id: Session ID
        files: Files to upload
        category: Category to store files in (default: data)
        session_manager: Session manager instance

    Returns:
        List of upload results

    Raises:
        HTTPException: If session not found or upload fails
    """
    base_path = _get_session_path(session_manager, session_id, category)
    settings = get_settings()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024) if settings.max_upload_mb > 0 else 0

    results = []
    for upload_file in files:
        if not upload_file.filename:
            continue

        # Sanitize filename (remove path components)
        filename = Path(upload_file.filename).name

        # Enforce upload size limit
        content = b""
        if max_bytes > 0:
            chunk_size = 65536
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                if len(content) > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File {filename} exceeds max size ({settings.max_upload_mb} MB)",
                    )
        else:
            content = await upload_file.read()

        # Save file
        file_path = base_path / filename
        try:
            with open(file_path, "wb") as f:
                f.write(content)

            results.append(FileUploadResponse(
                filename=filename,
                size=len(content),
                path=f"{category}/{filename}",
            ))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file {filename}: {str(e)}",
            )

    return results


@router.get(
    "/sessions/{session_id}/files",
    response_model=FileListResponse,
    responses={404: {"model": ErrorResponse}},
)
async def list_files(
    session_id: SessionIdPath,
    category: str = Query("data", description="File category (data, artifacts, notebooks)"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileListResponse:
    """List files in a session.

    Args:
        session_id: Session ID
        category: Category to list files from
        session_manager: Session manager instance

    Returns:
        List of files

    Raises:
        HTTPException: If session not found
    """
    base_path = _get_session_path(session_manager, session_id, category)

    files = []
    if base_path.exists():
        for file_path in base_path.iterdir():
            if file_path.is_file():
                files.append(_get_file_info(file_path, category))

    # Sort by modified time (newest first)
    files.sort(key=lambda f: f.modified, reverse=True)

    return FileListResponse(
        files=files,
        total=len(files),
        category=category,
    )


@router.get(
    "/sessions/{session_id}/files/{filename}",
    responses={404: {"model": ErrorResponse}},
)
async def download_file(
    session_id: SessionIdPath,
    filename: str,
    category: str = Query("data", description="File category (data, artifacts, notebooks)"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    """Download a file from a session.

    Args:
        session_id: Session ID
        filename: Name of file to download
        category: Category to download from
        session_manager: Session manager instance

    Returns:
        File content

    Raises:
        HTTPException: If session or file not found
    """
    base_path = _get_session_path(session_manager, session_id, category)
    file_path = _resolve_file_path(base_path, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {Path(filename).name} not found in {category}",
        )

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".parquet": "application/octet-stream",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".ipynb": "application/x-ipynb+json",
        ".py": "text/x-python",
        ".txt": "text/plain",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=media_type,
    )


@router.delete(
    "/sessions/{session_id}/files/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_file(
    session_id: SessionIdPath,
    filename: str,
    category: str = Query("data", description="File category (data, artifacts, notebooks)"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Delete a file from a session.

    Args:
        session_id: Session ID
        filename: Name of file to delete
        category: Category to delete from
        session_manager: Session manager instance

    Raises:
        HTTPException: If session or file not found
    """
    base_path = _get_session_path(session_manager, session_id, category)
    file_path = _resolve_file_path(base_path, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {Path(filename).name} not found in {category}",
        )

    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )
