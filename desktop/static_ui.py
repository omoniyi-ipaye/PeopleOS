"""Static UI middleware for packaged PeopleOS local builds."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response


_EXCLUDED_PREFIXES = ("/api", "/docs", "/redoc", "/openapi.json")


def _resolve_ui_file(ui_dir: Path, request_path: str) -> Path | None:
    relative = request_path.lstrip("/")
    candidates: list[Path] = []

    if not relative:
        candidates.append(ui_dir / "index.html")
    else:
        direct = ui_dir / relative
        candidates.extend([
            direct,
            direct / "index.html",
            ui_dir / f"{relative}.html",
        ])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(ui_dir.resolve())
        except (ValueError, OSError):
            continue
        if resolved.is_file():
            return resolved
    return None


def install_static_ui(app: FastAPI, ui_dir: Path) -> None:
    """Install a same-origin static UI layer around the existing API app.

    API and documentation routes always pass through untouched. Everything else
    is resolved from the exported Next.js assets so browser fetch('/api/...')
    remains same-origin and no CORS/proxy configuration is required.
    """
    root = ui_dir.resolve()

    @app.middleware("http")
    async def desktop_static_ui(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        path = request.url.path
        if path.startswith(_EXCLUDED_PREFIXES):
            return await call_next(request)

        asset = _resolve_ui_file(root, path)
        if asset is not None:
            media_type, _ = mimetypes.guess_type(str(asset))
            return FileResponse(asset, media_type=media_type)

        # Preserve normal FastAPI behavior for explicit non-UI routes first.
        response = await call_next(request)
        if response.status_code != 404:
            return response

        # Client-side navigation fallback for future statically exported pages.
        index = root / "index.html"
        return FileResponse(index, media_type="text/html") if index.exists() else response
