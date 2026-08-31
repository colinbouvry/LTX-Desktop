"""HTTP client for the LTX Desktop backend, with actionable error mapping."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:41954"

# Generation blocks for minutes; everything else is a fast local call. Splitting the
# timeouts keeps a hung backend from looking like a slow render.
_FAST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_GENERATE_TIMEOUT = httpx.Timeout(3600.0, connect=5.0)

# Backend error codes carry no prose, so map the ones an agent can actually act on.
_ERROR_HINTS: dict[str, str] = {
    "NO_DOWNLOADED_LTX_MODEL": (
        "No local LTX checkpoint is downloaded. Open LTX Desktop and download a model, "
        "or point LTX_APP_DATA_DIR at a data directory that already has one."
    ),
    "INVALID_LOCAL_RESOLUTION": (
        "This resolution/aspect-ratio pair is not offered by the active local model. "
        "Call ltx_list_models to see the exact envelope."
    ),
    "INVALID_LOCAL_A2V_RESOLUTION": (
        "This resolution/aspect-ratio pair is not offered for audio-to-video. "
        "Call ltx_list_models and read a2v_supported_resolutions_durations."
    ),
    "INVALID_FORCED_API_ASPECT_RATIO": (
        "The backend is in API mode, which only accepts 16:9 and 9:16. Ratios such as "
        "21:9 or 32:9 are local-generation only."
    ),
    "GENERATION_ALREADY_RUNNING": (
        "The backend holds a single GPU slot. Wait for the current generation or call "
        "ltx_cancel_generation."
    ),
}


class BackendError(RuntimeError):
    """A backend call failed in a way worth reporting verbatim to the agent."""


def base_url() -> str:
    return os.environ.get("LTX_BACKEND_URL", DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    # Electron generates a random per-session token. A standalone backend started with
    # no LTX_AUTH_TOKEN disables the middleware entirely, which is the headless setup.
    token = os.environ.get("LTX_AUTH_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _describe(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:400]}"
    code = payload.get("code") or ""
    message = payload.get("message") or response.text[:400]
    hint = _ERROR_HINTS.get(message) or _ERROR_HINTS.get(code)
    detail = f"HTTP {response.status_code} {code}: {message}".strip()
    return f"{detail}\n\nHint: {hint}" if hint else detail


async def request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    long_running: bool = False,
) -> Any:
    """Call the backend and return parsed JSON, or raise BackendError with guidance."""
    timeout = _GENERATE_TIMEOUT if long_running else _FAST_TIMEOUT
    url = f"{base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_headers()) as client:
            response = await client.request(method, url, json=json_body)
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Cannot reach the LTX Desktop backend at {base_url()}.\n\n"
            "Start it with LTX Desktop running, or headless:\n"
            "  cd backend && LTX_APP_DATA_DIR=<data dir> uv run python ltx2_server.py\n"
            "Set LTX_BACKEND_URL if it listens elsewhere."
        ) from exc
    except httpx.ReadTimeout as exc:
        raise BackendError(f"The backend did not answer {method} {path} in time.") from exc

    if response.status_code == 401:
        raise BackendError(
            "The backend rejected the request (401). It was started with an auth token "
            "(LTX Desktop does this per session). Set LTX_AUTH_TOKEN to the same value, "
            "or run a standalone backend without one."
        )
    if response.status_code >= 400:
        raise BackendError(_describe(response))
    return response.json()
