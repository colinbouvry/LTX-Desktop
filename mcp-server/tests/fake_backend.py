"""A real HTTP server standing in for the LTX Desktop backend.

Served over a real socket rather than patched in, so the tools exercise the same
httpx client, headers, status handling and timeouts they use in production.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


@dataclass
class FakeState:
    """Whatever the test wants the backend to report."""

    status: str = "idle"
    generate_delay_seconds: float = 0.0
    generate_error: tuple[int, str] | None = None
    video_path: str = r"C:\outputs\clip.mp4"
    require_token: str | None = None
    loras: list[dict[str, Any]] = field(default_factory=list)
    generate_calls: list[dict[str, Any]] = field(default_factory=list)


def build_app(state: FakeState) -> Starlette:
    def _unauthorized(request: Request) -> JSONResponse | None:
        if state.require_token is None:
            return None
        if request.headers.get("authorization") == f"Bearer {state.require_token}":
            return None
        return JSONResponse({"code": "HTTP_401", "message": "Unauthorized"}, status_code=401)

    async def generate(request: Request) -> JSONResponse:
        denied = _unauthorized(request)
        if denied is not None:
            return denied
        state.generate_calls.append(await request.json())
        if state.generate_error is not None:
            code, message = state.generate_error
            return JSONResponse({"code": f"HTTP_{code}", "message": message}, status_code=code)
        state.status = "running"
        await asyncio.sleep(state.generate_delay_seconds)
        state.status = "complete"
        return JSONResponse({"status": "complete", "video_path": state.video_path})

    async def progress(request: Request) -> JSONResponse:
        denied = _unauthorized(request)
        if denied is not None:
            return denied
        return JSONResponse(
            {
                "status": state.status,
                "phase": "inference" if state.status == "running" else "",
                "progress": 50 if state.status == "running" else 0,
                "currentStep": 4,
                "totalSteps": 8,
                "cancellable": state.status == "running",
                "result": state.video_path if state.status == "complete" else None,
                "id": "fake",
            }
        )

    async def cancel(request: Request) -> JSONResponse:
        if state.status != "running":
            return JSONResponse({"status": "no_active_generation"})
        state.status = "cancelled"
        return JSONResponse({"status": "cancelling"})

    async def models_specs(request: Request) -> JSONResponse:
        durations = [2, 5, 10]
        return JSONResponse(
            {
                "local_models": [
                    {
                        "pipeline": "fast",
                        "spec": {
                            "display_name": "LTX 2.5 Fast",
                            "supported_resolutions_durations": {
                                resolution: {"fps_to_durations": {"24": durations, "30": durations}}
                                for resolution in ("540p", "720p", "1080p")
                            },
                        },
                    }
                ],
                "api_models": [],
            }
        )

    async def loras(request: Request) -> JSONResponse:
        return JSONResponse({"loras": state.loras})

    async def runtime_policy(request: Request) -> JSONResponse:
        return JSONResponse({"force_api_generations": False})

    async def gpu_info(request: Request) -> JSONResponse:
        return JSONResponse({"gpu_name": "Fake GPU", "vram_gb": 31})

    return Starlette(
        routes=[
            Route("/api/generate", generate, methods=["POST"]),
            Route("/api/generate/cancel", cancel, methods=["POST"]),
            Route("/api/generation/progress", progress),
            Route("/api/generate/models-specs", models_specs),
            Route("/api/loras", loras),
            Route("/api/runtime-policy", runtime_policy),
            Route("/api/gpu-info", gpu_info),
        ]
    )


class FakeBackend:
    """Runs the fake app on a real loopback port for the duration of a test."""

    def __init__(self, state: FakeState) -> None:
        self.state = state
        self._server = uvicorn.Server(
            uvicorn.Config(build_app(state), host="127.0.0.1", port=0, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "FakeBackend":
        self._thread.start()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("fake backend failed to start")
            threading.Event().wait(0.02)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def url(self) -> str:
        port = self._server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"
