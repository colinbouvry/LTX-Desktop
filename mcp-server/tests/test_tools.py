"""Tool behaviour against a real (fake) backend over a real socket."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ltx_desktop_mcp import server
from tests.fake_backend import FakeBackend, FakeState


@pytest.fixture(autouse=True)
def _reset_active_generation():
    """Tools keep the in-flight render in a module global; don't leak it between tests."""
    server._active = None
    yield
    if server._active is not None and not server._active.task.done():
        server._active.task.cancel()
    server._active = None


async def _call(name: str, **arguments: object) -> str:
    """Return what the caller sees: the tool's text, or a raised ToolError's message.

    The SDK turns a ToolError into an error result carrying its message, so both paths
    end up in front of the agent and both are worth asserting on.
    """
    try:
        result = await server.mcp.call_tool(name, arguments)
    except ToolError as exc:
        return str(exc)
    return result.content[0].text


@pytest.fixture
def backend(monkeypatch):
    state = FakeState()
    with FakeBackend(state) as running:
        monkeypatch.setenv("LTX_BACKEND_URL", running.url)
        monkeypatch.delenv("LTX_AUTH_TOKEN", raising=False)
        yield running


@pytest.mark.anyio
async def test_generate_returns_before_the_render_finishes(backend):
    """The POST blocks for the whole render, so the tool must not wait on it."""
    backend.state.generate_delay_seconds = 30.0

    started = asyncio.get_running_loop().time()
    text = await _call("ltx_generate_video", prompt="a buoy on calm water")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 5.0, "ltx_generate_video blocked on the render"
    assert "started" in text.lower()
    assert server._active is not None and not server._active.task.done()


@pytest.mark.anyio
async def test_progress_reports_the_output_path_once_complete(backend):
    backend.state.generate_delay_seconds = 0.0
    await _call("ltx_generate_video", prompt="a buoy", resolution="540p")
    await asyncio.wait_for(server._active.task, timeout=10)

    text = await _call("ltx_generation_progress", response_format="json")
    assert json.loads(text)["video_path"] == backend.state.video_path


@pytest.mark.anyio
async def test_progress_surfaces_a_failed_render(backend):
    """A failed POST is only visible through the task; /progress would still say idle."""
    backend.state.generate_error = (400, "INVALID_LOCAL_RESOLUTION")
    await _call("ltx_generate_video", prompt="a buoy", resolution="2160p")
    with pytest.raises(Exception):
        await asyncio.wait_for(server._active.task, timeout=10)

    text = await _call("ltx_generation_progress")
    assert "failed" in text.lower()
    # The backend sends a bare code; the client must attach something actionable.
    assert "ltx_list_models" in text


@pytest.mark.anyio
async def test_generate_refuses_while_the_desktop_app_is_rendering(backend):
    """One GPU slot: a render started elsewhere must not be clobbered."""
    backend.state.status = "running"
    text = await _call("ltx_generate_video", prompt="a buoy")
    assert "already generating" in text.lower()
    assert server._active is None


@pytest.mark.anyio
async def test_generate_rejects_auto_duration_with_audio(backend):
    text = await _call(
        "ltx_generate_video", prompt="someone speaking", duration_seconds=None, audio_path="a.wav"
    )
    assert "audio" in text.lower()
    assert backend.state.generate_calls == []


@pytest.mark.anyio
async def test_generate_forwards_settings_with_backend_field_names(backend):
    await _call(
        "ltx_generate_video",
        prompt="a buoy",
        resolution="1080p",
        aspect_ratio="21:9",
        duration_seconds=10,
        fps=30,
        seed=42,
    )
    await asyncio.wait_for(server._active.task, timeout=10)

    sent = backend.state.generate_calls[0]
    assert sent["aspectRatio"] == "21:9"
    assert sent["resolution"] == "1080p"
    assert sent["duration"] == 10
    assert sent["fps"] == 30
    assert sent["seed"] == 42
    # Absent optionals must stay absent rather than being sent as null.
    assert "imagePath" not in sent and "audioPath" not in sent


@pytest.mark.anyio
async def test_list_models_collapses_identical_rows(backend):
    """Unlocked envelopes repeat one duration list across every pair; don't restate it."""
    text = await _call("ltx_list_models")
    rows = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(rows) == 1, text
    assert "540p, 720p, 1080p" in rows[0]
    assert "24, 30 fps" in rows[0]


@pytest.mark.anyio
async def test_list_loras_filters_to_downloaded(backend):
    backend.state.loras = [
        {"downloaded": True, "lora": {"name": "Cinematic"}},
        {"downloaded": False, "lora": {"name": "Not here"}},
    ]
    assert "Cinematic" in await _call("ltx_list_loras")
    assert "Not here" not in await _call("ltx_list_loras")
    assert "Not here" in await _call("ltx_list_loras", downloaded_only=False)


@pytest.mark.anyio
async def test_cancel_reports_when_nothing_is_running(backend):
    assert "no generation" in (await _call("ltx_cancel_generation")).lower()


@pytest.mark.anyio
async def test_unreachable_backend_explains_how_to_start_one(monkeypatch):
    # Port 1 is reserved and never listening.
    monkeypatch.setenv("LTX_BACKEND_URL", "http://127.0.0.1:1")
    text = await _call("ltx_backend_status")
    assert "cannot reach" in text.lower()
    assert "ltx2_server.py" in text


@pytest.mark.anyio
async def test_token_mismatch_explains_the_electron_session_token(monkeypatch):
    state = FakeState(require_token="expected")
    with FakeBackend(state) as running:
        monkeypatch.setenv("LTX_BACKEND_URL", running.url)
        monkeypatch.setenv("LTX_AUTH_TOKEN", "wrong")
        text = await _call("ltx_generation_progress")
        assert "401" in text or "LTX_AUTH_TOKEN" in text
