"""MCP server exposing LTX Desktop's local video generation.

The backend holds a single GPU slot and ``POST /api/generate`` blocks for the whole
render (minutes). Tools therefore start a generation in the background and return at
once; ``ltx_generation_progress`` reports on it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import client, media
from .models import (
    AspectRatio,
    AudioPath,
    DurationSeconds,
    Fps,
    ImagePath,
    NegativePrompt,
    Prompt,
    Resolution,
    ResponseFormat,
    Seed,
)

mcp = MCPServer("ltx_desktop_mcp")


@dataclass
class _ActiveGeneration:
    """The in-flight render this server started, if any."""

    task: asyncio.Task[Any]
    summary: dict[str, Any] = field(default_factory=dict)


_active: _ActiveGeneration | None = None


def _render(payload: dict[str, Any], response_format: ResponseFormat, title: str) -> str:
    if response_format == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    lines = [f"## {title}"]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def _envelope_lines(table: dict[str, Any]) -> list[str]:
    """Render a resolution/fps/duration table, collapsing rows that repeat.

    Every (resolution, fps) pair usually carries the same duration list, so one line
    each restates it dozens of times for no added information. Group by what actually
    differs and print only that.
    """
    by_shape: dict[tuple[str, str], list[str]] = {}
    for resolution, detail in table.items():
        for fps, durations in (detail.get("fps_to_durations") or {}).items():
            key = (str(fps), ", ".join(str(d) for d in durations))
            by_shape.setdefault(key, []).append(str(resolution))

    merged: dict[tuple[str, str], list[str]] = {}
    for (fps, durations), resolutions in by_shape.items():
        merged.setdefault((durations, ", ".join(resolutions)), []).append(fps)

    return [
        f"- {resolutions} @ {', '.join(fps_list)} fps -> {durations}s"
        for (durations, resolutions), fps_list in merged.items()
    ]


async def _backend_is_busy() -> bool:
    """True when any generation is running, including one started by the desktop app."""
    progress = await client.request("GET", "/api/generation/progress")
    return bool(progress.get("status") == "running")


@mcp.tool(
    name="ltx_generate_video",
    title="Start an LTX video generation",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
)
async def ltx_generate_video(
    prompt: Prompt,
    resolution: Resolution = "720p",
    aspect_ratio: AspectRatio = "16:9",
    duration_seconds: DurationSeconds = 5,
    fps: Fps = 24,
    seed: Seed = None,
    negative_prompt: NegativePrompt = None,
    image_path: ImagePath = None,
    audio_path: AudioPath = None,
) -> str:
    """Start a local video generation and return immediately without waiting for it.

    Renders take minutes (a 1080p/10s clip measured ~5 minutes on an RTX 5090), so this
    tool only starts the work. Poll ltx_generation_progress for status and the output
    path. Supplying image_path makes it image-to-video; audio_path makes it
    audio-to-video, which synchronises motion (including lips) to the track.

    Only one generation can run at a time: the backend owns a single GPU slot, shared
    with the LTX Desktop app if it is open.

    Args:
        prompt: What to generate.
        resolution: Short-edge tier; cost grows with pixels x frames.
        aspect_ratio: Ratios beyond 16:9 and 9:16 are local-generation only, and
            delivered dimensions snap to the /64 grid the pipeline requires, so the
            real ratio can differ by up to ~2.5%.
        duration_seconds: Clip length, or null for model-chosen duration.
        fps: 24 and 48 land exactly on the VAE temporal grid; other rates round down
            slightly unless fps x duration is divisible by 8.
        seed: Fixed seed for reproducibility.
        negative_prompt: What to avoid.
        image_path: First-frame image, for image-to-video.
        audio_path: Audio track, for audio-to-video.

    Returns:
        Confirmation that the render started, with the settings in effect.
    """
    global _active

    if duration_seconds is None and audio_path:
        return (
            "Audio-to-video takes its length from the audio track. Give a "
            "duration_seconds, or drop audio_path."
        )
    if _active is not None and not _active.task.done():
        return (
            "A generation started by this server is still running. Poll "
            "ltx_generation_progress, or call ltx_cancel_generation to stop it."
        )
    if await _backend_is_busy():
        return (
            "The backend is already generating - most likely from the LTX Desktop "
            "window. It has one GPU slot; wait for it or cancel it there."
        )

    body: dict[str, Any] = {
        "prompt": prompt,
        "resolution": resolution,
        "aspectRatio": aspect_ratio,
        "duration": duration_seconds,
        "fps": fps,
        "model": "fast",
    }
    for key, value in (
        ("seed", seed),
        ("negativePrompt", negative_prompt),
        ("imagePath", image_path),
        ("audioPath", audio_path),
    ):
        if value is not None:
            body[key] = value

    mode = (
        "audio-to-video" if audio_path
        else "image-to-video" if image_path
        else "text-to-video"
    )
    summary = {
        "mode": mode,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "duration_seconds": duration_seconds,
        "fps": fps,
    }

    # Fire and keep the handle: the POST only returns once the render is finished.
    task = asyncio.create_task(
        client.request("POST", "/api/generate", json_body=body, long_running=True)
    )
    _active = _ActiveGeneration(task=task, summary=summary)

    return _render(
        {**summary, "started": True, "next_step": "Poll ltx_generation_progress."},
        "markdown",
        "Generation started",
    )


@mcp.tool(
    name="ltx_generation_progress",
    title="Check generation progress",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_generation_progress(response_format: ResponseFormat = "markdown") -> str:
    """Report on the current generation, including the output path once it finishes.

    Covers renders started by this server and by the LTX Desktop window. When a render
    this server started has failed, the backend error is surfaced here rather than lost.

    Args:
        response_format: "markdown" (default) or "json".

    Returns:
        Status, phase, percentage, step counter, and the video path when complete.
    """
    global _active

    # A failed background POST is only visible through its task, not through /progress.
    if _active is not None and _active.task.done():
        try:
            result = _active.task.result()
        except client.BackendError as exc:
            _active = None
            return f"The generation failed.\n\n{exc}"
        except asyncio.CancelledError:
            _active = None
            return "The generation was cancelled."
        _active = None
        if isinstance(result, dict) and result.get("video_path"):
            return _render(
                {"status": "complete", "video_path": result["video_path"]},
                response_format,
                "Generation complete",
            )

    progress = await client.request("GET", "/api/generation/progress")
    payload = {
        "status": progress.get("status"),
        "phase": progress.get("phase"),
        "progress_percent": progress.get("progress"),
        "step": progress.get("currentStep"),
        "total_steps": progress.get("totalSteps"),
        "cancellable": progress.get("cancellable"),
        "result": progress.get("result"),
    }
    return _render(payload, response_format, "Generation progress")


@mcp.tool(
    name="ltx_cancel_generation",
    title="Cancel the running generation",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_cancel_generation() -> str:
    """Cancel the generation in progress, discarding its work.

    Returns:
        Whether a generation was cancelled or none was running.
    """
    global _active
    result = await client.request("POST", "/api/generate/cancel")
    if _active is not None and not _active.task.done():
        _active.task.cancel()
    _active = None
    status = result.get("status", "unknown")
    if status == "no_active_generation":
        return "No generation was running."
    return f"Cancellation requested (status: {status})."


@mcp.tool(
    name="ltx_list_models",
    title="List models and their supported settings",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_list_models(response_format: ResponseFormat = "markdown") -> str:
    """List available models and the resolution/fps/duration combinations they accept.

    Call this before generating with unusual settings: the envelope is the authority on
    what the active local checkpoint allows, and rejected combinations return an error
    rather than being clamped.

    Args:
        response_format: "markdown" (default) or "json".

    Returns:
        Local and API model entries with their supported_resolutions_durations tables.
    """
    specs = await client.request("GET", "/api/generate/models-specs")
    if response_format == "json":
        return json.dumps(specs, indent=2, ensure_ascii=False)

    lines: list[str] = []
    for group in ("local_models", "api_models"):
        entries = specs.get(group) or []
        lines.append(f"## {group.replace('_', ' ')} ({len(entries)})")
        for item in entries:
            spec = item.get("spec", {})
            lines.append(f"\n### {spec.get('display_name', item.get('pipeline'))}")
            lines.extend(_envelope_lines(spec.get("supported_resolutions_durations") or {}))
    return "\n".join(lines) or "No models reported."


@mcp.tool(
    name="ltx_list_loras",
    title="List LoRAs",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_list_loras(
    downloaded_only: bool = True, response_format: ResponseFormat = "markdown"
) -> str:
    """List LoRA adapters known to the backend.

    LoRAs apply to local generation only; the API path ignores them.

    Args:
        downloaded_only: Keep only LoRAs whose weights are on disk (default true).
        response_format: "markdown" (default) or "json".

    Returns:
        The matching LoRAs with their identifiers and download state.
    """
    data = await client.request("GET", "/api/loras")
    items = data.get("loras") or []
    if downloaded_only:
        items = [entry for entry in items if entry.get("downloaded")]
    if response_format == "json":
        return json.dumps(items, indent=2, ensure_ascii=False)
    if not items:
        return "No LoRAs found." + (" Try downloaded_only=false." if downloaded_only else "")
    lines = [f"## LoRAs ({len(items)})"]
    for entry in items:
        lora = entry.get("lora") or {}
        name = lora.get("name") or lora.get("id") or "?"
        lines.append(f"- **{name}** (downloaded: {entry.get('downloaded')})")
    return "\n".join(lines)


@mcp.tool(
    name="ltx_backend_status",
    title="Report backend and GPU status",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_backend_status(response_format: ResponseFormat = "markdown") -> str:
    """Report whether the backend is reachable, and its GPU and generation mode.

    Use this first when anything fails: it distinguishes an unreachable backend from an
    API-only runtime policy, which silently rules out local generation and LoRAs.

    Args:
        response_format: "markdown" (default) or "json".

    Returns:
        Runtime policy and GPU details.
    """
    policy = await client.request("GET", "/api/runtime-policy")
    try:
        gpu = await client.request("GET", "/api/gpu-info")
    except client.BackendError:
        gpu = {}
    payload = {"backend_url": client.base_url(), "runtime_policy": policy, "gpu": gpu}
    return _render(payload, response_format, "Backend status")


@mcp.tool(
    name="ltx_extract_shots",
    title="Split a multi-shot video into one still per shot",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_extract_shots(
    video_path: str,
    scene_threshold: float = media.DEFAULT_SCENE_THRESHOLD,
    out_dir: str | None = None,
) -> str:
    """Recover each shot of a multi-shot generation as a separate still image.

    A multi-shot prompt yields one file containing several angles of the same place.
    This turns it into a usable scouting pass: one frame per shot, each ready to seed a
    longer continuous take via ltx_generate_video's image_path.

    Only hard cuts are found. Dissolves score below the threshold and are skipped on
    purpose -- a dissolve has no single representative frame.

    Args:
        video_path: Absolute path to the generated video.
        scene_threshold: Scene-change score counted as a cut (0-1). Lower finds more
            cuts and more false positives; raise it if gentle camera moves are split.
        out_dir: Where to write the stills. Defaults to a folder beside the video.

    Returns:
        One line per shot with its index, timestamp and image path.
    """
    try:
        shots = await asyncio.to_thread(
            media.extract_shot_frames, video_path, threshold=scene_threshold, out_dir=out_dir
        )
    except media.MediaError as exc:
        raise client.BackendError(str(exc)) from exc

    if len(shots) == 1:
        # A multi-shot render that holds one decor across its cuts scores low, because
        # consecutive shots genuinely look alike. Probing says whether the cuts are
        # there at all, instead of leaving a dead end at the default threshold.
        probed = await asyncio.to_thread(media.probe_thresholds, video_path, scene_threshold)
        if probed:
            found = ", ".join(f"{t} -> {n} cut(s)" for t, n in probed)
            return (
                f"No cut above {scene_threshold}, but lower thresholds find some: {found}.\n\n"
                "Shots that share a decor score low by design. Re-run with the lowest "
                "threshold that still gives the number of shots you wrote."
            )
        return (
            f"One shot only (no cut found down to {media.MIN_PROBE_THRESHOLD}). Opening "
            f"frame: {shots[0]['path']}\n\n"
            "The render is a single continuous take."
        )
    lines = [f"{len(shots)} shots:"]
    lines.extend(f"- shot {s['shot']} at {s['time_seconds']}s -> {s['path']}" for s in shots)
    return "\n".join(lines)


@mcp.tool(
    name="ltx_extract_last_frame",
    title="Extract a clip's final frame",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def ltx_extract_last_frame(video_path: str, out_dir: str | None = None) -> str:
    """Take the final frame of a clip, to continue the shot in a new generation.

    Local 2.5 has no Extend, so a longer sequence is built by feeding this frame back in
    as ltx_generate_video's image_path. Expect drift: colour and sharpness degrade with
    each hand-off, usually visibly by the third or fourth link.

    Args:
        video_path: Absolute path to the clip to continue.
        out_dir: Where to write the still. Defaults to a folder beside the video.

    Returns:
        The path of the extracted frame.
    """
    try:
        path = await asyncio.to_thread(media.extract_last_frame, video_path, out_dir=out_dir)
    except media.MediaError as exc:
        raise client.BackendError(str(exc)) from exc
    return f"Final frame: {path}\n\nPass it as image_path to continue the shot."


def main() -> None:
    """Entry point: serve over stdio, the transport local MCP clients expect."""
    mcp.run()
