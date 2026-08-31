"""Frame extraction, so a generated clip can seed the next one.

Local 2.5 has no Extend, so a longer sequence is built by handing a frame forward into
image-to-video. And a multi-shot generation is only useful as a scouting pass if its
individual shots can be recovered as separate stills.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Scene score above which ffmpeg calls it a cut. 0.4 is ffmpeg's common default for
# hard cuts; dissolves score lower and are deliberately not caught, since a dissolve
# has no single representative frame.
DEFAULT_SCENE_THRESHOLD = 0.4

# Floor for probing. A multi-shot render that deliberately holds one decor scores far
# below the default -- a measured three-shot cabin render put its cuts under 0.2 -- so
# finding nothing at 0.4 says little on its own.
MIN_PROBE_THRESHOLD = 0.1
_PROBE_STEPS = (0.3, 0.2, 0.15, MIN_PROBE_THRESHOLD)

_PTS_TIME = re.compile(r"pts_time:([0-9.]+)")


class MediaError(RuntimeError):
    """ffmpeg is missing, or refused the file."""


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MediaError(
            "ffmpeg is not on PATH. Install it (winget install ffmpeg, brew install "
            "ffmpeg, apt install ffmpeg) and restart the MCP server."
        )
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaError(f"Could not run ffmpeg: {exc}") from exc


def _require_video(video_path: str) -> Path:
    path = Path(video_path)
    if not path.is_file():
        raise MediaError(f"No such video: {video_path}")
    return path


def _frames_dir(video: Path, out_dir: str | None) -> Path:
    directory = Path(out_dir) if out_dir else video.parent / f"{video.stem}_frames"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _extract_at(video: Path, seconds: float, destination: Path) -> None:
    # -ss before -i seeks by keyframe and is far faster; accurate enough to pick a
    # representative still, which is all these frames are used for.
    result = _run([
        _ffmpeg(), "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
        "-frames:v", "1", "-y", str(destination),
    ])
    if result.returncode != 0 or not destination.is_file():
        raise MediaError(f"ffmpeg could not extract a frame at {seconds:.3f}s: {result.stderr[:300]}")


def detect_cut_times(video_path: str, threshold: float = DEFAULT_SCENE_THRESHOLD) -> list[float]:
    """Timestamps where ffmpeg scores a scene change above ``threshold``.

    Reads them from the showinfo filter rather than the lavfi movie source, which would
    need the input path escaped inside a filter string -- unreliable on Windows, where
    the drive colon and backslashes both carry meaning to the filter parser.
    """
    video = _require_video(video_path)
    result = _run([
        _ffmpeg(), "-v", "info", "-i", str(video),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ])
    return [float(match) for match in _PTS_TIME.findall(result.stderr)]


def extract_shot_frames(
    video_path: str,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    out_dir: str | None = None,
) -> list[dict[str, object]]:
    """One still per shot: the opening frame, then the first frame after each cut.

    The opening frame is always included -- no scene change precedes it, so detection
    alone would silently drop the first shot.
    """
    video = _require_video(video_path)
    directory = _frames_dir(video, out_dir)

    # Nudge past the cut: the detected timestamp lands on the boundary, where the frame
    # can still carry the outgoing shot.
    starts = [0.0] + [cut + 0.05 for cut in detect_cut_times(video_path, threshold)]

    shots: list[dict[str, object]] = []
    for index, start in enumerate(starts, start=1):
        destination = directory / f"{video.stem}_shot{index:02d}.png"
        _extract_at(video, start, destination)
        shots.append({"shot": index, "time_seconds": round(start, 3), "path": str(destination)})
    return shots


def extract_last_frame(video_path: str, *, out_dir: str | None = None) -> str:
    """The final frame, for handing a clip forward into image-to-video.

    Seeking to the exact end lands past the last frame and yields nothing, and decoding
    in reverse buffers the whole clip (a 40s 1080p render is gigabytes). Instead decode
    the last second with ``-update 1``, where each frame overwrites the output, so what
    remains on disk is the final one.
    """
    video = _require_video(video_path)
    directory = _frames_dir(video, out_dir)
    destination = directory / f"{video.stem}_last.png"

    result = _run([
        _ffmpeg(), "-v", "error", "-sseof", "-1", "-i", str(video),
        "-update", "1", "-y", str(destination),
    ])
    if result.returncode != 0 or not destination.is_file():
        raise MediaError(f"ffmpeg could not extract the last frame: {result.stderr[:300]}")
    return str(destination)


def probe_thresholds(video_path: str, below: float) -> list[tuple[float, int]]:
    """Cut counts at thresholds under ``below``, to tell a quiet cut from no cut at all.

    Returns only thresholds that find something, lowest count first, so the caller can
    pick the least permissive one that still recovers the shots.
    """
    found: list[tuple[float, int]] = []
    for threshold in _PROBE_STEPS:
        if threshold >= below:
            continue
        count = len(detect_cut_times(video_path, threshold))
        if count:
            found.append((threshold, count))
    return found
