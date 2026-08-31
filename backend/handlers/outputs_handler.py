"""Listing of the generated-media output folder."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from api_types import OutputGenerationParams, OutputItem, OutputsListResponse
from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Sidecar carrying what produced a file. Written at generation time because the
# parameters are known then and nowhere afterwards: probing the file recovers fps and
# duration but never the prompt, model or mode.
_SIDECAR_SUFFIX = ".gen.json"


def sidecar_path_for(video_path: Path | str) -> Path:
    return Path(str(video_path) + _SIDECAR_SUFFIX)


def write_generation_sidecar(video_path: Path | str, params: OutputGenerationParams) -> None:
    """Record provenance next to a generated file. Never fails the generation."""
    try:
        sidecar_path_for(video_path).write_text(
            params.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError:
        # The render succeeded; losing its provenance must not turn that into a failure.
        logger.warning("Could not write generation sidecar for %s", video_path, exc_info=True)


def read_generation_sidecar(video_path: Path) -> OutputGenerationParams | None:
    sidecar = sidecar_path_for(video_path)
    if not sidecar.is_file():
        return None
    try:
        return OutputGenerationParams.model_validate(json.loads(sidecar.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        # Hand-edited or written by an older version: treat as absent, not as an error.
        logger.warning("Ignoring unreadable generation sidecar %s", sidecar, exc_info=True)
        return None

# Everything the pipelines write. Audio-to-video still produces an .mp4.
_MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv", ".png", ".jpg", ".jpeg"})

# Intermediates the pipelines drop next to the real results (resampled sources,
# control videos). They are inputs to a generation, not something to offer back.
_INTERMEDIATE_PREFIX = "_"

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500


class OutputsHandler:
    """Reads the outputs directory. Owns no state: the filesystem is the truth."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def list_outputs(self, *, limit: int = _DEFAULT_LIMIT, offset: int = 0) -> OutputsListResponse:
        outputs_dir = self._config.outputs_dir
        entries = self._collect(outputs_dir)
        # Newest first: the file someone just generated is the one they want.
        entries.sort(key=lambda item: item.modified_at, reverse=True)

        capped = max(1, min(limit, _MAX_LIMIT))
        start = max(0, offset)
        page = entries[start : start + capped]
        return OutputsListResponse(
            outputs=page,
            total_count=len(entries),
            has_more=start + len(page) < len(entries),
            next_offset=start + len(page) if start + len(page) < len(entries) else None,
            outputs_dir=str(outputs_dir),
        )

    def _collect(self, outputs_dir: Path) -> list[OutputItem]:
        if not outputs_dir.is_dir():
            # A fresh install has generated nothing yet; that is not an error.
            return []
        items: list[OutputItem] = []
        for path in outputs_dir.iterdir():
            if path.name.startswith(_INTERMEDIATE_PREFIX):
                continue
            if path.suffix.lower() not in _MEDIA_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                # Raced with a delete, or unreadable. Skip rather than fail the listing.
                continue
            if not path.is_file():
                continue
            items.append(
                OutputItem(
                    path=str(path),
                    name=path.name,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    generation_params=read_generation_sidecar(path),
                )
            )
        return items
