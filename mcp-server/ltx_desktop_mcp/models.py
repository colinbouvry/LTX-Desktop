"""Shared parameter types. Values mirror the backend Literals in backend/api_types.py."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

AspectRatio = Literal["16:9", "9:16", "21:9", "32:9", "4:3", "3:4", "1:1"]
Resolution = Literal["540p", "720p", "1080p", "1440p", "2160p"]
Fps = Literal[24, 25, 30, 48, 50, 60]
ResponseFormat = Literal["markdown", "json"]

Prompt = Annotated[
    str,
    Field(
        description="What to generate, e.g. 'a lighthouse keeper climbing stairs at dawn'.",
        min_length=1,
        max_length=2000,
    ),
]

DurationSeconds = Annotated[
    int | None,
    Field(
        description=(
            "Clip length in seconds. Null asks the model to pick one (text- and "
            "image-to-video only, and only when the DurationHead weights are present). "
            "Audio-to-video takes its length from the audio track instead."
        ),
        ge=1,
        le=40,
    ),
]

Seed = Annotated[int | None, Field(description="Fixed seed for a reproducible run.", ge=0)]

NegativePrompt = Annotated[
    str | None,
    Field(description="What to avoid. Omit to use the backend default.", max_length=2000),
]

ImagePath = Annotated[
    str | None,
    Field(description="Absolute path to a first-frame image. Makes this image-to-video."),
]

AudioPath = Annotated[
    str | None,
    Field(
        description=(
            "Absolute path to a .wav/.flac/.ogg/.mp3/.aac/.m4a track. Makes this "
            "audio-to-video, which synchronises motion (including lip movement for "
            "speech) to the audio. Validated by file header, not by extension."
        )
    ),
]
