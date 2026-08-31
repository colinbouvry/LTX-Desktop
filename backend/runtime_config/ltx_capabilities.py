"""Desktop LTX offering capabilities: feature flags + 16:9 pixel maps.

Duration/fps envelopes stay on LTXVideoGenerationSpec. This matrix is the
feature/pixel SSOT those specs do not have. Local 2.5 is the on-device distilled
offering; its flags are independent of the API Fast rows.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from api_types import (
    LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    LTXLocalModelId,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
)

LtxCapabilityFeature = Literal[
    "t2v",
    "i2v",
    "a2v",
    "ic_lora",
    "retake",
    "extend",
    "multi_keyframe",
    "user_loras",
    "camera_motion",
    "auto_duration",
]
LtxAspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "32:9"]

# w:h numerators used by pixels_for() to rescale a resolution's pixel budget.
# Ratios other than 16:9/9:16 are LOCAL-ONLY: the API path rejects them via
# FORCED_API_ALLOWED_ASPECT_RATIOS in video_generation_handler.py.
_ASPECT_NUMERATORS: dict[LtxAspectRatio, tuple[int, int]] = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
    # Super-ultrawide (dual 16:9). Extreme box: at 1080p budget this is
    # 2752x768, so vertical detail is genuinely low -- framing matters.
    "32:9": (32, 9),
}


@dataclass(frozen=True)
class LtxOfferingCapabilities:
    t2v: bool
    i2v: bool
    a2v: bool
    ic_lora: bool
    retake: bool
    extend: bool
    multi_keyframe: bool
    multi_keyframe_max_count: int
    user_loras: bool
    camera_motion: bool
    # t2v/i2v only: send duration=null and the cloud worker picks length from the prompt.
    auto_duration: bool
    # Label → (width, height) for 16:9; 9:16 is swapped at pixels_for().
    resolution_pixels_16_9: dict[LTXVideoGenResolution, tuple[int, int]]


@dataclass(frozen=True)
class LocalOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so local_caps()'s return
    type can't be confused with api_caps()'s at the type-checker level."""


@dataclass(frozen=True)
class ApiOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so api_caps()'s return
    type can't be confused with local_caps()'s at the type-checker level."""


# Local Fast sizes: one /64 two-stage grid for 2.3 and 2.5. Splitting 540p
# (2.3 960×544 vs 2.5 1024×576) made a model switch fail assert_resolution.
_LOCAL_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "540p": (1024, 576),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
    # Unlocked locally. Heights snap DOWN to the /64 two-stage grid that
    # assert_resolution() enforces (1440 -> 1408, 2160 -> 2112), same reason
    # 1080p is 1088 and not 1080. VRAM-heavy: see notes in the PR/commit.
    "1440p": (2560, 1408),
    "2160p": (3840, 2112),
}

# Non-distilled 2.5. Same feature surface as the distilled model minus IC-LoRA, whose
# adapters are trained against the distilled checkpoint, and minus Retake/Extend, which
# are separate pipelines. Its own gain -- a real CFG scale and a negative prompt -- is not
# a capability flag: the distilled pipeline has no parameter for either.
_LOCAL_2_5_DEV = LocalOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=True,
    multi_keyframe_max_count=LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    user_loras=True,
    camera_motion=True,
    auto_duration=True,
    resolution_pixels_16_9=_LOCAL_PIXELS_16_9,
)

_API_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}

_LOCAL_2_3 = LocalOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=True,
    retake=True,
    extend=True,
    multi_keyframe=True,
    multi_keyframe_max_count=LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    user_loras=True,
    camera_motion=True,
    auto_duration=False,
    resolution_pixels_16_9=_LOCAL_PIXELS_16_9,
)

# DistilledA2V is wired for local 2.5. Auto duration is DurationHead on the
# distilled checkpoint (t2v/i2v; A2V length comes from the audio). Advertised
# only when those weights are on disk — see effective_local_caps().
# Retake/Extend match API 2.5 (unsupported); local 2.3 still offers both.
_LOCAL_2_5 = LocalOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=True,
    retake=False,
    extend=False,
    multi_keyframe=True,
    multi_keyframe_max_count=LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    user_loras=True,
    camera_motion=True,
    auto_duration=True,
    resolution_pixels_16_9=_LOCAL_PIXELS_16_9,
)

# API rows follow ltxv-api handlers. camera_motion is a named LoRA on the tia2v
# stack, not a Desktop capability on Fast.
_API_FAST = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=False,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=False,
    auto_duration=False,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

_API_FAST_2_5 = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=False,
    auto_duration=True,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

_API_PRO = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=True,
    extend=True,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=True,
    auto_duration=False,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

# ltxv-api retake/extend accept ltx-2-pro / ltx-2-3-pro. Auto duration is on both
# 2.5 API variants (t2v/i2v duration=null).
_API_PRO_2_5 = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=True,
    auto_duration=True,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)


def local_caps(model_id: LTXLocalModelId) -> LocalOfferingCapabilities:
    match model_id:
        case "ltx-2.5-22b-distilled":
            return _LOCAL_2_5
        case "ltx-2.5-22b-dev":
            return _LOCAL_2_5_DEV
        case "ltx-2.3-22b-distilled" | "ltx-2.3-22b-distilled-1.1":
            return _LOCAL_2_3
        case _:
            assert_never(model_id)


def effective_local_caps(
    model_id: LTXLocalModelId,
    *,
    duration_head_ready: bool,
) -> LocalOfferingCapabilities:
    """Static offering flags, with Auto duration on only when DurationHead weights are on disk.

    Local-only. API 2.5 Auto duration is independent of this file.
    """
    caps = local_caps(model_id)
    if caps.auto_duration and not duration_head_ready:
        return replace(caps, auto_duration=False)
    return caps


def api_caps(pipeline: LTXVideoGenPipeline) -> ApiOfferingCapabilities:
    match pipeline:
        case "fast":
            return _API_FAST
        case "fast-2.5":
            return _API_FAST_2_5
        case "pro":
            return _API_PRO
        case "pro-2.5":
            return _API_PRO_2_5
        case _:
            assert_never(pipeline)


def supports(caps: LtxOfferingCapabilities, feature: LtxCapabilityFeature) -> bool:
    match feature:
        case "t2v":
            return caps.t2v
        case "i2v":
            return caps.i2v
        case "a2v":
            return caps.a2v
        case "ic_lora":
            return caps.ic_lora
        case "retake":
            return caps.retake
        case "extend":
            return caps.extend
        case "multi_keyframe":
            return caps.multi_keyframe
        case "user_loras":
            return caps.user_loras
        case "camera_motion":
            return caps.camera_motion
        case "auto_duration":
            return caps.auto_duration
        case _:
            assert_never(feature)


_GRID = 64


def _snap_to_grid(value: float) -> int:
    """Round to the nearest multiple of 64, never below one grid cell.

    assert_resolution() in ltx_pipelines rejects anything not divisible by 64 on
    the two-stage pipeline, so this is a hard requirement, not a preference.
    """
    return max(_GRID, round(value / _GRID) * _GRID)


def pixels_for(
    caps: LtxOfferingCapabilities,
    resolution: LTXVideoGenResolution,
    aspect: LtxAspectRatio,
) -> tuple[int, int]:
    size = caps.resolution_pixels_16_9.get(resolution)
    if size is None:
        raise KeyError(resolution)
    width, height = size
    if aspect == "16:9":
        return width, height
    if aspect == "9:16":
        return height, width
    # Other ratios reuse the label's pixel budget so token count -- and therefore
    # VRAM -- stays comparable to 16:9, then snap to the /64 two-stage grid.
    ratio_w, ratio_h = _ASPECT_NUMERATORS[aspect]
    ratio = ratio_w / ratio_h
    # Snap the height first, then derive width FROM the snapped height. Rounding
    # both axes independently lets the errors compound in opposite directions
    # (540p 32:9 landed on 1472x384 = 3.83 against a 3.56 target); anchoring on
    # one axis keeps the delivered ratio within ~2% of the request.
    snapped_height = _snap_to_grid(math.sqrt((width * height) / ratio))
    return _snap_to_grid(snapped_height * ratio), snapped_height
