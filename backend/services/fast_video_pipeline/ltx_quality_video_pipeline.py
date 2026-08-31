"""Non-distilled two-stage pipeline, exposed under the same protocol as the fast one.

Stage 1 runs the base transformer with a real CFG scale and a negative prompt; stage 2
upsamples and refines with a distilled adapter. Neither guidance nor a negative prompt
exists on the distilled path -- ``DistilledPipeline.__call__`` has no parameter for
either -- so this is what those controls come from, not a quality setting.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from api_types import ImageConditioningInput
from frame_math import AutoDurationSpec
from services.ltx_pipeline_common import (
    auto_tiling_config,
    build_model_paths,
    encode_video_output,
    offload_mode_for_prefetch_count,
    video_chunks_number,
)
from services.services_utils import device_supports_fp8

from services.services_utils import AudioOrNone, PipelineTilingType, TilingConfigType

if TYPE_CHECKING:
    from collections.abc import Iterator


class LTXQualityVideoPipeline:
    """Wraps ``TI2VidTwoStagesPipeline`` for the non-distilled checkpoint."""

    pipeline_kind: ClassVar[Literal["fast"]] = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
        stage_2_lora_path: str | None = None,
    ) -> "LTXQualityVideoPipeline":
        return LTXQualityVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            loras=loras or [],
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
            stage_2_lora_path=stage_2_lora_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
        stage_2_lora_path: str | None = None,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import detect_params

        if stage_2_lora_path is None:
            raise ValueError(
                "The non-distilled pipeline needs its stage-2 refinement LoRA. Without it "
                "stage 2 upsamples with no adapter and the result is soft."
            )

        self._checkpoint_path = checkpoint_path
        self._device = device
        # Sampler settings come from the checkpoint's own model_version rather than a
        # constant: the library ships presets per generation and picking one by hand
        # silently mismatches the schedule.
        self._params = detect_params(checkpoint_path)

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in (loras or [])
        ]

        self.pipeline = TI2VidTwoStagesPipeline(
            model_paths=build_model_paths(
                checkpoint_path,
                gemma_root,
                video_vae_path=video_vae_path,
                audio_vae_path=audio_vae_path,
                duration_head_path=duration_head_path,
            ),
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    path=stage_2_lora_path,
                    strength=1.0,
                    # The renaming map is not optional: without it the adapter's keys match
                    # nothing in the transformer and stage 2 runs unrefined, with no error.
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
            spatial_upsampler_path=upsampler_path,
            loras=lora_entries,
            device=device,
            quantization=build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None,
            offload_mode=offload_mode_for_prefetch_count(streaming_prefetch_count, device),
        )

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: PipelineTilingType,
    ) -> tuple["torch.Tensor | Iterator[torch.Tensor]", AudioOrNone, int, TilingConfigType | None]:
        # The app carries its own conditioning and duration types; the pipeline wants the
        # library's. Same conversion the distilled wrapper performs.
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.types import AutoDuration

        pipeline_num_frames: int | AutoDuration = (
            AutoDuration(min_seconds=num_frames.min_seconds, max_seconds=num_frames.max_seconds)
            if isinstance(num_frames, AutoDurationSpec)
            else num_frames
        )

        video, audio, resolved_frames, resolved_tiling = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            num_inference_steps=self._params.num_inference_steps,
            video_guider_params=self._params.video_guider_params,
            audio_guider_params=self._params.audio_guider_params,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            num_frames=pipeline_num_frames,
            tiling_config=tiling_config,
        )
        return video, audio, resolved_frames, resolved_tiling

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        *,
        guide_all_images: bool = False,
        negative_prompt: str | None = None,
    ) -> None:
        video, audio, resolved_frames, resolved_tiling = self._run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=auto_tiling_config(),
        )
        chunks = video_chunks_number(resolved_frames, resolved_tiling)
        encode_video_output(
            video=video,
            audio=audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        try:
            video, audio, resolved_frames, resolved_tiling = self._run_inference(
                prompt="test warmup",
                negative_prompt="",
                seed=42,
                height=256,
                width=384,
                num_frames=9,
                frame_rate=8,
                images=[],
                tiling_config=auto_tiling_config(),
            )
            chunks = video_chunks_number(resolved_frames, resolved_tiling)
            encode_video_output(
                video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        # torch.compile is not wired for this path yet. The distilled pipeline compiles a
        # single resident transformer; this one streams two stages, so the same treatment
        # does not transfer unchanged.
        return
