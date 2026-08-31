"""GET /api/outputs lists generated media so the app can show renders it did not start."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _write(outputs_dir: Path, name: str, *, mtime: float, content: bytes = b"x") -> Path:
    path = outputs_dir / name
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))
    return path


class TestListOutputs:
    def test_empty_directory_is_not_an_error(self, client):
        response = client.get("/api/outputs")

        assert response.status_code == 200
        body = response.json()
        assert body["outputs"] == []
        assert body["total_count"] == 0
        assert body["has_more"] is False

    def test_lists_media_newest_first(self, client, test_state):
        outputs = test_state.config.outputs_dir
        _write(outputs, "older.mp4", mtime=1_000)
        _write(outputs, "newest.mp4", mtime=3_000)
        _write(outputs, "middle.mp4", mtime=2_000)

        body = client.get("/api/outputs").json()

        assert [item["name"] for item in body["outputs"]] == ["newest.mp4", "middle.mp4", "older.mp4"]
        assert body["total_count"] == 3
        assert body["outputs_dir"] == str(outputs)

    def test_reports_size_and_path(self, client, test_state):
        outputs = test_state.config.outputs_dir
        path = _write(outputs, "clip.mp4", mtime=1_000, content=b"0123456789")

        item = client.get("/api/outputs").json()["outputs"][0]

        assert item["path"] == str(path)
        assert item["size_bytes"] == 10
        assert item["modified_at"] == 1_000

    def test_skips_pipeline_intermediates(self, client, test_state):
        """Resampled sources and control videos are inputs, not results to offer back."""
        outputs = test_state.config.outputs_dir
        _write(outputs, "_resampled_30fps_abc.mp4", mtime=2_000)
        _write(outputs, "_control_depth_abc.mp4", mtime=2_000)
        _write(outputs, "ltx2_video_20260831_a1b2c3d4.mp4", mtime=1_000)

        names = [item["name"] for item in client.get("/api/outputs").json()["outputs"]]

        assert names == ["ltx2_video_20260831_a1b2c3d4.mp4"]

    def test_skips_non_media_files(self, client, test_state):
        outputs = test_state.config.outputs_dir
        _write(outputs, "notes.txt", mtime=1_000)
        _write(outputs, "clip.mp4", mtime=1_000)
        (outputs / "a_directory.mp4").mkdir()

        names = [item["name"] for item in client.get("/api/outputs").json()["outputs"]]

        assert names == ["clip.mp4"]

    def test_paginates(self, client, test_state):
        outputs = test_state.config.outputs_dir
        for index in range(5):
            _write(outputs, f"clip{index}.mp4", mtime=1_000 + index)

        first = client.get("/api/outputs", params={"limit": 2}).json()
        assert [item["name"] for item in first["outputs"]] == ["clip4.mp4", "clip3.mp4"]
        assert first["has_more"] is True
        assert first["next_offset"] == 2

        last = client.get("/api/outputs", params={"limit": 2, "offset": 4}).json()
        assert [item["name"] for item in last["outputs"]] == ["clip0.mp4"]
        assert last["has_more"] is False
        assert last["next_offset"] is None

    def test_rejects_out_of_range_limit(self, client):
        assert client.get("/api/outputs", params={"limit": 0}).status_code == 422
        assert client.get("/api/outputs", params={"limit": 501}).status_code == 422
        assert client.get("/api/outputs", params={"offset": -1}).status_code == 422


class TestGenerationProvenance:
    """A render started outside the app must carry the same provenance as one from the UI."""

    def test_sidecar_is_written_next_to_the_video(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()

        response = client.post(
            "/api/generate",
            json={
                "prompt": "a red buoy on calm water",
                "resolution": "540p",
                "aspectRatio": "32:9",
                "model": "fast",
                "duration": 2,
                "fps": 30,
                "seed": 7,
            },
        )

        assert response.status_code == 200
        video_path = Path(response.json()["video_path"])
        sidecar = Path(str(video_path) + ".gen.json")
        assert sidecar.is_file()

        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert recorded["mode"] == "text-to-video"
        assert recorded["prompt"] == "a red buoy on calm water"
        assert recorded["aspect_ratio"] == "32:9"
        assert recorded["resolution"] == "540p"
        assert recorded["fps"] == 30
        assert recorded["duration"] == 2
        assert recorded["seed"] == 7

    def test_listing_returns_recorded_params(self, client, test_state):
        outputs = test_state.config.outputs_dir
        _write(outputs, "clip.mp4", mtime=1_000)
        (outputs / "clip.mp4.gen.json").write_text(
            json.dumps(
                {
                    "mode": "audio-to-video",
                    "prompt": "someone speaking",
                    "model": "fast",
                    "duration": 5,
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "fps": 24,
                    "audio": True,
                    "camera_motion": "none",
                }
            ),
            encoding="utf-8",
        )

        item = client.get("/api/outputs").json()["outputs"][0]

        assert item["name"] == "clip.mp4"
        assert item["generation_params"]["mode"] == "audio-to-video"
        assert item["generation_params"]["prompt"] == "someone speaking"

    def test_sidecars_are_not_listed_as_outputs(self, client, test_state):
        outputs = test_state.config.outputs_dir
        _write(outputs, "clip.mp4", mtime=1_000)
        (outputs / "clip.mp4.gen.json").write_text("{}", encoding="utf-8")

        names = [item["name"] for item in client.get("/api/outputs").json()["outputs"]]

        assert names == ["clip.mp4"]

    def test_file_without_a_sidecar_reports_no_params(self, client, test_state):
        _write(test_state.config.outputs_dir, "hand_dropped.mp4", mtime=1_000)

        assert client.get("/api/outputs").json()["outputs"][0]["generation_params"] is None

    def test_unreadable_sidecar_is_ignored_rather_than_failing_the_listing(self, client, test_state):
        outputs = test_state.config.outputs_dir
        _write(outputs, "clip.mp4", mtime=1_000)
        (outputs / "clip.mp4.gen.json").write_text("not json at all", encoding="utf-8")

        body = client.get("/api/outputs").json()

        assert body["outputs"][0]["generation_params"] is None

    def test_ic_lora_renders_are_recorded_too(self, client, tmp_path, create_fake_model_files, fake_services):
        """CrossView and other IC-LoRA runs write to the same folder as plain generations.

        Without a sidecar they look like hand-dropped files, so the app never picks them
        up -- which is exactly how multicam angles went missing.
        """
        create_fake_model_files()
        assert client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"}).status_code == 200
        image = tmp_path / "reference.png"
        image.write_bytes(b"\x89PNG\r\n")

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(image),
                "control_values": {"duration": 5},
                "prompt": "crossview. new camera angle: to the right, lower, closer.",
                "conditioning_type": "custom",
            },
        )

        assert response.status_code == 200
        sidecar = Path(response.json()["video_path"] + ".gen.json")
        assert sidecar.is_file()

        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert recorded["mode"] == "ic-lora"
        assert recorded["model"] == "ingredients-v1"
        assert recorded["prompt"].startswith("crossview.")
