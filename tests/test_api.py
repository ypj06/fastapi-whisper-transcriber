"""接口级冒烟测试：用 FakeEngine 替换真实推理，验证路由、渲染与错误处理。"""

from __future__ import annotations

import base64
import io
from pathlib import Path


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "fastapi-whisper-transcriber"

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert len(payload["presets"]) == 5


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_name"] == "base"
    assert info["device"] == "cpu"


def test_list_presets(client):
    presets = client.get("/preprocess/presets").json()
    assert "meeting_room" in presets
    assert presets["lightweight"]["vad"] == "off"


def test_transcribe_file_returns_srt_and_txt(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/transcribe/file",
        files={"file": (name, content, mime)},
        params={"language": "en", "output_format": "both"},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["filename"] == name
    assert data["txt"] == "hello world\nthis is a test"
    assert data["srt"].startswith("1\n00:00:00,000 --> 00:00:01,500\nhello world")
    assert len(data["segments"]) == 3


def test_transcribe_file_only_txt(client, wav_upload):
    name, content, mime = wav_upload
    data = client.post(
        "/transcribe/file",
        files={"file": (name, content, mime)},
        params={"output_format": "txt"},
    ).json()
    assert data["srt"] is None
    assert data["txt"]


def test_transcribe_file_without_translation_keeps_original(client, wav_upload):
    """未配置百度密钥时，target_lang 不应破坏原文输出。"""
    name, content, mime = wav_upload
    data = client.post(
        "/transcribe/file",
        files={"file": (name, content, mime)},
        params={"output_format": "srt", "target_lang": "zh"},
    ).json()
    assert "hello world" in data["srt"]


def test_transcribe_file_rejects_unsupported_type(client):
    resp = client.post(
        "/transcribe/file",
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 415


def test_transcribe_file_validates_output_format(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/transcribe/file",
        files={"file": (name, content, mime)},
        params={"output_format": "docx"},
    )
    assert resp.status_code == 422


def test_batch_writes_srt_and_txt(client, sample_wav: Path, tmp_path: Path):
    import shutil

    target_dir = tmp_path / "batch"
    target_dir.mkdir()
    shutil.copy(sample_wav, target_dir / "clip.wav")

    resp = client.post(
        "/transcribe/batch",
        json={"directory": str(target_dir), "languages": ["en"], "model_choice": 2},
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["succeeded"] == 1
    assert (target_dir / "clip.srt").exists()
    assert (target_dir / "clip_en.txt").exists()


def test_batch_reports_missing_directory(client):
    resp = client.post(
        "/transcribe/batch",
        json={"directory": "definitely/not/here", "languages": ["en"]},
    )
    assert resp.status_code == 404


def test_preprocess_audio_returns_base64_wav(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/preprocess/audio",
        files={"file": (name, content, mime)},
        params={"preset": "lightweight"},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    audio_bytes = base64.b64decode(data["processed_audio_base64"])
    assert audio_bytes[:4] == b"RIFF"
    assert data["stats"]["input_duration_s"] > 0


def test_preprocess_audio_raw_sets_headers(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/preprocess/audio/raw",
        files={"file": (name, content, mime)},
        params={"preset": "lightweight"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["X-Preset"] == "lightweight"


def test_preprocess_unknown_preset_is_400(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/preprocess/audio",
        files={"file": (name, content, mime)},
        params={"preset": "studio"},
    )
    assert resp.status_code == 400


def test_preprocess_and_transcribe_returns_comparison(client, wav_upload):
    name, content, mime = wav_upload
    resp = client.post(
        "/preprocess-and-transcribe",
        files={"file": (name, content, mime)},
        params={"preset": "lightweight", "language": "en", "output_format": "both"},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["comparison"]["baseline_text"]
    assert data["comparison"]["enhanced_text"]
    assert "segments" in data["baseline"]
