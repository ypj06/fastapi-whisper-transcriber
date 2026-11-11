"""pytest 公共夹具。"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """生成 1 秒 440 Hz 正弦波 WAV，用于预处理与上传相关测试。"""
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), audio, sr)
    return path


class FakeEngine:
    """替代真实 Whisper 推理，返回固定分段，便于验证路由与渲染逻辑。"""

    def __init__(self, segments: list[dict[str, Any]] | None = None) -> None:
        self.device = "cpu"
        self.compute_type = "int8"
        self._segments = segments or [
            {"start": 0.0, "end": 1.5, "text": " hello world "},
            {"start": 1.5, "end": 3.0, "text": "this is a test"},
            {"start": 3.0, "end": 3.5, "text": "   "},
        ]

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def choice(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "base"

    def info(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "model_choice": 2,
            "model_name": "base",
            "device": self.device,
            "compute_type": self.compute_type,
            "model_dir": "models",
        }

    def transcribe(self, media_path, language: str = "en", vad_filter: bool = True):
        return list(self._segments), {"language": language, "language_probability": 0.99, "duration": 3.5}


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(fake_engine: FakeEngine):
    """构造已替换推理后端的 TestClient。"""
    from fastapi.testclient import TestClient

    from app.deps import get_engine
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_engine] = lambda: fake_engine
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def wav_upload(sample_wav: Path) -> tuple[str, io.BytesIO, str]:
    """构造 multipart 上传用的 (文件名, 内容, MIME)。"""
    return (sample_wav.name, io.BytesIO(sample_wav.read_bytes()), "audio/wav")
