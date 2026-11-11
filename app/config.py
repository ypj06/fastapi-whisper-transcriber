"""全局配置：模型映射、媒体类型、路径与运行参数。

所有可变配置均通过环境变量注入，避免把个人机器上的绝对路径写进代码。
参见项目根目录的 ``.env.example``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # 读取项目根目录下的 .env（不存在时忽略）
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv 未安装时退化为纯环境变量
    pass

# 模型编号 → faster-whisper 模型名（保持与前端下拉框一致）
MODEL_MAP: dict[int, str] = {
    1: "tiny",
    2: "base",
    3: "small",
    4: "medium",
    5: "large-v1",
    6: "large-v2",
    7: "large-v3",
}

DEFAULT_MODEL_CHOICE = 2

# 单文件上传与批量扫描支持的音视频扩展名
MEDIA_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov", ".wmv",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
)

# 百度翻译语言代码（仅在配置了密钥时启用）
LANG_MAP: dict[str, str] = {
    "zh": "zh",
    "en": "en",
    "ja": "jp",
    "ko": "kor",
    "fr": "fra",
    "de": "de",
}

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "zh", "ja", "ko", "fr", "de")

# 单次上传大小上限，默认 500 MB
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。"""

    model_dir: Path = field(default_factory=lambda: _env_path("WHISPER_MODEL_DIR", PROJECT_ROOT / "models"))
    local_model_path: Path | None = field(
        default_factory=lambda: _env_path("WHISPER_LOCAL_PATH", Path("/nonexistent"))
        if os.getenv("WHISPER_LOCAL_PATH")
        else None
    )
    device: str | None = os.getenv("WHISPER_DEVICE") or None
    compute_type: str | None = os.getenv("WHISPER_COMPUTE_TYPE") or None
    default_model_choice: int = int(os.getenv("WHISPER_MODEL_CHOICE", str(DEFAULT_MODEL_CHOICE)))

    baidu_app_id: str = os.getenv("BAIDU_APP_ID", "")
    baidu_app_key: str = os.getenv("BAIDU_APP_KEY", "")

    batch_overwrite: bool = os.getenv("BATCH_OVERWRITE", "false").lower() in {"1", "true", "yes"}
    max_upload_mb: int = MAX_UPLOAD_MB

    @property
    def translation_enabled(self) -> bool:
        return bool(self.baidu_app_id and self.baidu_app_key)


settings = Settings()

# 让 tokenizers 不并行，避免和 uvicorn 的 worker 抢线程
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")


def model_name(choice: int | None) -> str:
    """把前端传来的模型编号转换成模型名，非法编号回落到默认模型。"""
    return MODEL_MAP.get(choice or settings.default_model_choice, "base")
