"""接口出入参模型（Pydantic v2）。

集中定义数据模型的好处：Swagger 文档自动生成完整字段说明，
非法参数会在进入业务逻辑之前被拦截并返回 422。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OutputFormat = Literal["srt", "txt", "both"]


class Segment(BaseModel):
    """一条带时间戳的转写片段。"""

    start: float = Field(description="片段开始时间（秒）")
    end: float = Field(description="片段结束时间（秒）")
    text: str = Field(description="片段文本")


class TranscribeFileResponse(BaseModel):
    filename: str
    language: str
    segments: list[Segment] = Field(description="带时间戳的分段结果")
    srt: str | None = Field(default=None, description="SRT 字幕内容")
    txt: str | None = Field(default=None, description="纯文本内容")


class BatchRequest(BaseModel):
    """批量转写请求。"""

    directory: str = Field(description="待转写的音视频所在目录（服务端路径）")
    languages: list[str] = Field(default_factory=lambda: ["en"], description="识别语言列表，可一次跑多种语言")
    target_lang: str | None = Field(default=None, description="设置后用百度翻译生成目标语言字幕")
    model_choice: int = Field(default=4, ge=1, le=7, description="模型编号 1~7")
    gen_srt: bool = True
    gen_txt: bool = True
    overwrite: bool = Field(default=False, description="为 False 时跳过已存在的输出文件")


class BatchFileResult(BaseModel):
    file: str
    status: Literal["ok", "skipped", "failed"]
    outputs: list[str] = Field(default_factory=list)
    error: str | None = None


class BatchResponse(BaseModel):
    total: int
    succeeded: int
    skipped: int
    failed: int
    results: list[BatchFileResult]
    log: list[str]


class PreprocessRequest(BaseModel):
    """音频预处理参数。``None`` 表示沿用预设的默认值。"""

    preset: Literal[
        "mobile", "desktop_high_quality", "noisy_environment", "meeting_room", "lightweight"
    ] = "meeting_room"
    enable_vad: bool | None = None
    enable_nr: bool | None = None
    enable_se: bool | None = None
    nr_strength: float | None = Field(default=None, ge=0.0, le=1.0, description="降噪强度 0~1")


class PreprocessResponse(BaseModel):
    filename: str
    preset_used: str
    sample_rate: int
    original_duration_s: float
    processed_duration_s: float
    processed_audio_base64: str = Field(description="处理后音频（base64 编码的 WAV）")
    stats: dict


class TranscribeComparison(BaseModel):
    baseline_text: str
    enhanced_text: str
    note: str


class PreprocessCompareResponse(BaseModel):
    """预处理 + 转写对比结果：同一段音频在预处理前后的转写差异。"""

    filename: str
    preset_used: str
    language: str
    stats: dict
    baseline: dict = Field(description="原始音频转写结果（txt/srt/segments）")
    enhanced: dict = Field(description="预处理后音频转写结果")
    comparison: TranscribeComparison


class ModelInfo(BaseModel):
    loaded: bool
    model_choice: int
    model_name: str
    device: str
    compute_type: str
    model_dir: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: str
    translation_enabled: bool
    presets: list[str]


class ModelLoadResponse(BaseModel):
    message: str
    model_name: str
    device: str
    compute_type: str
