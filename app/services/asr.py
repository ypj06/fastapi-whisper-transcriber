"""Whisper 模型管理与转写核心。

相对原实现的三点改动：
1. ``faster-whisper`` / ``torch`` 改为函数内延迟导入——没装推理依赖时
   项目其它部分（如字幕生成、路由）仍可被导入和测试；
2. 转写结果统一返回 ``list[dict]``，上层再派生 SRT / TXT，避免同一段
   音频被重复转写三次；
3. 模型来源按"项目内置目录 → 环境变量指定路径 → 在线下载"顺序查找，
   去掉了写死的个人机器绝对路径。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from app.config import model_name, settings

logger = logging.getLogger(__name__)


def _detect_device() -> tuple[str, str]:
    """决定推理设备与计算精度，可用环境变量强制覆盖。"""
    device = settings.device
    compute_type = settings.compute_type

    if device is None:
        try:
            import torch  # 延迟导入：CPU-only 环境不装 torch 也能跑
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class WhisperEngine:
    """进程内单例的 Whisper 推理封装。"""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._choice: int = settings.default_model_choice
        self.device, self.compute_type = _detect_device()

    # ------------------------------------------------------------------ 状态
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def choice(self) -> int:
        return self._choice

    @property
    def name(self) -> str:
        return model_name(self._choice)

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_choice": self._choice,
            "model_name": self.name,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_dir": str(settings.model_dir),
        }

    # ------------------------------------------------------------------ 加载
    def load(self, choice: int | None = None) -> dict[str, Any]:
        """加载/切换模型。重复调用同一编号时直接复用已加载的模型。"""
        choice = choice or settings.default_model_choice
        if self._model is not None and choice == self._choice:
            return self.info()

        from faster_whisper import WhisperModel  # 延迟导入

        name = model_name(choice)
        model_path = self._resolve_path(name)
        logger.info("加载模型 %s（device=%s, compute_type=%s）", name, self.device, self.compute_type)

        self._model = WhisperModel(
            str(model_path) if model_path else name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=None if model_path else str(settings.model_dir),
        )
        self._choice = choice
        return self.info()

    def _resolve_path(self, name: str) -> Path | None:
        settings.model_dir.mkdir(parents=True, exist_ok=True)

        bundled = settings.model_dir / f"faster-whisper-{name}"
        if bundled.exists() and any(bundled.iterdir()):
            logger.info("使用项目内置模型：%s", bundled)
            return bundled

        local = settings.local_model_path
        if local and local.is_dir():
            logger.info("使用 WHISPER_LOCAL_PATH 指定的模型：%s", local)
            return local

        logger.info("未找到本地模型，将下载到 %s（可用 HF_ENDPOINT 设置镜像）", settings.model_dir)
        return None

    # ------------------------------------------------------------------ 转写
    def transcribe(
        self,
        media_path: str | os.PathLike[str],
        language: str = "en",
        vad_filter: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """转写音视频文件，返回 ``(segments, meta)``。

        ``segments`` 中每项为 ``{"start", "end", "text"}``；``meta`` 含
        识别出的语言与置信度，便于前端展示。
        """
        if self._model is None:
            raise RuntimeError("模型尚未加载，请先调用 /model/load 或等待服务启动完成")

        segments_iter, info = self._model.transcribe(
            str(media_path),
            language=None if language in ("auto", "") else language,
            task="transcribe",
            vad_filter=vad_filter,
        )
        segments = [
            {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": seg.text.strip()}
            for seg in segments_iter
        ]
        meta = {
            "language": getattr(info, "language", language),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            "duration": round(float(getattr(info, "duration", 0.0)), 2),
        }
        return segments, meta
