"""音频预处理路由：降噪 / VAD / 增强，以及"预处理 + 转写对比"一体化接口。"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.deps import get_engine, get_translator
from app.schemas import (
    OutputFormat,
    PreprocessCompareResponse,
    PreprocessResponse,
    TranscribeComparison,
)
from app.services.asr import WhisperEngine
from app.services.media import (
    FileTooLargeError,
    UnsupportedMediaError,
    persist_upload,
    safe_remove,
)
from app.services.preprocess import UnknownPresetError, build_pipeline, list_presets
from app.services.render import render_outputs
from app.services.translate import BaiduTranslator
from app.preprocessing import load_audio, save_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/preprocess", tags=["音频预处理"])


async def _preprocess_upload(
    file: UploadFile,
    *,
    preset: str,
    enable_vad: bool | None,
    enable_nr: bool | None,
    enable_se: bool | None,
    nr_strength: float | None,
) -> tuple[np.ndarray, int, np.ndarray, Any, str]:
    """把上传文件读成波形 → 跑预处理 → 返回 (原始, sr, 处理后, stats, 文件名)。"""
    try:
        temp_path = await persist_upload(file, settings.max_upload_mb * 1024 * 1024)
    except UnsupportedMediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        audio, sr = load_audio(temp_path, target_sr=16000)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        safe_remove(temp_path)

    try:
        pipeline = build_pipeline(
            preset,
            enable_vad=enable_vad,
            enable_nr=enable_nr,
            enable_se=enable_se,
            nr_strength=nr_strength,
        )
    except UnknownPresetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processed, stats = pipeline.process(audio, sr)
    return audio, sr, processed, stats, file.filename or "upload"


def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    save_audio(audio, buffer, sr)
    buffer.seek(0)
    return buffer.read()


@router.get("/presets", summary="列出所有预处理预设")
def get_presets() -> dict[str, dict[str, Any]]:
    return list_presets()


@router.post(
    "/audio",
    response_model=PreprocessResponse,
    summary="预处理音频，返回 base64 编码的 WAV",
)
async def preprocess_audio(
    file: UploadFile = File(..., description="音视频文件"),
    preset: str = Query("meeting_room", description="预处理预设"),
    enable_vad: bool | None = Query(None),
    enable_nr: bool | None = Query(None),
    enable_se: bool | None = Query(None),
    nr_strength: float | None = Query(None, ge=0.0, le=1.0),
) -> PreprocessResponse:
    _, sr, processed, stats, filename = await _preprocess_upload(
        file,
        preset=preset,
        enable_vad=enable_vad,
        enable_nr=enable_nr,
        enable_se=enable_se,
        nr_strength=nr_strength,
    )
    return PreprocessResponse(
        filename=filename,
        preset_used=preset,
        sample_rate=sr,
        original_duration_s=round(stats.input_duration, 2),
        processed_duration_s=round(stats.output_duration, 2),
        processed_audio_base64=base64.b64encode(_to_wav_bytes(processed, sr)).decode("ascii"),
        stats=stats.to_dict(),
    )


@router.post(
    "/audio/raw",
    summary="预处理音频，直接返回 WAV 二进制（可给 <audio> 标签播放）",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def preprocess_audio_raw(
    file: UploadFile = File(...),
    preset: str = Query("meeting_room"),
    enable_vad: bool | None = Query(None),
    enable_nr: bool | None = Query(None),
    enable_se: bool | None = Query(None),
    nr_strength: float | None = Query(None, ge=0.0, le=1.0),
) -> Response:
    _, sr, processed, stats, _ = await _preprocess_upload(
        file,
        preset=preset,
        enable_vad=enable_vad,
        enable_nr=enable_nr,
        enable_se=enable_se,
        nr_strength=nr_strength,
    )
    return Response(
        content=_to_wav_bytes(processed, sr),
        media_type="audio/wav",
        headers={
            "X-Original-Duration": str(round(stats.input_duration, 2)),
            "X-Processed-Duration": str(round(stats.output_duration, 2)),
            "X-Processing-Time-Ms": str(round(stats.processing_time_total * 1000, 1)),
            "X-Vad-Segments": str(stats.vad_segments_detected),
            "X-Preset": preset,
        },
    )


compare_router = APIRouter(tags=["音频预处理"])


def _transcribe_array(
    engine: WhisperEngine,
    audio: np.ndarray,
    sr: int,
    *,
    language: str,
    output_format: OutputFormat,
    translator: BaiduTranslator,
) -> dict[str, Any]:
    """对内存中的波形做转写（faster-whisper 只接受文件路径，先落临时 WAV）。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        save_audio(audio, tmp.name, sr)
        tmp_path = tmp.name
    try:
        segments, meta = engine.transcribe(tmp_path, language=language)
    finally:
        safe_remove(tmp_path)

    result = render_outputs(
        segments, output_format, translator=translator, source_lang=language
    )
    result["segments"] = segments
    result["meta"] = meta
    return result


@compare_router.post(
    "/preprocess-and-transcribe",
    response_model=PreprocessCompareResponse,
    summary="预处理后转写，并与原始音频的转写结果对比",
)
async def preprocess_and_transcribe(
    file: UploadFile = File(...),
    preset: str = Query("meeting_room"),
    language: str = Query("en"),
    output_format: OutputFormat = Query("both"),
    enable_vad: bool | None = Query(None),
    enable_nr: bool | None = Query(None),
    enable_se: bool | None = Query(None),
    nr_strength: float | None = Query(None, ge=0.0, le=1.0),
    engine: WhisperEngine = Depends(get_engine),
    translator: BaiduTranslator = Depends(get_translator),
) -> PreprocessCompareResponse:
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载，请稍后重试或调用 /model/load")

    audio, sr, processed, stats, filename = await _preprocess_upload(
        file,
        preset=preset,
        enable_vad=enable_vad,
        enable_nr=enable_nr,
        enable_se=enable_se,
        nr_strength=nr_strength,
    )

    baseline = _transcribe_array(
        engine, audio, sr,
        language=language, output_format=output_format, translator=translator,
    )
    enhanced = _transcribe_array(
        engine, processed, sr,
        language=language, output_format=output_format, translator=translator,
    )

    return PreprocessCompareResponse(
        filename=filename,
        preset_used=preset,
        language=language,
        stats=stats.to_dict(),
        baseline=baseline,
        enhanced=enhanced,
        comparison=TranscribeComparison(
            baseline_text=baseline.get("txt") or "",
            enhanced_text=enhanced.get("txt") or "",
            note="对比两段文本：噪声环境下预处理后的转写通常错误更少。",
        ),
    )
