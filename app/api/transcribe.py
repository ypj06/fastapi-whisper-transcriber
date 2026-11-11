"""转写路由：单文件上传与目录批量处理。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.config import settings
from app.deps import get_engine, get_translator
from app.schemas import (
    BatchFileResult,
    BatchRequest,
    BatchResponse,
    OutputFormat,
    TranscribeFileResponse,
)
from app.services.asr import WhisperEngine
from app.services.media import (
    FileTooLargeError,
    UnsupportedMediaError,
    persist_upload,
    safe_remove,
    scan_media_files,
    write_text_output,
)
from app.services.render import render_outputs
from app.services.subtitle import build_txt
from app.services.translate import BaiduTranslator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcribe", tags=["转写"])


@router.post(
    "/file",
    response_model=TranscribeFileResponse,
    summary="上传单个音视频文件并转写",
    response_description="转写文本、SRT 字幕与带时间戳的分段",
)
async def transcribe_file(
    file: UploadFile = File(..., description="音视频文件"),
    language: str = Query("en", description="识别语言；auto 表示自动检测"),
    target_lang: str | None = Query(None, description="翻译目标语言，例如 zh"),
    output_format: OutputFormat = Query("both", description="输出内容：srt / txt / both"),
    bilingual: bool = Query(False, description="翻译时是否输出原文+译文双语字幕"),
    vad_filter: bool = Query(True, description="是否启用 Whisper 内置的 VAD 过滤静音"),
    engine: WhisperEngine = Depends(get_engine),
    translator: BaiduTranslator = Depends(get_translator),
) -> TranscribeFileResponse:
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载，请稍后重试或调用 /model/load")

    try:
        temp_path = await persist_upload(file, settings.max_upload_mb * 1024 * 1024)
    except UnsupportedMediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        segments, meta = engine.transcribe(temp_path, language=language, vad_filter=vad_filter)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        safe_remove(temp_path)

    rendered = render_outputs(
        segments,
        output_format,
        translator=translator,
        source_lang=language,
        target_lang=target_lang,
        bilingual=bilingual,
    )
    return TranscribeFileResponse(
        filename=file.filename or "upload",
        language=meta["language"],
        segments=segments,
        **rendered,
    )


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="批量转写目录下的音视频文件",
)
def transcribe_batch(
    req: BatchRequest,
    engine: WhisperEngine = Depends(get_engine),
    translator: BaiduTranslator = Depends(get_translator),
) -> BatchResponse:
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载，请稍后重试或调用 /model/load")

    try:
        media_files = scan_media_files(req.directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results: list[BatchFileResult] = []
    log: list[str] = []

    for path in media_files:
        stem = Path(path).stem
        log.append(f"处理中：{Path(path).name}")
        outputs: list[str] = []
        try:
            segments, _ = engine.transcribe(path, language=req.languages[0])
            if len(req.languages) > 1:
                log.append(f"  [warn] 暂只使用首个语言 {req.languages[0]}，其余语言已忽略")

            if req.gen_srt:
                suffix = f"_{req.target_lang}.srt" if req.target_lang else ".srt"
                if _exists(req.directory, stem, suffix, req.overwrite):
                    log.append(f"  跳过（已存在）：{stem}{suffix}")
                else:
                    srt = render_outputs(
                        segments, "srt", translator=translator,
                        source_lang=req.languages[0], target_lang=req.target_lang,
                    )["srt"]
                    outputs.append(write_text_output(req.directory, stem, suffix, srt or ""))
                    log.append(f"  写入：{stem}{suffix}")

            if req.gen_txt:
                suffix = f"_{req.languages[0]}.txt"
                if _exists(req.directory, stem, suffix, req.overwrite):
                    log.append(f"  跳过（已存在）：{stem}{suffix}")
                else:
                    outputs.append(
                        write_text_output(req.directory, stem, suffix, build_txt(segments))
                    )
                    log.append(f"  写入：{stem}{suffix}")

            status: Literal["ok", "skipped", "failed"] = "ok" if outputs else "skipped"
            results.append(BatchFileResult(file=str(path), status=status, outputs=outputs))

        except Exception as exc:  # 单个文件失败不影响整批
            logger.exception("批量转写失败：%s", path)
            log.append(f"  失败：{exc}")
            results.append(BatchFileResult(file=str(path), status="failed", error=str(exc)))

    return BatchResponse(
        total=len(results),
        succeeded=sum(r.status == "ok" for r in results),
        skipped=sum(r.status == "skipped" for r in results),
        failed=sum(r.status == "failed" for r in results),
        results=results,
        log=log,
    )


def _exists(directory: str, stem: str, suffix: str, overwrite: bool) -> bool:
    return not overwrite and (Path(directory) / f"{stem}{suffix}").exists()
