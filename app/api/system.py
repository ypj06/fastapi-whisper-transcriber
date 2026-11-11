"""系统路由：根路径与健康检查。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.config import settings
from app.deps import get_engine
from app.schemas import HealthResponse
from app.services.asr import WhisperEngine
from app.preprocessing import PRESETS

router = APIRouter(tags=["系统"])


@router.get("/", summary="服务运行状态")
def root(engine: WhisperEngine = Depends(get_engine)) -> dict:
    return {
        "service": "fastapi-whisper-transcriber",
        "version": __version__,
        "status": "running",
        "device": engine.device,
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(engine: WhisperEngine = Depends(get_engine)) -> HealthResponse:
    """供容器编排 / 前端探活使用：模型未就绪时返回 degraded。"""
    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        device=engine.device,
        translation_enabled=settings.translation_enabled,
        presets=list(PRESETS),
    )
