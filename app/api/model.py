"""模型管理路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps import get_engine
from app.schemas import ModelInfo, ModelLoadResponse
from app.services.asr import WhisperEngine

router = APIRouter(prefix="/model", tags=["模型"])


@router.post("/load", response_model=ModelLoadResponse, summary="加载或切换 Whisper 模型")
def load_model(
    model_choice: int = Query(4, ge=1, le=7, description="模型编号 1~7"),
    engine: WhisperEngine = Depends(get_engine),
) -> ModelLoadResponse:
    try:
        info = engine.load(model_choice)
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail="未安装 faster-whisper，请先执行 pip install -r requirements.txt"
        ) from exc
    except Exception as exc:  # 模型下载失败、显存不足等
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc

    return ModelLoadResponse(
        message=f"模型 {info['model_name']} 已加载（{info['device']}）",
        model_name=info["model_name"],
        device=info["device"],
        compute_type=info["compute_type"],
    )


@router.get("/info", response_model=ModelInfo, summary="查询当前模型状态")
def model_info(engine: WhisperEngine = Depends(get_engine)) -> ModelInfo:
    return ModelInfo(**engine.info())
