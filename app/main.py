"""应用入口。

启动：
    uvicorn app.main:app --host 0.0.0.0 --port 8000

接口文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import api_router
from app.config import settings
from app.deps import get_engine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
基于 **faster-whisper** 的音视频转写服务，附带一套音频预处理能力。

**主要能力**

* `POST /transcribe/file` —— 单文件转写，输出 SRT / TXT / 带时间戳分段，可选翻译
* `POST /transcribe/batch` —— 目录批量转写，单文件失败不影响整批
* `POST /preprocess/audio` —— 降噪 + VAD + 信号增强，5 套场景预设
* `POST /preprocess-and-transcribe` —— 预处理前后的转写结果对比
* `POST /model/load` —— 运行时切换模型（tiny ~ large-v3）

完整说明见仓库 `docs/API.md`。
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型。

    即使模型加载失败也照常对外提供服务（`/health` 会返回 degraded），
    方便在没有 GPU、没有网络的机器上先调通接口。
    """
    engine = get_engine()
    try:
        info = engine.load(settings.default_model_choice)
        logger.info("模型就绪：%s（device=%s）", info["model_name"], info["device"])
    except Exception as exc:  # noqa: BLE001 - 启动期兜底
        logger.warning("模型预加载失败，服务将以降级模式启动：%s", exc)
    yield
    logger.info("服务关闭")


def create_app() -> FastAPI:
    """应用工厂，便于测试时构造独立实例。"""
    application = FastAPI(
        title="Whisper 音视频转写服务",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        contact={"name": "ypj06", "url": "https://github.com/ypj06"},
        license_info={"name": "MIT"},
    )

    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
    )
