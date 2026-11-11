"""路由聚合。"""

from fastapi import APIRouter

from app.api import model, preprocess, system, transcribe

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(model.router)
api_router.include_router(transcribe.router)
api_router.include_router(preprocess.router)
api_router.include_router(preprocess.compare_router)

__all__ = ["api_router"]
