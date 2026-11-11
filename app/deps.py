"""共享依赖：进程内单例。

Whisper 模型常驻内存，必须全局唯一；翻译器带缓存，同样只建一个实例。
用 ``Depends`` 注入而不是在路由里 import 全局变量，测试时可以方便地替换。
"""

from __future__ import annotations

from app.config import settings
from app.services.asr import WhisperEngine
from app.services.translate import BaiduTranslator

_engine = WhisperEngine()
_translator = BaiduTranslator(settings.baidu_app_id, settings.baidu_app_key)


def get_engine() -> WhisperEngine:
    return _engine


def get_translator() -> BaiduTranslator:
    return _translator
