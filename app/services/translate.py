"""字幕翻译（百度翻译开放平台）。

设计要点：
1. 未配置密钥时 ``enabled`` 为 False，调用方直接返回原文，功能不报错；
2. 逐条调用会出现"一个视频几百次请求"的问题，这里按行批量提交
   （百度接口支持多行文本，返回同样多行的结果），失败时自动回落到逐条；
3. 结果带本地缓存，重复文案不会重复计费。
"""

from __future__ import annotations

import hashlib
import logging
import random

import requests

from app.config import LANG_MAP

logger = logging.getLogger(__name__)

ENDPOINT = "https://fanyi-api.baidu.com/api/trans/vip/translate"
MAX_LINES_PER_REQUEST = 20


class BaiduTranslator:
    """百度通用翻译 API 的轻量封装。"""

    def __init__(self, app_id: str, app_key: str, timeout: float = 10.0) -> None:
        self.app_id = app_id
        self.app_key = app_key
        self.timeout = timeout
        self._cache: dict[tuple[str, str, str], str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_key)

    # ------------------------------------------------------------------ 内部
    def _sign(self, text: str, salt: int, target_lang: str) -> dict[str, str | int]:
        sign = hashlib.md5(
            f"{self.app_id}{text}{salt}{self.app_key}".encode("utf-8")
        ).hexdigest()
        return {
            "q": text,
            "from": "auto",
            "to": target_lang,
            "appid": self.app_id,
            "salt": salt,
            "sign": sign,
        }

    def _request(self, text: str, source_lang: str, target_lang: str) -> list[str] | None:
        params = self._sign(text, random.randint(32768, 65536), LANG_MAP.get(target_lang, target_lang))
        params["from"] = LANG_MAP.get(source_lang, "auto")
        try:
            response = requests.get(ENDPOINT, params=params, timeout=self.timeout)
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:  # 网络异常 / 非 JSON
            logger.warning("百度翻译请求失败：%s", exc)
            return None

        if payload.get("error_code"):
            logger.warning("百度翻译返回错误：%s %s", payload.get("error_code"), payload.get("error_msg"))
            return None

        results = payload.get("trans_result") or []
        return [item.get("dst", "") for item in results]

    # ------------------------------------------------------------------ 对外
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> str:
        """翻译单条文本；任何异常都回退为原文，保证转写主流程不中断。"""
        if not self.enabled or not text.strip():
            return text

        key = (text, source_lang, target_lang)
        if key in self._cache:
            return self._cache[key]

        lines = self._request(text, source_lang, target_lang)
        result = lines[0] if lines else text
        self._cache[key] = result
        return result

    def translate_lines(
        self, texts: list[str], source_lang: str = "auto", target_lang: str = "zh"
    ) -> list[str]:
        """批量翻译多行文本，行数与入参严格一致。"""
        if not self.enabled:
            return list(texts)

        output: list[str] = []
        for start in range(0, len(texts), MAX_LINES_PER_REQUEST):
            chunk = texts[start : start + MAX_LINES_PER_REQUEST]
            cached = [self._cache.get((t, source_lang, target_lang)) for t in chunk]

            missing = [t for t, hit in zip(chunk, cached) if hit is None]
            fresh: list[str] = []
            if missing:
                fresh = self._request("\n".join(missing), source_lang, target_lang) or []
                if len(fresh) != len(missing):
                    # 行数对不上：退回逐条翻译，保证对齐不串行
                    fresh = [self.translate(t, source_lang, target_lang) for t in missing]

            fresh_iter = iter(fresh)
            for text, hit in zip(chunk, cached):
                if hit is not None:
                    output.append(hit)
                    continue
                translated = next(fresh_iter, text)
                self._cache[(text, source_lang, target_lang)] = translated
                output.append(translated)

        return output
