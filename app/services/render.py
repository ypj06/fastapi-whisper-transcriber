"""把转写分段渲染成对外的输出（SRT / TXT）。

转写只在 :mod:`app.services.asr` 里跑一次，SRT、纯文本、双语字幕都由
这里从同一批 segments 派生，避免原实现中"一次请求转写三次"的浪费。
"""

from __future__ import annotations

from typing import Literal

from app.services.subtitle import build_bilingual_srt, build_srt, build_txt
from app.services.translate import BaiduTranslator

OutputFormat = Literal["srt", "txt", "both"]


def _translate_fn(translator: BaiduTranslator, texts: list[str], source: str, target: str):
    """把批量翻译结果包装成"单条文本 → 译文"的回调，供字幕生成器使用。"""
    mapping = dict(zip(texts, translator.translate_lines(texts, source, target)))
    return lambda text: mapping.get(text, text)


def render_outputs(
    segments: list[dict],
    output_format: OutputFormat,
    *,
    translator: BaiduTranslator,
    source_lang: str,
    target_lang: str | None = None,
    bilingual: bool = False,
) -> dict[str, str | None]:
    """生成 ``{"srt": ..., "txt": ...}``。

    Args:
        segments: 转写分段。
        output_format: 需要的产物。
        translator: 翻译器；未配置密钥时其 ``translate_lines`` 原样返回。
        source_lang: 源语言。
        target_lang: 目标语言；为 ``None`` 时输出原文字幕。
        bilingual: 是否输出"原文 + 译文"两行字幕。
    """
    srt: str | None = None
    txt: str | None = None

    if output_format in ("srt", "both"):
        if target_lang:
            translate = _translate_fn(
                translator, [seg["text"] for seg in segments], source_lang, target_lang
            )
            srt = (
                build_bilingual_srt(segments, translate)
                if bilingual
                else build_srt(segments, translate=translate)
            )
        else:
            srt = build_srt(segments)

    if output_format in ("txt", "both"):
        txt = build_txt(segments)

    return {"srt": srt, "txt": txt}
