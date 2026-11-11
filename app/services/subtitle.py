"""字幕与纯文本生成。

这些函数是纯函数（只依赖入参），因此可以直接单元测试，
也是"一次转写、多种输出"改造的关键：转写只跑一遍，
SRT / TXT / 分段信息都从同一批 segments 派生。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

SegmentLike = Mapping[str, Any]


def format_timestamp(seconds: float) -> str:
    """把秒数格式化成 SRT 时间戳 ``HH:MM:SS,mmm``。

    >>> format_timestamp(3661.5)
    '01:01:01,500'
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _normalize(segment: SegmentLike) -> tuple[float, float, str]:
    return (
        float(segment["start"]),
        float(segment["end"]),
        str(segment["text"]).strip(),
    )


def build_srt(
    segments: Iterable[SegmentLike],
    translate: Callable[[str], str] | None = None,
) -> str:
    """由转写分段生成 SRT 字幕。

    Args:
        segments: 每项包含 ``start`` / ``end`` / ``text`` 的映射。
        translate: 可选翻译函数；传入后字幕正文替换为译文（双语字幕时
            可自行拼装）。

    Returns:
        完整的 SRT 文本；空文本的片段会被跳过，避免出现空字幕块。
    """
    blocks: list[str] = []
    index = 0
    for segment in segments:
        start, end, text = _normalize(segment)
        if not text:
            continue
        if translate is not None:
            text = translate(text)
        index += 1
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n"
        )
    return "\n".join(blocks)


def build_bilingual_srt(
    segments: Iterable[SegmentLike],
    translate: Callable[[str], str],
) -> str:
    """生成原文 + 译文两行的双语 SRT 字幕。"""
    blocks: list[str] = []
    index = 0
    for segment in segments:
        start, end, text = _normalize(segment)
        if not text:
            continue
        translated = translate(text)
        index += 1
        body = text if translated == text else f"{text}\n{translated}"
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{body}\n"
        )
    return "\n".join(blocks)


def build_txt(segments: Iterable[SegmentLike]) -> str:
    """由转写分段生成纯文本（每行一句，方便后续给 LLM 处理）。"""
    return "\n".join(
        text for _, _, text in (_normalize(s) for s in segments) if text
    )


def count_words(text: str) -> int:
    """粗略统计词数：空白切分，对中文等无空格语言会低估，仅用于对比展示。"""
    return len(text.split())
