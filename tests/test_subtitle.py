"""字幕与纯文本生成的单元测试（纯函数，无需任何推理依赖）。"""

from __future__ import annotations

import pytest

from app.services.subtitle import (
    build_bilingual_srt,
    build_srt,
    build_txt,
    count_words,
    format_timestamp,
)

SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": " Hello world "},
    {"start": 1.5, "end": 3601.25, "text": "Second line"},
]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.004, "00:01:01,004"),
        (3661.5, "01:01:01,500"),
        (-3, "00:00:00,000"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_build_srt_has_sequence_and_timestamps():
    srt = build_srt(SEGMENTS)

    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello world")
    assert "\n\n2\n00:00:01,500 --> 01:00:01,250\nSecond line" in srt
    assert srt.endswith("\n")


def test_build_srt_skips_blank_segments():
    srt = build_srt(
        [
            {"start": 0, "end": 1, "text": "keep"},
            {"start": 1, "end": 2, "text": "   "},
            {"start": 2, "end": 3, "text": "keep too"},
        ]
    )
    assert "1\n" in srt and "2\n" in srt
    assert "3\n" not in srt


def test_build_srt_applies_translation():
    srt = build_srt(SEGMENTS, translate=lambda text: text.upper())
    assert "HELLO WORLD" in srt


def test_build_bilingual_srt_contains_both_languages():
    srt = build_bilingual_srt([{"start": 0, "end": 1, "text": "hi"}], lambda _: "你好")
    body = srt.splitlines()[2:4]
    assert body == ["hi", "你好"]


def test_build_bilingual_srt_does_not_duplicate_when_untranslated():
    srt = build_bilingual_srt([{"start": 0, "end": 1, "text": "hi"}], lambda text: text)
    assert srt.splitlines()[2] == "hi"
    assert len([line for line in srt.splitlines() if line == "hi"]) == 1


def test_build_txt_joins_lines_without_blank():
    assert build_txt(SEGMENTS) == "Hello world\nSecond line"


def test_count_words():
    assert count_words("a b c") == 3
    assert count_words("") == 0
