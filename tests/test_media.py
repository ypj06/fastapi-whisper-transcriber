"""文件工具测试：目录扫描、上传落盘、超限与类型校验。"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.services.media import (
    FileTooLargeError,
    UnsupportedMediaError,
    persist_upload,
    safe_remove,
    scan_media_files,
    write_text_output,
)


def _upload(content: bytes, filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def test_scan_media_files_is_case_insensitive_and_sorted(tmp_path: Path):
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    found = scan_media_files(tmp_path)

    assert [Path(p).name for p in found] == ["a.wav", "b.MP4"]


def test_scan_media_files_rejects_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_media_files(tmp_path / "nope")


def test_scan_media_files_rejects_file(tmp_path: Path):
    target = tmp_path / "a.wav"
    target.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        scan_media_files(target)


def test_persist_upload_writes_temp_file():
    path = asyncio.run(persist_upload(_upload(b"0123456789", "clip.wav"), max_bytes=1024))
    try:
        assert Path(path).read_bytes() == b"0123456789"
        assert Path(path).suffix == ".wav"
    finally:
        safe_remove(path)


def test_persist_upload_rejects_unsupported_type():
    with pytest.raises(UnsupportedMediaError):
        asyncio.run(persist_upload(_upload(b"data", "report.pdf"), max_bytes=1024))


def test_persist_upload_enforces_size_limit():
    with pytest.raises(FileTooLargeError):
        asyncio.run(persist_upload(_upload(b"x" * 4096, "big.wav"), max_bytes=1024))


def test_write_text_output(tmp_path: Path):
    output = write_text_output(str(tmp_path), "demo", ".srt", "hello")
    assert Path(output).read_text(encoding="utf-8") == "hello"


def test_safe_remove_is_idempotent(tmp_path: Path):
    target = tmp_path / "tmp.bin"
    target.write_bytes(b"x")
    safe_remove(str(target))
    safe_remove(str(target))  # 再删一次不应抛异常
    assert not target.exists()
