"""文件与媒体工具：上传落盘、扩展名校验、目录扫描、临时文件清理。"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import UploadFile

from app.config import MEDIA_EXTENSIONS


class UnsupportedMediaError(ValueError):
    """上传的文件扩展名不在支持列表内。"""


class FileTooLargeError(ValueError):
    """上传文件超过大小上限。"""


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def scan_media_files(directory: str | Path) -> list[str]:
    """列出目录下的音视频文件（不递归，扩展名不区分大小写）。

    Raises:
        FileNotFoundError: 目录不存在。
        NotADirectoryError: 传入的是文件而非目录。
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    files = [
        str(item) for item in sorted(directory.iterdir())
        if item.is_file() and is_supported(item)
    ]
    return files


async def persist_upload(upload: UploadFile, max_bytes: int) -> str:
    """把上传文件写入临时文件并返回路径。

    分块写入以便在超限时立即中止，而不是先把整个文件读进内存。

    Raises:
        UnsupportedMediaError: 扩展名不支持。
        FileTooLargeError: 超过 ``max_bytes``。
    """
    filename = upload.filename or "upload.bin"
    if not is_supported(filename):
        raise UnsupportedMediaError(
            f"不支持的文件类型：{Path(filename).suffix or '(无扩展名)'}，"
            f"支持：{', '.join(MEDIA_EXTENSIONS)}"
        )

    suffix = Path(filename).suffix.lower()
    written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                safe_remove(tmp.name)
                raise FileTooLargeError(f"文件超过 {max_bytes // (1024 * 1024)} MB 上限")
            tmp.write(chunk)
        return tmp.name
    return tmp.name  # pragma: no cover - with 语句异常时不返回


def safe_remove(path: str | None, retries: int = 3, delay: float = 0.3) -> None:
    """删除临时文件。

    Windows 上 librosa / pydub 有时还没释放文件句柄，直接删除会抛
    ``PermissionError``；这里重试几次后放弃（系统 TEMP 目录会自动清理）。
    """
    if not path:
        return
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                print(f"[warn] 临时文件暂时无法删除，留给系统清理：{path}")
                return
            time.sleep(delay)
        except OSError:
            return


def write_text_output(directory: str, stem: str, suffix: str, content: str) -> str:
    """把字幕/文本写到指定目录，返回完整路径。"""
    out_path = Path(directory) / f"{stem}{suffix}"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)
