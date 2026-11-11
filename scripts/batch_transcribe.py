"""命令行批量转写客户端。

适合在服务器上跑批：发请求给已启动的 FastAPI 服务，把 SRT/TXT 落到源目录。

用法：
    python scripts/batch_transcribe.py --dir ./samples --lang en
    python scripts/batch_transcribe.py --dir ./samples --lang en --target zh --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="调用 /transcribe/batch 批量转写目录下的音视频")
    parser.add_argument("--dir", required=True, help="音视频所在目录（服务端可见的路径）")
    parser.add_argument("--lang", default="en", help="识别语言，默认 en")
    parser.add_argument("--target", default=None, help="翻译目标语言，如 zh；不填则不翻译")
    parser.add_argument("--model", type=int, default=4, choices=range(1, 8), metavar="1-7", help="模型编号")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的字幕文件")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="后端地址")
    args = parser.parse_args()

    payload = {
        "directory": args.dir,
        "languages": [args.lang],
        "target_lang": args.target,
        "model_choice": args.model,
        "gen_srt": True,
        "gen_txt": True,
        "overwrite": args.overwrite,
    }

    try:
        response = requests.post(f"{args.api}/transcribe/batch", json=payload, timeout=None)
    except requests.RequestException as exc:
        print(f"[error] 无法连接后端 {args.api}：{exc}", file=sys.stderr)
        return 2

    if not response.ok:
        print(f"[error] {response.status_code} {response.text}", file=sys.stderr)
        return 1

    data = response.json()
    for line in data["log"]:
        print(line)
    print(f"\n完成：成功 {data['succeeded']} / 跳过 {data['skipped']} / 失败 {data['failed']}")
    return 0 if data["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
