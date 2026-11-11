# 接口文档

服务启动后可在 `/docs`（Swagger UI）与 `/redoc` 查看交互式文档，本文件是同一份内容的文字版本。

- Base URL：`http://127.0.0.1:8000`
- 请求体：除文件上传外均为 JSON；上传接口使用 `multipart/form-data`
- 通用错误格式：`{"detail": "错误说明"}`

## 状态码约定

| 状态码 | 场景 |
| --- | --- |
| 200 | 成功 |
| 400 | 参数合法但语义不成立（目录不是目录、未知预处理预设、音频无法解码） |
| 404 | 批量转写时目录不存在 |
| 413 | 上传文件超过 `MAX_UPLOAD_MB` |
| 415 | 上传文件扩展名不在支持列表 |
| 422 | Pydantic 校验失败（如 `output_format=docx`） |
| 503 | 模型尚未加载 / 未安装 faster-whisper |

---

## 系统

### `GET /`

```json
{
  "service": "fastapi-whisper-transcriber",
  "version": "2.0.0",
  "status": "running",
  "device": "cpu",
  "docs": "/docs"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "translation_enabled": false,
  "presets": ["mobile", "desktop_high_quality", "noisy_environment", "meeting_room", "lightweight"]
}
```

`status` 为 `degraded` 表示进程起来了但模型没加载成功（例如首次下载中断），此时转写接口返回 503。

---

## 转写

### `POST /transcribe/file`

上传单个音视频文件转写。

| 参数 | 位置 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `file` | form | file | 必填 | 支持 mp4/mkv/webm/flv/avi/mov/mp3/wav/m4a/flac/ogg |
| `language` | query | string | `en` | 识别语言；`auto` 表示自动检测 |
| `target_lang` | query | string | 空 | 翻译目标语言，如 `zh` |
| `output_format` | query | enum | `both` | `srt` / `txt` / `both` |
| `bilingual` | query | bool | `false` | 翻译时输出原文 + 译文两行 |
| `vad_filter` | query | bool | `true` | 是否用 Whisper 内置 VAD 过滤静音 |

```bash
curl -X POST http://127.0.0.1:8000/transcribe/file \
  -F "file=@samples/english_video.mp4" \
  -G --data-urlencode "language=en" \
     --data-urlencode "output_format=both" \
     --data-urlencode "target_lang=zh" \
     --data-urlencode "bilingual=true"
```

响应：

```json
{
  "filename": "english_video.mp4",
  "language": "en",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "Welcome to this short demo."}
  ],
  "srt": "1\n00:00:00,000 --> 00:00:03,200\nWelcome to this short demo.\n",
  "txt": "Welcome to this short demo."
}
```

> 未配置百度翻译密钥时，`target_lang` 不会报错，字幕保持原文。

### `POST /transcribe/batch`

批量转写目录下的音视频，字幕写在源文件同目录。

```bash
curl -X POST http://127.0.0.1:8000/transcribe/batch \
  -H "Content-Type: application/json" \
  -d '{"directory":"./samples","languages":["en"],"gen_srt":true,"gen_txt":true,"overwrite":false}'
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `directory` | string | 必填 | 服务端可见的目录路径 |
| `languages` | string[] | `["en"]` | 识别语言；当前实现取第一个 |
| `target_lang` | string? | `null` | 设置后输出 `{stem}_{target}.srt` |
| `model_choice` | int | `4` | 模型编号 1~7 |
| `gen_srt` / `gen_txt` | bool | `true` | 是否生成对应文件 |
| `overwrite` | bool | `false` | 为 false 时跳过已存在的输出 |

响应包含逐个文件的结果与人类可读日志：

```json
{
  "total": 1,
  "succeeded": 1,
  "skipped": 0,
  "failed": 0,
  "results": [
    {"file": "samples/english_video.mp4", "status": "ok",
     "outputs": ["samples/english_video.srt", "samples/english_video_en.txt"], "error": null}
  ],
  "log": ["处理中：english_video.mp4", "  写入：english_video.srt", "  写入：english_video_en.txt"]
}
```

---

## 音频预处理

### `GET /preprocess/presets`

返回 5 套预设的生效配置：

```json
{
  "meeting_room": {
    "vad": "spectral_entropy",
    "noise_reduction": "wiener",
    "drc": true,
    "eq": "clarity",
    "agc": true,
    "deess": true,
    "dereverb": true
  }
}
```

### `POST /preprocess/audio`

对上传音频做降噪 / VAD / 增强，返回 base64 编码的 WAV。

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `file` | form | 音视频文件 |
| `preset` | query | 预设名，默认 `meeting_room` |
| `enable_vad` / `enable_nr` / `enable_se` | query | 覆盖预设中的开关，不传则沿用预设 |
| `nr_strength` | query | 降噪强度 0~1，越界会被裁剪 |

```json
{
  "filename": "noisy.wav",
  "preset_used": "meeting_room",
  "sample_rate": 16000,
  "original_duration_s": 12.4,
  "processed_duration_s": 10.9,
  "processed_audio_base64": "UklGRi...",
  "stats": {
    "input_duration_s": 12.4,
    "output_duration_s": 10.9,
    "duration_reduction_pct": 12.1,
    "vad_segments": 7,
    "vad_speech_ratio": 0.88,
    "processing_time_total_ms": 640.2,
    "input_rms_db": -22.1,
    "output_rms_db": -18.4,
    "noise_reduction_applied": true,
    "signal_enhancement_applied": true
  }
}
```

### `POST /preprocess/audio/raw`

同上，但直接返回 `audio/wav` 二进制，便于浏览器 `<audio>` 播放。统计信息放在响应头：

```
X-Original-Duration: 12.4
X-Processed-Duration: 10.9
X-Processing-Time-Ms: 640.2
X-Vad-Segments: 7
X-Preset: meeting_room
```

### `POST /preprocess-and-transcribe`

一体化对比接口：先用原始音频转写，再预处理后转写。

```bash
curl -X POST http://127.0.0.1:8000/preprocess-and-transcribe \
  -F "file=@samples/english_video.mp4" \
  -G --data-urlencode "preset=meeting_room" \
     --data-urlencode "language=en" \
     --data-urlencode "output_format=both"
```

响应结构：

```json
{
  "filename": "english_video.mp4",
  "preset_used": "meeting_room",
  "language": "en",
  "stats": {"...": "同 /preprocess/audio 的 stats"},
  "baseline": {"srt": "...", "txt": "...", "segments": [], "meta": {}},
  "enhanced": {"srt": "...", "txt": "...", "segments": [], "meta": {}},
  "comparison": {
    "baseline_text": "...",
    "enhanced_text": "...",
    "note": "对比两段文本：噪声环境下预处理后的转写通常错误更少。"
  }
}
```

---

## 模型

### `POST /model/load`

```bash
curl -X POST "http://127.0.0.1:8000/model/load?model_choice=4"
```

| 编号 | 模型 | 体积（约） |
| --- | --- | --- |
| 1 | tiny | 150 MB |
| 2 | base | 300 MB |
| 3 | small | 1 GB |
| 4 | medium | 3 GB |
| 5 | large-v1 | 6 GB |
| 6 | large-v2 | 6 GB |
| 7 | large-v3 | 6 GB |

重复加载同一编号会直接复用已驻留内存的模型，不会重复占用显存。

### `GET /model/info`

```json
{
  "loaded": true,
  "model_choice": 2,
  "model_name": "base",
  "device": "cpu",
  "compute_type": "int8",
  "model_dir": "D:/work/fastapi-whisper-transcriber/models"
}
```
