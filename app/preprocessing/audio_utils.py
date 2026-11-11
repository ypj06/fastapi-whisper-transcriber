"""音频读写与常用信号工具。

``librosa`` / ``matplotlib`` 体积大、编译依赖多，这里改为**函数内延迟导入**：
只做音频读写时不需要装 matplotlib，CPU 环境下缺少 librosa 也不会影响
服务其它模块的导入（缺失时会在调用处给出明确报错）。
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import soundfile as sf

if TYPE_CHECKING:  # 仅用于类型检查，运行时不导入
    import matplotlib.pyplot as plt

# 含音频轨的视频容器格式
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov", ".wmv", ".m4v"}


def _import_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise ImportError("该功能需要 librosa，请先安装：pip install librosa") from exc
    return librosa


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """读取音频/视频文件并重采样到目标采样率。

    读取顺序（越靠前依赖越轻）：

    1. 视频容器 → pydub + ffmpeg 抽音轨；
    2. ``soundfile`` 支持的格式（wav / flac / ogg / mp3）→ 直接读，
       需要时用 scipy 做多相重采样；
    3. 其余格式 → 回退到 librosa（需额外安装）。

    Args:
        file_path: 输入文件路径。
        target_sr: 目标采样率（Hz）。
        mono: 是否合并为单声道。
        duration: 只读取前 N 秒。
        offset: 从第 N 秒开始读。

    Returns:
        ``(samples, sample_rate)``。
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() in VIDEO_EXTENSIONS:
        return _load_audio_from_video(file_path, target_sr, mono, duration, offset)

    try:
        with sf.SoundFile(str(file_path)) as handle:
            sr = handle.samplerate
            if offset > 0:
                handle.seek(int(offset * sr))
            frames = -1 if duration is None else int(duration * sr)
            audio = handle.read(frames=frames, dtype="float32", always_2d=True)
        if mono:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        return _resample(audio, sr, target_sr), target_sr

    except (RuntimeError, sf.LibsndfileError):
        # 罕见容器（如部分 m4a/wma）交给 librosa + ffmpeg 处理
        librosa = _import_librosa()
        return librosa.load(
            str(file_path), sr=target_sr, mono=mono, duration=duration, offset=offset
        )


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """用 scipy 多相滤波重采样，音质与 librosa 默认 soxr 相当但依赖更少。"""
    if sr == target_sr or audio.size == 0:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(sr), int(target_sr))
    return resample_poly(audio, target_sr // divisor, sr // divisor).astype(np.float32)


def _load_audio_from_video(
    file_path: Path,
    target_sr: int,
    mono: bool,
    duration: Optional[float],
    offset: float,
) -> Tuple[np.ndarray, int]:
    """用 pydub 抽取视频音轨；pydub/ffmpeg 不可用时退回 librosa。"""
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(str(file_path))
        if offset > 0 or duration is not None:
            start_ms = int(offset * 1000)
            end_ms = int((offset + duration) * 1000) if duration else len(segment)
            segment = segment[start_ms:end_ms]
        if mono and segment.channels > 1:
            segment = segment.set_channels(1)
        if segment.frame_rate != target_sr:
            segment = segment.set_frame_rate(target_sr)

        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples /= float(2 ** (8 * segment.sample_width - 1))  # 整数 → [-1, 1]
        return samples, target_sr

    except (ImportError, FileNotFoundError):
        warnings.warn(
            f"pydub/ffmpeg 不可用（{file_path.name}），回退到 librosa 读取", stacklevel=2
        )
        try:
            librosa = _import_librosa()
            return librosa.load(
                str(file_path), sr=target_sr, mono=mono, duration=duration, offset=offset
            )
        except Exception as exc:
            raise RuntimeError(
                f"无法读取 {file_path.name}。视频文件需要 ffmpeg：\n"
                "  Windows: 下载 https://ffmpeg.org/download.html 并把 bin 目录加入 PATH\n"
                "  或改用纯音频格式（.wav / .mp3 / .m4a）"
            ) from exc


load_media = load_audio


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path, io.BytesIO],
    sample_rate: int = 16000,
    format: str = "WAV",
) -> None:
    """保存音频到文件或内存缓冲区（``io.BytesIO`` 时写 WAV）。"""
    if isinstance(file_path, io.BytesIO):
        sf.write(file_path, audio, sample_rate, format=format)
        return

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format=format)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """把音频 RMS 归一化到目标电平（dBFS）。"""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-10:
        return audio
    return audio * (10 ** (target_db / 20) / rms)


def compute_rms(audio: np.ndarray) -> float:
    """均方根能量。"""
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """计算信噪比（dB），越大越干净。"""
    signal_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-15:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def mix_audio_with_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定 SNR 把噪声混入干净语音，用于构造评测数据。"""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[: len(clean)]

    clean_rms = float(np.sqrt(np.mean(clean ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms < 1e-10:
        return clean

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    return clean + noise * (target_noise_rms / noise_rms)


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """返回音视频文件时长（秒）。优先用 soundfile，失败再走 librosa。"""
    try:
        with sf.SoundFile(str(file_path)) as handle:
            return float(len(handle)) / float(handle.samplerate)
    except (RuntimeError, sf.LibsndfileError):
        librosa = _import_librosa()
        return float(librosa.get_duration(path=str(file_path)))


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """裁掉首尾静音。"""
    librosa = _import_librosa()
    trimmed, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed


def segment_audio(
    audio: np.ndarray, sr: int, segment_duration: float = 30.0, overlap: float = 1.0
) -> list[np.ndarray]:
    """把长音频切成带重叠的片段（供滑动窗口推理使用）。"""
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    step = max(1, segment_samples - overlap_samples)

    chunks: list[np.ndarray] = []
    for start in range(0, len(audio) - overlap_samples, step):
        end = min(start + segment_samples, len(audio))
        chunks.append(audio[start:end])
        if end >= len(audio):
            break
    return chunks


# --------------------------------------------------------------------- 可视化
def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("可视化功能需要 matplotlib：pip install matplotlib") from exc
    return plt


def plot_waveform(
    audio: np.ndarray, sr: int, title: str = "Waveform",
    ax: Optional["plt.Axes"] = None, color: str = "steelblue",
) -> "plt.Axes":
    """绘制波形图。"""
    plt = _import_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio, color=color, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, max(float(time[-1]) if len(time) else 0.0, 1e-6))
    return ax


def plot_spectrogram(
    audio: np.ndarray, sr: int, title: str = "Spectrogram",
    ax: Optional["plt.Axes"] = None, n_fft: int = 512, hop_length: int = 128,
) -> "plt.Axes":
    """绘制梅尔频谱图。"""
    plt = _import_pyplot()
    librosa = _import_librosa()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    s_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length),
        ref=np.max,
    )
    img = librosa.display.specshow(
        s_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return ax


def compare_audio(
    original: np.ndarray, processed: np.ndarray, sr: int,
    title_original: str = "Original", title_processed: str = "Processed",
) -> "plt.Figure":
    """画 2×2 对比图：原始/处理后的波形与梅尔谱。"""
    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_waveform(original, sr, title_original, ax=axes[0, 0])
    plot_waveform(processed, sr, title_processed, ax=axes[0, 1])
    plot_spectrogram(original, sr, f"{title_original} Spectrogram", ax=axes[1, 0])
    plot_spectrogram(processed, sr, f"{title_processed} Spectrogram", ax=axes[1, 1])
    plt.tight_layout()
    return fig
