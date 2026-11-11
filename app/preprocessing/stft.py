"""轻量 STFT / ISTFT 实现（仅依赖 numpy）。

预处理里的降噪、去混响、谱熵 VAD 都需要短时傅里叶变换。原实现调用
``librosa.stft`` / ``librosa.istft``，为了两个函数引入 librosa（连同 numba、
llvmlite 等依赖）代价过大，这里用 numpy 实现一版，参数语义与 librosa 保持一致：

* 周期 Hann 窗（``np.hanning(n+1)[:-1]``）；
* ``center=True`` 时两端各补 ``n_fft // 2``；
* 帧移 ``hop_length``，输出形状 ``(n_fft // 2 + 1, n_frames)``；
* 逆变换用重叠相加 + 窗平方归一化，可近似无损还原。
"""

from __future__ import annotations

import numpy as np


def _window(n_fft: int) -> np.ndarray:
    """周期 Hann 窗，与 librosa/scipy 的 ``hann`` 约定一致。"""
    return np.hanning(n_fft + 1)[:-1].astype(np.float64)


def stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    center: bool = True,
) -> np.ndarray:
    """短时傅里叶变换，返回复数谱 ``(n_fft // 2 + 1, n_frames)``。"""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex128)

    if center:
        pad = n_fft // 2
        audio = np.pad(audio, (pad, pad), mode="constant")

    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size), mode="constant")

    n_frames = 1 + (audio.size - n_fft) // hop_length
    frame_idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = audio[frame_idx] * _window(n_fft)
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(
    spectrum: np.ndarray,
    hop_length: int = 128,
    n_fft: int | None = None,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """逆短时傅里叶变换（重叠相加 + 窗平方归一化）。

    Args:
        spectrum: ``stft`` 的输出，形状 ``(n_fft // 2 + 1, n_frames)``。
        hop_length: 帧移，需与正变换一致。
        n_fft: 窗长，缺省时由频谱长度反推。
        center: 是否按 ``center=True`` 的约定裁掉两端补零。
        length: 指定输出长度（样本数）。
    """
    spectrum = np.asarray(spectrum)
    n_freqs, n_frames = spectrum.shape
    if n_fft is None:
        n_fft = 2 * (n_freqs - 1)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    window = _window(n_fft)
    frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1) * window

    total = n_fft + hop_length * (n_frames - 1)
    output = np.zeros(total, dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_length
        output[start : start + n_fft] += frames[i]
        weight[start : start + n_fft] += window**2

    nonzero = weight > 1e-10
    output[nonzero] /= weight[nonzero]

    if center:
        pad = n_fft // 2
        output = output[pad : max(pad, total - pad)]

    if length is not None:
        if output.size < length:
            output = np.pad(output, (0, length - output.size))
        else:
            output = output[:length]

    return output.astype(np.float32)
