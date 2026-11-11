"""
Noise Reduction module for audio preprocessing.

Implements multiple noise reduction techniques:
1. Spectral Gating (via noisereduce)
2. Spectral Subtraction (classic DSP approach)
3. Wiener Filtering
4. Kalman-based adaptive filtering
5. Subspace-based denoising

Each method is tuned for ASR-friendly output that preserves speech
intelligibility while suppressing background noise.
"""

import inspect
from typing import Optional

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz


# =============================================================================
# Spectral Gating (using noisereduce library)
# =============================================================================

def spectral_gating_noise_reduction(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    prop_decrease: float = 0.9,
    n_std_thresh: float = 1.5,
    use_torch: bool = False,
) -> np.ndarray:
    """Apply spectral gating noise reduction.

    Uses the noisereduce library which implements a noise gate in the
    frequency domain. Non-speech frames are estimated and used to
    create a spectral mask.

    Args:
        audio: Noisy audio input.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        prop_decrease: Proportion of noise reduction (0-1).
        n_std_thresh: Number of standard deviations above noise floor.
        use_torch: Use PyTorch backend if available.

    Returns:
        Denoised audio.
    """
    try:
        import noisereduce as nr

        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh,
            use_torch=use_torch,
            stationary=False,
        )
        return reduced.astype(np.float32)
    except ImportError:
        import warnings
        warnings.warn(
            "noisereduce not installed. Falling back to spectral subtraction."
        )
        return spectral_subtraction(audio, sr, n_fft, hop_length)


# =============================================================================
# Classic Spectral Subtraction (Boll, 1979)
# =============================================================================

def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_frames: int = 10,
    over_subtraction: float = 1.0,
    floor_db: float = -40.0,
) -> np.ndarray:
    """Classic spectral subtraction for noise reduction.

    Estimates noise spectrum from initial silent frames and subtracts
    it from the noisy signal's magnitude spectrum. A spectral floor
    prevents negative magnitude values.

    Algorithm:
    1. Compute STFT of noisy signal
    2. Estimate noise spectrum from first `noise_frames` frames
    3. Subtract noise magnitude from signal magnitude
    4. Apply spectral floor
    5. Reconstruct signal via inverse STFT

    Args:
        audio: Noisy audio input.
        sr: Sample rate (for floor calculation only).
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        noise_frames: Number of initial frames to use as noise estimate.
        over_subtraction: Multiplier for noise subtraction (>1 reduces more).
        floor_db: Spectral floor in dB below max.

    Returns:
        Denoised audio.
    """
    # Compute STFT
    D = librosa_stft(audio, n_fft, hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Estimate noise spectrum from initial frames
    "noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)"
    noise_estimate = _get_min_energy_noise_profile(magnitude, noise_frames) 

    # Spectral subtraction with oversubtraction
    subtracted = magnitude - over_subtraction * noise_estimate

    # Apply spectral floor
    max_mag = np.max(subtracted)
    floor = 10 ** (floor_db / 20) * max_mag
    subtracted = np.maximum(subtracted, floor)

    # Reconstruct
    D_clean = subtracted * np.exp(1j * phase)
    return librosa_istft(D_clean, hop_length)


# =============================================================================
# Wiener Filter
# =============================================================================

def wiener_filter(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_frames: int = 10,
    smoothing: float = 0.98,
) -> np.ndarray:
    """Wiener filter for noise reduction.

    Implements a frequency-domain Wiener filter that estimates the
    clean signal spectrum using a priori SNR estimation. Uses
    decision-directed approach for SNR estimation.

    Args:
        audio: Noisy audio input.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        noise_frames: Number of initial frames for noise estimation.
        smoothing: Smoothing factor for SNR estimation.

    Returns:
        Filtered audio.
    """
    D = librosa_stft(audio, n_fft, hop_length)
    magnitude = np.abs(D)
    power = magnitude ** 2
    phase = np.angle(D)

    # Noise power spectrum estimate
    "noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)"
    noise_power = _get_min_energy_noise_profile(power, noise_frames)

    # Decision-directed a priori SNR estimation
    n_freqs, n_frames = power.shape
    xi = np.zeros_like(power)  # A priori SNR
    gamma = np.zeros_like(power)  # A posteriori SNR

    for t in range(n_frames):
        gamma[:, t : t + 1] = (
            np.maximum(power[:, t : t + 1] / (noise_power + 1e-10), 1e-10) - 1
        )

        if t == 0:
            xi[:, t : t + 1] = np.maximum(gamma[:, t : t + 1], 0)
        else:
            prev_mag = magnitude[:, t - 1 : t]
            xi[:, t : t + 1] = (
                smoothing * (prev_mag ** 2) / (noise_power + 1e-10)
                + (1 - smoothing) * np.maximum(gamma[:, t : t + 1], 0)
            )

        # Ensure non-negative
        xi[:, t : t + 1] = np.maximum(xi[:, t : t + 1], 1e-10)

    # Wiener gain
    gain = xi / (xi + 1)
    D_filtered = gain * magnitude * np.exp(1j * phase)

    return librosa_istft(D_filtered, hop_length)


# =============================================================================
# Kalman-based Adaptive Filtering
# =============================================================================

def kalman_denoise(
    audio: np.ndarray,
    sr: int,
    Q: float = 0.001,
    R: float = 0.1,
) -> np.ndarray:
    """Simple Kalman filter for audio denoising.

    Models the clean audio as a first-order AR process and applies
    scalar Kalman filtering sample by sample.

    Args:
        audio: Noisy audio input.
        sr: Sample rate (unused, for API consistency).
        Q: Process noise covariance (lower = more smoothing).
        R: Measurement noise covariance (higher = less trust in measurement).

    Returns:
        Filtered audio.
    """
    n = len(audio)
    x_hat = np.zeros(n)
    P = 1.0  # Initial error covariance

    # First-order AR coefficient (slowly varying signal)
    A = 0.99

    x_hat[0] = audio[0]

    for k in range(1, n):
        # Prediction
        x_pred = A * x_hat[k - 1]
        P_pred = A * P * A + Q

        # Kalman gain
        K = P_pred / (P_pred + R)

        # Update
        x_hat[k] = x_pred + K * (audio[k] - x_pred)
        P = (1 - K) * P_pred

    return x_hat.astype(np.float32)


# =============================================================================
# Multi-band Spectral Subtraction (improved)
# =============================================================================

def multi_band_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_frames: int = 10,
    freq_bands: Optional[list] = None,
) -> np.ndarray:
    """Multi-band spectral subtraction with band-specific oversubtraction.

    Different frequency bands get different subtraction factors:
    - Low frequencies (< 1kHz): less subtraction (preserves speech fundamentals)
    - Mid frequencies (1-4 kHz): moderate subtraction (speech formants)
    - High frequencies (> 4kHz): more aggressive subtraction (mostly noise)

    Args:
        audio: Noisy audio input.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        noise_frames: Number of initial frames for noise estimate.
        freq_bands: List of (low_hz, high_hz, oversubtraction_factor) tuples.
                    If None, uses default bands tailored for speech.

    Returns:
        Denoised audio.
    """
    if freq_bands is None:
        # Default bands tailored for speech frequencies
        freq_bands = [
            (0, 500, 0.8),     # Sub-bass / fundamental: gentle
            (500, 2000, 1.0),  # Speech formants: standard
            (2000, 4000, 1.3), # Higher formants / fricatives: moderate
            (4000, sr // 2, 1.6),  # High frequencies: aggressive
        ]

    D = librosa_stft(audio, n_fft, hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)[: magnitude.shape[0]]

    "noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)"
    noise_magnitude = _get_min_energy_noise_profile(magnitude, noise_frames)

    # Create band mask and apply different subtraction per band
    subtracted = magnitude.copy()
    for low_hz, high_hz, over_sub in freq_bands:
        band_mask = (freqs >= low_hz) & (freqs < high_hz)
        band_mask = band_mask[:, np.newaxis]
        subtracted = np.where(
            band_mask,
            np.maximum(
                magnitude - over_sub * noise_magnitude,
                0.001 * magnitude,  # Minimum floor
            ),
            subtracted,
        )

    D_clean = subtracted * np.exp(1j * phase)
    return librosa_istft(D_clean, hop_length)


# =============================================================================
# Helper STFT/ISTFT wrappers
# =============================================================================

def librosa_stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """Compute STFT using librosa."""
    import librosa

    return librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)


def librosa_istft(
    D: np.ndarray,
    hop_length: int = 128,
) -> np.ndarray:
    """Compute inverse STFT using librosa."""
    import librosa

    return librosa.istft(D, hop_length=hop_length)


# =============================================================================
# Unified noise reduction interface
# =============================================================================

def reduce_noise(
    audio: np.ndarray,
    sr: int,
    method: str = "spectral_gating",
    **kwargs,
) -> np.ndarray:
    """Unified interface for noise reduction.

    Args:
        audio: Noisy audio input.
        sr: Sample rate.
        method: One of 'spectral_gating', 'spectral_subtraction',
                'wiener', 'kalman', 'multi_band'.
        **kwargs: Method-specific parameters.

    Returns:
        Denoised audio.

    Raises:
        ValueError: If method is unknown.
    """
    methods = {
        "spectral_gating": spectral_gating_noise_reduction,
        "spectral_subtraction": spectral_subtraction,
        "wiener": wiener_filter,
        "kalman": kalman_denoise,
        "multi_band": multi_band_spectral_subtraction,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(methods.keys())}"
        )

    # Only pass kwargs the target function actually accepts
    fn = methods[method]
    sig = inspect.signature(fn)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return fn(audio, sr, **filtered_kwargs)

def _get_min_energy_noise_profile(spectrogram: np.ndarray, noise_frames: int) -> np.ndarray:
    """
    遍历整个频谱图，寻找能量最小（最安静）的连续 noise_frames 帧，
    并返回这段区域的平均频谱作为噪声模板。
    """
    n_frames = spectrogram.shape[1]
    
    # 如果音频总长度比需要的噪声帧还要短，只能直接取全部平均
    if n_frames <= noise_frames:
        return np.mean(spectrogram, axis=1, keepdims=True)

    # 1. 计算每一帧的总能量 (沿着频率轴求和)
    frame_energies = np.sum(spectrogram, axis=0)

    # 2. 使用滑动窗口计算连续 noise_frames 帧的能量总和
    window = np.ones(noise_frames)
    sliding_energy = np.convolve(frame_energies, window, mode='valid')

    # 3. 找到能量最小的窗口的起始索引
    min_energy_idx = np.argmin(sliding_energy)

    # 4. 提取最安静的窗口，并计算其平均频谱
    quietest_spectrogram = spectrogram[:, min_energy_idx : min_energy_idx + noise_frames]
    
    return np.mean(quietest_spectrogram, axis=1, keepdims=True)