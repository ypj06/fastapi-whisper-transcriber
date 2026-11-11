"""
Signal Enhancement module for audio preprocessing.

Improves speech quality through:
1. Dynamic Range Compression (DRC)
2. Equalization (EQ) - emphasizing speech-critical frequencies
3. Automatic Gain Control (AGC)
4. De-essing (sibilance reduction)
5. Dereverberation
6. Pitch-aware enhancement

All methods aim to make speech clearer and more consistent for
downstream ASR processing, particularly in challenging acoustic
environments.
"""

from typing import Optional

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz


# =============================================================================
# Dynamic Range Compression
# =============================================================================

def dynamic_range_compression(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    """Apply dynamic range compression to audio.

    Reduces the dynamic range by attenuating loud parts above the
    threshold. This makes quiet speech more audible and prevents
    clipping during ASR processing.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        threshold_db: Threshold in dB above which compression is applied.
        ratio: Compression ratio (e.g., 4:1 means 4dB in = 1dB out above threshold).
        attack_ms: Attack time in ms (how fast compression kicks in).
        release_ms: Release time in ms (how fast compression releases).
        makeup_gain_db: Makeup gain in dB applied after compression.

    Returns:
        Compressed audio.
    """
    # Compute envelope (RMS over short windows)
    window_ms = 5.0
    window_samples = int(window_ms * sr / 1000)
    hop_samples = window_samples // 4

    n_frames = (len(audio) - window_samples) // hop_samples + 1
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        frame = audio[start : start + window_samples]
        envelope[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

    envelope_db = 20 * np.log10(envelope + 1e-10)

    # Compression gain curve
    gain_db = np.zeros_like(envelope_db)
    attack_samples = int(attack_ms * sr / 1000 / hop_samples)
    release_samples = int(release_ms * sr / 1000 / hop_samples)

    target_gain = np.zeros_like(envelope_db)
    above_threshold = envelope_db > threshold_db
    target_gain[above_threshold] = (
        (threshold_db - envelope_db[above_threshold]) * (1 - 1 / ratio)
    )

    # Smooth gain changes (attack/release)
    for i in range(1, n_frames):
        if target_gain[i] < gain_db[i - 1]:
            # Attack: gain decreasing
            alpha = np.exp(-1 / max(attack_samples, 1))
        else:
            # Release: gain increasing
            alpha = np.exp(-1 / max(release_samples, 1))
        gain_db[i] = alpha * gain_db[i - 1] + (1 - alpha) * target_gain[i]

    # Apply gain
    gain_linear = 10 ** ((gain_db + makeup_gain_db) / 20)

    output = np.zeros_like(audio)
    for i in range(n_frames):
        start = i * hop_samples
        end = min(start + window_samples, len(audio))
        seg_len = end - start
        output[start:end] += gain_linear[i] * audio[start:end]

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0.99:
        output = output / max_val * 0.99

    return output.astype(np.float32)


# =============================================================================
# Parametric Equalizer for Speech Enhancement
# =============================================================================

def speech_equalizer(
    audio: np.ndarray,
    sr: int,
    preset: str = "telephone",
) -> np.ndarray:
    """Apply equalization optimized for speech intelligibility.

    Different presets for different scenarios:
    - 'telephone': Emphasizes 300-3400 Hz (telephone band)
    - 'clarity': Boosts 2-5 kHz for consonant clarity
    - 'warmth': Slight bass boost + clarity
    - 'flat': No EQ, just reference

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        preset: EQ preset name.

    Returns:
        Equalized audio.
    """
    presets = {
        "telephone": [
            ("highpass", 300, 0.7),   # Cut below 300 Hz
            ("lowpass", 3400, 0.7),   # Cut above 3400 Hz
            ("peaking", 1000, 3.0, 1.5),  # Boost mid-range
        ],
        "clarity": [
            ("highpass", 80, 0.7),    # Gentle rumble removal
            ("peaking", 3000, 4.0, 2.0),  # Boost speech formants
            ("peaking", 6000, 3.0, 2.0),  # Boost fricatives
            ("highshelf", 8000, 2.0),  # Air band boost
        ],
        "warmth": [
            ("lowshelf", 200, 2.0),   # Gentle bass boost
            ("peaking", 3000, 3.0, 1.5),  # Clarity boost
            ("highshelf", 8000, 2.0),  # Air
        ],
        "flat": [],
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

    return _apply_eq_filters(audio, sr, presets[preset])


def _apply_eq_filters(
    audio: np.ndarray,
    sr: int,
    filters: list,
) -> np.ndarray:
    """Apply a list of EQ filter specifications to audio."""
    output = audio.copy().astype(np.float64)

    for filt in filters:
        filt_type = filt[0]

        if filt_type == "highpass":
            _, freq, order = filt
            b, a = _design_butter(filt_type, freq, sr, order)
        elif filt_type == "lowpass":
            _, freq, order = filt
            b, a = _design_butter(filt_type, freq, sr, order)
        elif filt_type == "peaking":
            _, freq, gain_db, q = filt
            b, a = _design_peaking(freq, gain_db, q, sr)
        elif filt_type == "lowshelf":
            _, freq, gain_db = filt
            b, a = _design_shelf("low", freq, gain_db, sr)
        elif filt_type == "highshelf":
            _, freq, gain_db = filt
            b, a = _design_shelf("high", freq, gain_db, sr)
        else:
            continue

        output = lfilter(b, a, output)

    return output.astype(np.float32)


def _design_butter(
    filt_type: str,
    cutoff: float,
    sr: int,
    order: float,
) -> tuple:
    """Design a Butterworth filter."""
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(int(order), normalized_cutoff, btype=filt_type)
    return b, a


def _design_peaking(
    freq: float,
    gain_db: float,
    q: float,
    sr: int,
) -> tuple:
    """Design a peaking (bell) EQ filter."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    A = 10 ** (gain_db / 40)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


def _design_shelf(
    shelf_type: str,
    freq: float,
    gain_db: float,
    sr: int,
) -> tuple:
    """Design a shelving EQ filter."""
    w0 = 2 * np.pi * freq / sr
    S = 1.0  # Slope parameter
    A = 10 ** (gain_db / 40)

    if shelf_type == "low":
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * np.sin(w0))
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * np.sin(w0))
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * np.sin(w0)
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * np.sin(w0)
    else:
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * np.sin(w0))
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * np.sin(w0))
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * np.sin(w0)
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * np.sin(w0)

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


# =============================================================================
# Automatic Gain Control (AGC)
# =============================================================================

def automatic_gain_control(
    audio: np.ndarray,
    sr: int,
    target_level_db: float = -23.0,
    window_ms: float = 100.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """Apply automatic gain control.

    Normalizes audio to a consistent level, compensating for
    varying microphone distances and recording levels. Critical
    for consistent ASR performance.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        target_level_db: Target RMS level in dB.
        window_ms: Analysis window in ms.
        max_gain_db: Maximum gain to apply in dB.

    Returns:
        Level-normalized audio.
    """
    window_samples = int(window_ms * sr / 1000)
    hop_samples = window_samples // 4

    n_frames = (len(audio) - window_samples) // hop_samples + 1
    output = np.zeros_like(audio)
    weights = np.zeros_like(audio)

    target_linear = 10 ** (target_level_db / 20)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        frame = audio[start:end]

        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        gain = np.clip(target_linear / rms, 0, 10 ** (max_gain_db / 20))

        # Apply with overlap-add
        output[start:end] += gain * frame
        weights[start:end] += 1

    # Normalize by overlap count
    weights = np.maximum(weights, 1)
    output = output / weights

    # Final peak normalization
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output / peak * 0.99

    return output.astype(np.float32)


# =============================================================================
# De-essing (Sibilance Reduction)
# =============================================================================

def de_ess(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
    center_freq: float = 6000.0,
    bandwidth: float = 2000.0,
    ratio: float = 3.0,
) -> np.ndarray:
    """Apply de-essing to reduce harsh sibilant sounds.

    'S', 'sh', 'ch' sounds can cause ASR confusion. This reduces
    energy in the sibilance frequency range (typically 4-10 kHz).

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        threshold_db: Threshold above which de-essing is applied.
        center_freq: Center frequency of sibilance band in Hz.
        bandwidth: Bandwidth of sibilance band in Hz.
        ratio: Compression ratio for sibilant frequencies.

    Returns:
        De-essed audio.
    """
    # Bandpass filter for sibilance detection
    low_freq = max(center_freq - bandwidth / 2, 100)
    high_freq = min(center_freq + bandwidth / 2, sr / 2 - 100)

    b, a = butter(4, [low_freq / (sr / 2), high_freq / (sr / 2)], btype="band")
    sibilance_signal = lfilter(b, a, audio)

    # Detect sibilance based on energy in band
    window_samples = int(0.01 * sr)  # 10ms windows
    hop_samples = window_samples // 2

    n_frames = (len(audio) - window_samples) // hop_samples + 1
    gain = np.ones(n_frames)

    for i in range(n_frames):
        start = i * hop_samples
        frame = sibilance_signal[start : start + window_samples]
        level_db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-10)

        if level_db > threshold_db:
            reduction_db = (level_db - threshold_db) * (1 - 1 / ratio)
            gain[i] = 10 ** (-reduction_db / 20)

    # Apply gain with overlap-add
    output = np.zeros_like(audio)
    weights = np.zeros_like(audio)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        output[start:end] += gain[i] * audio[start:end]
        weights[start:end] += 1

    weights = np.maximum(weights, 1)
    return (output / weights).astype(np.float32)


# =============================================================================
# Dereverberation
# =============================================================================

def dereverberation(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """Simple spectral subtraction-based dereverberation.

    Late reverberation is treated as additive noise and removed
    using spectral subtraction in the modulation frequency domain.
    This is a simplified version suitable for mild reverberation.

    For more severe reverb, consider using WPE (Weighted Prediction
    Error) dereverberation.

    Args:
        audio: Reverberant audio.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.

    Returns:
        Dereverberated audio.
    """
    import librosa

    # Compute STFT
    D = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Estimate late reverberation as smoothed version of magnitude
    # (late reverb is diffuse and slowly varying)
    n_freqs, n_frames = magnitude.shape
    smooth_magnitude = np.zeros_like(magnitude)

    smoothing_time = 0.1  # 100ms smoothing for late reverb estimate
    alpha = np.exp(-hop_length / sr / smoothing_time)

    for f in range(n_freqs):
        smooth_magnitude[f, 0] = magnitude[f, 0]
        for t in range(1, n_frames):
            smooth_magnitude[f, t] = (
                alpha * smooth_magnitude[f, t - 1]
                + (1 - alpha) * magnitude[f, t]
            )

    # Use a portion of the smoothed magnitude as reverb estimate
    reverb_estimate = 0.3 * smooth_magnitude

    # Spectral subtraction
    clean_mag = np.maximum(magnitude - reverb_estimate, 0.001 * magnitude)

    # Reconstruct
    D_clean = clean_mag * np.exp(1j * phase)
    return librosa.istft(D_clean, hop_length=hop_length)


# =============================================================================
# Full signal enhancement pipeline
# =============================================================================

def enhance_signal(
    audio: np.ndarray,
    sr: int,
    enable_drc: bool = True,
    enable_eq: bool = True,
    eq_preset: str = "telephone",
    enable_agc: bool = True,
    enable_deess: bool = True,
    enable_dereverb: bool = False,
) -> np.ndarray:
    """Apply a comprehensive signal enhancement pipeline.

    The pipeline is ordered for optimal results:
    1. Dereverberation (if needed) - clean up room acoustics first
    2. De-essing - reduce harsh sibilance
    3. EQ - shape frequency response for speech
    4. DRC - compress dynamic range
    5. AGC - normalize overall level

    Args:
        audio: Input audio.
        sr: Sample rate.
        enable_drc: Enable dynamic range compression.
        enable_eq: Enable equalization.
        eq_preset: EQ preset ('telephone', 'clarity', 'warmth', 'flat').
        enable_agc: Enable automatic gain control.
        enable_deess: Enable de-essing.
        enable_dereverb: Enable dereverberation.

    Returns:
        Enhanced audio.
    """
    output = audio.copy().astype(np.float32)

    if enable_dereverb:
        output = dereverberation(output, sr)

    if enable_deess:
        output = de_ess(output, sr)

    if enable_eq:
        output = speech_equalizer(output, sr, preset=eq_preset)

    if enable_drc:
        output = dynamic_range_compression(output, sr)

    if enable_agc:
        output = automatic_gain_control(output, sr)

    return output
