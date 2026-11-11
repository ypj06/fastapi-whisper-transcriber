"""
Voice Activity Detection (VAD) module.

Detects speech vs. non-speech segments in audio, essential for:
- Removing silence/non-speech before ASR
- Segmenting continuous audio into speaker utterances
- Reducing computational load on downstream ASR models

Methods implemented:
1. Energy-based VAD (simple thresholding)
2. Silero VAD (deep learning based, state-of-the-art)
3. WebRTC VAD (lightweight, mobile-friendly)
4. Spectral entropy VAD
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import inspect

import numpy as np


@dataclass
class SpeechSegment:
    """A detected speech segment."""

    start: float  # Start time in seconds
    end: float    # End time in seconds
    audio: np.ndarray  # Audio samples for this segment
    confidence: float = 1.0  # Detection confidence


# =============================================================================
# Energy-based VAD
# =============================================================================

def energy_based_vad(
    audio: np.ndarray,
    sr: int,
    frame_duration: float = 0.03,
    energy_threshold: float = 0.01,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.2,
) -> List[SpeechSegment]:
    """Simple energy-based voice activity detection.

    Computes short-time energy per frame and applies thresholding.
    Works well in quiet environments but struggles with background noise.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        frame_duration: Frame length in seconds.
        energy_threshold: Frame energy threshold (0-1, relative to max).
        min_speech_duration: Minimum speech segment duration.
        min_silence_duration: Minimum silence/gap duration.

    Returns:
        List of detected SpeechSegments.
    """
    frame_length = int(frame_duration * sr)
    hop_length = frame_length // 2
    n_frames = (len(audio) - frame_length) // hop_length + 1

    # Compute frame energies
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        energies[i] = np.mean(frame ** 2)

    # Normalize energies
    if np.max(energies) > 1e-10:
        energies = energies / np.max(energies)

    # Detect speech frames
    is_speech = energies > energy_threshold

    # Merge and filter segments
    return _merge_speech_frames(
        is_speech,
        hop_length,
        sr,
        audio,
        min_speech_duration,
        min_silence_duration,
    )


# =============================================================================
# Silero VAD (Deep learning based)
# =============================================================================

def silero_vad(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.3,
) -> List[SpeechSegment]:
    """Voice activity detection using Silero VAD model.

    Silero VAD is a pre-trained neural network that provides
    state-of-the-art VAD performance. It's compact enough for
    local/edge deployment.

    Args:
        audio: Input audio samples (must be 16000 Hz).
        sr: Sample rate (should be 16000 for Silero).
        threshold: Speech probability threshold (0-1).
        min_speech_duration: Minimum speech segment duration.
        min_silence_duration: Minimum silence/gap duration.

    Returns:
        List of detected SpeechSegments.
    """
    try:
        import torch

        # Silero VAD expects 16000 Hz
        if sr != 16000:
            from scipy import signal as sp_signal
            audio = sp_signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000

        # Load model (automatically downloads on first use)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        (get_speech_timestamps, _, _, _, _) = utils

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=threshold,
            sampling_rate=sr,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000),
        )

        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start = ts["start"] / sr
            end = ts["end"] / sr
            segment_audio = audio[ts["start"] : ts["end"]]
            confidence = ts.get("confidence", threshold)
            segments.append(
                SpeechSegment(start, end, segment_audio, confidence)
            )

        return segments

    except ImportError:
        import warnings
        warnings.warn(
            "Silero VAD requires PyTorch. Falling back to energy-based VAD."
        )
        return energy_based_vad(
            audio, sr, min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )
    except Exception as e:
        import warnings
        warnings.warn(f"Silero VAD failed: {e}. Falling back to energy-based VAD.")
        return energy_based_vad(
            audio, sr, min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )


# =============================================================================
# WebRTC VAD (lightweight, mobile-friendly)
# =============================================================================

def webrtc_vad(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.2,
) -> List[SpeechSegment]:
    """Voice activity detection using WebRTC VAD.

    WebRTC VAD is extremely lightweight and designed for real-time
    communication. Ideal for mobile/local deployment scenarios.

    Args:
        audio: Input audio samples.
        sr: Sample rate (8000, 16000, 32000, or 48000).
        aggressiveness: 0 (least) to 3 (most aggressive filtering).
        frame_duration_ms: Frame size in ms (10, 20, or 30).
        min_speech_duration: Minimum speech segment duration.
        min_silence_duration: Minimum silence/gap duration.

    Returns:
        List of detected SpeechSegments.
    """
    try:
        import webrtcvad

        vad = webrtcvad.Vad(aggressiveness)

        # WebRTC VAD requires 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        frame_size = int(sr * frame_duration_ms / 1000)
        n_frames = len(audio_int16) // frame_size

        is_speech = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            start = i * frame_size
            frame = audio_int16[start : start + frame_size]
            try:
                is_speech[i] = vad.is_speech(frame.tobytes(), sr)
            except Exception:
                is_speech[i] = False

        return _merge_speech_frames(
            is_speech,
            frame_size,
            sr,
            audio,
            min_speech_duration,
            min_silence_duration,
        )

    except ImportError:
        import warnings
        warnings.warn(
            "webrtcvad not installed. Falling back to energy-based VAD."
        )
        return energy_based_vad(
            audio, sr, min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )


# =============================================================================
# Spectral Entropy VAD
# =============================================================================

def spectral_entropy_vad(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    entropy_threshold: float = 0.5,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.2,
) -> List[SpeechSegment]:
    """VAD based on spectral entropy.

    Speech has lower spectral entropy (more structured) than noise
    (more random). This method exploits that difference.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        entropy_threshold: Normalized entropy threshold (0-1).
        min_speech_duration: Minimum speech segment duration.
        min_silence_duration: Minimum silence/gap duration.

    Returns:
        List of detected SpeechSegments.
    """
    import librosa

    D = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Normalize each frame to get probability distribution
    n_freqs, n_frames = magnitude.shape
    entropy = np.zeros(n_frames)

    for t in range(n_frames):
        frame = magnitude[:, t]
        total = np.sum(frame)
        if total > 1e-10:
            # Normalize to probability distribution
            probs = frame / total
            # Compute entropy: -sum(p * log(p))
            non_zero = probs[probs > 0]
            entropy[t] = -np.sum(non_zero * np.log2(non_zero))
        else:
            entropy[t] = np.log2(n_freqs)  # Max entropy

    # Normalize entropy: 0 = pure tone, 1 = white noise
    max_entropy = np.log2(n_freqs)
    normalized_entropy = entropy / max_entropy

    # Low entropy = likely speech
    is_speech = normalized_entropy < entropy_threshold

    return _merge_speech_frames(
        is_speech,
        hop_length,
        sr,
        audio,
        min_speech_duration,
        min_silence_duration,
    )


# =============================================================================
# Helper functions
# =============================================================================

def _merge_speech_frames(
    is_speech: np.ndarray,
    hop_length: int,
    sr: int,
    audio: np.ndarray,
    min_speech_duration: float,
    min_silence_duration: float,
) -> List[SpeechSegment]:
    """Merge consecutive speech frames into segments, filtering by duration."""
    min_speech_frames = int(min_speech_duration * sr / hop_length)
    min_silence_frames = int(min_silence_duration * sr / hop_length)

    segments = []
    in_speech = False
    speech_start = 0
    silence_start = 0

    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            # Speech start
            if segments and (i - silence_start) < min_silence_frames:
                # Short gap, merge with previous
                in_speech = True
            else:
                speech_start = i
                in_speech = True
        elif not is_speech[i] and in_speech:
            # Potential speech end
            silence_start = i
            in_speech = False
        elif not is_speech[i] and not in_speech:
            # Continue silence
            pass
        elif is_speech[i] and in_speech:
            # Continue speech
            pass

    # Handle final segment
    if in_speech:
        end_frame = len(is_speech)

    # Filter by minimum speech duration
    if in_speech:
        end_frame = len(is_speech)
        duration = (end_frame - speech_start) * hop_length / sr
        if duration >= min_speech_duration:
            start_sample = speech_start * hop_length
            end_sample = min(end_frame * hop_length, len(audio))
            start_time = start_sample / sr
            end_time = end_sample / sr
            segments.append(
                SpeechSegment(start_time, end_time, audio[start_sample:end_sample])
            )

    # If we ended in silence, close the last segment
    else:
        if "speech_start" in locals():
            end_frame = silence_start
            duration = (end_frame - speech_start) * hop_length / sr
            if duration >= min_speech_duration:
                start_sample = speech_start * hop_length
                end_sample = min(end_frame * hop_length, len(audio))
                start_time = start_sample / sr
                end_time = end_sample / sr
                segments.append(
                    SpeechSegment(start_time, end_time, audio[start_sample:end_sample])
                )

    return segments


def get_voice_activity_mask(
    segments: List[SpeechSegment],
    audio_length: int,
    sr: int,
) -> np.ndarray:
    """Create a boolean mask indicating speech regions.

    Args:
        segments: List of detected SpeechSegments.
        audio_length: Total number of audio samples.
        sr: Sample rate.

    Returns:
        Boolean array of length audio_length.
    """
    mask = np.zeros(audio_length, dtype=bool)
    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        mask[start_sample:end_sample] = True
    return mask


def extract_speech(
    audio: np.ndarray,
    segments: List[SpeechSegment],
    padding: float = 0.05,
) -> np.ndarray:
    """Extract only speech portions from audio.

    Args:
        audio: Full audio.
        segments: Detected speech segments.
        padding: Padding to add around each segment in seconds.

    Returns:
        Concatenated speech-only audio.
    """
    if not segments:
        return np.array([], dtype=audio.dtype)

    sr = int(len(audio) / (segments[-1].end if segments else 1))
    speech_parts = []
    for seg in segments:
        start = max(0, int((seg.start - padding) * sr))
        end = min(len(audio), int((seg.end + padding) * sr))
        speech_parts.append(audio[start:end])

    return np.concatenate(speech_parts) if speech_parts else np.array([])


# =============================================================================
# Unified VAD interface
# =============================================================================

def detect_voice_activity(
    audio: np.ndarray,
    sr: int,
    method: str = "silero",
    **kwargs,
) -> List[SpeechSegment]:
    """Unified interface for voice activity detection.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        method: One of 'energy', 'silero', 'webrtc', 'spectral_entropy'.
        **kwargs: Method-specific parameters.

    Returns:
        List of detected SpeechSegments.

    Raises:
        ValueError: If method is unknown.
    """
    methods = {
        "energy": energy_based_vad,
        "silero": silero_vad,
        "webrtc": webrtc_vad,
        "spectral_entropy": spectral_entropy_vad,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown VAD method '{method}'. Choose from: {list(methods.keys())}"
        )

    # Only pass kwargs the target function actually accepts
    fn = methods[method]
    sig = inspect.signature(fn)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return fn(audio, sr, **filtered_kwargs)
