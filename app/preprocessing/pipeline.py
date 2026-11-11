"""
Main Audio Preprocessing Pipeline for ASR.

Coordinates all preprocessing stages in a configurable pipeline:
1. Voice Activity Detection → extract speech segments
2. Noise Reduction → suppress background noise
3. Signal Enhancement → improve speech clarity and consistency

Designed for local/edge deployment with configurable trade-offs
between quality and computational cost.
"""

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .noise_reduction import reduce_noise
from .signal_enhancement import enhance_signal
from .voice_activity_detection import detect_voice_activity, extract_speech


class VadMethod(str, Enum):
    """Available VAD methods."""
    SILERO = "silero"
    WEBRTC = "webrtc"
    ENERGY = "energy"
    SPECTRAL_ENTROPY = "spectral_entropy"
    NONE = "none"


class NoiseReductionMethod(str, Enum):
    """Available noise reduction methods."""
    SPECTRAL_GATING = "spectral_gating"
    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER = "wiener"
    KALMAN = "kalman"
    MULTI_BAND = "multi_band"
    NONE = "none"


@dataclass
class PreprocessingConfig:
    """Configuration for the audio preprocessing pipeline.

    All parameters have sensible defaults tuned for ASR performance.
    Adjust based on your deployment environment and quality requirements.
    """

    # Voice Activity Detection
    enable_vad: bool = True
    vad_method: VadMethod = VadMethod.SPECTRAL_ENTROPY
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.1
    vad_min_silence_duration: float = 0.3

    # Noise Reduction
    enable_noise_reduction: bool = True
    noise_reduction_method: NoiseReductionMethod = NoiseReductionMethod.SPECTRAL_GATING
    noise_reduction_prop_decrease: float = 0.85
    noise_reduction_n_fft: int = 512
    noise_reduction_hop_length: int = 128

    # Signal Enhancement
    enable_signal_enhancement: bool = True
    enable_drc: bool = True
    enable_eq: bool = True
    eq_preset: str = "telephone"
    enable_agc: bool = True
    enable_deess: bool = False
    enable_dereverb: bool = False

    # Audio format
    target_sr: int = 16000  # 16kHz is standard for ASR

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "enable_vad": self.enable_vad,
            "vad_method": self.vad_method.value,
            "vad_threshold": self.vad_threshold,
            "enable_noise_reduction": self.enable_noise_reduction,
            "noise_reduction_method": self.noise_reduction_method.value,
            "enable_signal_enhancement": self.enable_signal_enhancement,
            "enable_drc": self.enable_drc,
            "enable_eq": self.enable_eq,
            "eq_preset": self.eq_preset,
            "enable_agc": self.enable_agc,
            "enable_deess": self.enable_deess,
            "enable_dereverb": self.enable_dereverb,
            "target_sr": self.target_sr,
        }


# =============================================================================
# Preset configurations for common scenarios
# =============================================================================

PRESETS = {
    "mobile": PreprocessingConfig(
        # Lightweight config for mobile devices
        enable_vad=True,
        vad_method=VadMethod.WEBRTC,
        enable_noise_reduction=True,
        noise_reduction_method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
        enable_signal_enhancement=True,
        enable_drc=True,
        enable_eq=True,
        eq_preset="telephone",
        enable_agc=True,
        enable_deess=False,
        enable_dereverb=False,
    ),
    "desktop_high_quality": PreprocessingConfig(
        # High quality for desktop/server
        enable_vad=True,
        vad_method=VadMethod.SPECTRAL_ENTROPY,
        enable_noise_reduction=True,
        noise_reduction_method=NoiseReductionMethod.SPECTRAL_GATING,
        enable_signal_enhancement=True,
        enable_drc=True,
        enable_eq=True,
        eq_preset="clarity",
        enable_agc=True,
        enable_deess=True,
        enable_dereverb=False,
    ),
    "noisy_environment": PreprocessingConfig(
        # Aggressive preprocessing for noisy environments
        enable_vad=True,
        vad_method=VadMethod.SPECTRAL_ENTROPY,
        vad_threshold=0.3,  # More sensitive in noise
        enable_noise_reduction=True,
        noise_reduction_method=NoiseReductionMethod.MULTI_BAND,
        noise_reduction_prop_decrease=0.95,
        enable_signal_enhancement=True,
        enable_drc=True,
        enable_eq=True,
        eq_preset="telephone",
        enable_agc=True,
        enable_deess=False,
        enable_dereverb=False,
    ),
    "meeting_room": PreprocessingConfig(
        # Optimized for multi-speaker meeting scenarios
        enable_vad=True,
        vad_method=VadMethod.SPECTRAL_ENTROPY,
        vad_threshold=0.4,
        enable_noise_reduction=True,
        noise_reduction_method=NoiseReductionMethod.WIENER,
        enable_signal_enhancement=True,
        enable_drc=True,
        enable_eq=True,
        eq_preset="clarity",
        enable_agc=True,
        enable_deess=True,
        enable_dereverb=True,
    ),
    "lightweight": PreprocessingConfig(
        # Minimal processing for speed
        enable_vad=False,
        enable_noise_reduction=True,
        noise_reduction_method=NoiseReductionMethod.KALMAN,
        enable_signal_enhancement=True,
        enable_drc=False,
        enable_eq=False,
        enable_agc=True,
        enable_deess=False,
        enable_dereverb=False,
    ),
}


# =============================================================================
# Processing statistics
# =============================================================================

@dataclass
class ProcessingStats:
    """Statistics collected during preprocessing."""

    input_duration: float = 0.0
    output_duration: float = 0.0
    vad_segments_detected: int = 0
    vad_speech_ratio: float = 0.0
    processing_time_total: float = 0.0
    processing_time_vad: float = 0.0
    processing_time_noise_reduction: float = 0.0
    processing_time_enhancement: float = 0.0
    input_rms_db: float = 0.0
    output_rms_db: float = 0.0
    noise_reduction_applied: bool = False
    signal_enhancement_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_duration_s": round(self.input_duration, 3),
            "output_duration_s": round(self.output_duration, 3),
            "duration_reduction_pct": round(
                (1 - self.output_duration / max(self.input_duration, 0.001)) * 100, 1
            ),
            "vad_segments": self.vad_segments_detected,
            "vad_speech_ratio": round(self.vad_speech_ratio, 3),
            "processing_time_total_ms": round(self.processing_time_total * 1000, 1),
            "processing_time_vad_ms": round(self.processing_time_vad * 1000, 1),
            "processing_time_nr_ms": round(self.processing_time_noise_reduction * 1000, 1),
            "processing_time_enhancement_ms": round(self.processing_time_enhancement * 1000, 1),
            "input_rms_db": round(self.input_rms_db, 1),
            "output_rms_db": round(self.output_rms_db, 1),
            "noise_reduction_applied": self.noise_reduction_applied,
            "signal_enhancement_applied": self.signal_enhancement_applied,
        }


# =============================================================================
# Main Pipeline
# =============================================================================

class AudioPreprocessingPipeline:
    """Complete audio preprocessing pipeline for ASR.

    The pipeline applies VAD, noise reduction, and signal enhancement
    in sequence. Each stage can be independently configured or disabled.

    Usage:
        pipeline = AudioPreprocessingPipeline(config)
        processed_audio, stats = pipeline.process(audio, sr)

        # Or use a preset:
        pipeline = AudioPreprocessingPipeline.from_preset("mobile")
        processed, stats = pipeline.process(audio, sr)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the pipeline.

        Args:
            config: PreprocessingConfig. Uses default if None.
        """
        self.config = config or PreprocessingConfig()

    @classmethod
    def from_preset(cls, preset_name: str) -> "AudioPreprocessingPipeline":
        """Create a pipeline from a named preset.

        Args:
            preset_name: One of 'mobile', 'desktop_high_quality',
                         'noisy_environment', 'meeting_room', 'lightweight'.

        Returns:
            Configured AudioPreprocessingPipeline.

        Raises:
            ValueError: If preset_name is unknown.
        """
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(PRESETS.keys())}"
            )
        return cls(PRESETS[preset_name])

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        return_stats: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ProcessingStats]]:
        """Run the full preprocessing pipeline.

        Args:
            audio: Input audio samples.
            sr: Sample rate.
            return_stats: If True, also return ProcessingStats.

        Returns:
            Processed audio, and optionally ProcessingStats.
        """
        import time

        stats = ProcessingStats()
        stats.input_duration = len(audio) / sr
        stats.input_rms_db = float(
            20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
        )

        output = audio.copy().astype(np.float32)
        t_start = time.time()

        # Resample if needed
        if sr != self.config.target_sr:
            from scipy import signal as sp_signal
            output = sp_signal.resample(
                output, int(len(output) * self.config.target_sr / sr)
            )
            sr = self.config.target_sr

        # Stage 1: Voice Activity Detection
        if self.config.enable_vad:
            t_vad = time.time()
            segments = detect_voice_activity(
                output,
                sr,
                method=self.config.vad_method.value,
                threshold=self.config.vad_threshold,
                min_speech_duration=self.config.vad_min_speech_duration,
                min_silence_duration=self.config.vad_min_silence_duration,
            )
            stats.vad_segments_detected = len(segments)

            if segments:
                total_speech = sum(seg.end - seg.start for seg in segments)
                stats.vad_speech_ratio = total_speech / stats.input_duration
                output = extract_speech(output, segments)
            else:
                # No speech detected, return as-is
                stats.vad_speech_ratio = 0.0

            stats.processing_time_vad = time.time() - t_vad

        # Stage 2: Noise Reduction
        if self.config.enable_noise_reduction:
            t_nr = time.time()
            if len(output) > 0:
                output = reduce_noise(
                    output,
                    sr,
                    method=self.config.noise_reduction_method.value,
                    n_fft=self.config.noise_reduction_n_fft,
                    hop_length=self.config.noise_reduction_hop_length,
                    prop_decrease=self.config.noise_reduction_prop_decrease,
                )
                stats.noise_reduction_applied = True
            stats.processing_time_noise_reduction = time.time() - t_nr

        # Stage 3: Signal Enhancement
        if self.config.enable_signal_enhancement:
            t_enh = time.time()
            if len(output) > 0:
                output = enhance_signal(
                    output,
                    sr,
                    enable_drc=self.config.enable_drc,
                    enable_eq=self.config.enable_eq,
                    eq_preset=self.config.eq_preset,
                    enable_agc=self.config.enable_agc,
                    enable_deess=self.config.enable_deess,
                    enable_dereverb=self.config.enable_dereverb,
                )
                stats.signal_enhancement_applied = True
            stats.processing_time_enhancement = time.time() - t_enh

        stats.output_duration = len(output) / sr if len(output) > 0 else 0
        stats.output_rms_db = float(
            20 * np.log10(np.sqrt(np.mean(output ** 2)) + 1e-10)
        ) if len(output) > 0 else -100
        stats.processing_time_total = time.time() - t_start

        if return_stats:
            return output, stats
        return output

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, ProcessingStats]:
        """Process an audio file end-to-end.

        Args:
            input_path: Path to input audio file.
            output_path: Optional path to save processed audio.

        Returns:
            Tuple of (processed_audio, stats).
        """
        from .audio_utils import load_audio, save_audio

        audio, sr = load_audio(str(input_path), target_sr=self.config.target_sr)
        processed, stats = self.process(audio, sr)

        if output_path:
            save_audio(processed, str(output_path), sr)

        return processed, stats


# =============================================================================
# Convenience function
# =============================================================================

def preprocess_for_asr(
    audio: np.ndarray,
    sr: int,
    preset: str = "desktop_high_quality",
    **overrides,
) -> Tuple[np.ndarray, ProcessingStats]:
    """One-shot convenience function to preprocess audio for ASR.

    Args:
        audio: Input audio.
        sr: Sample rate.
        preset: Preset name.
        **overrides: Override specific config values (e.g., enable_vad=False).

    Returns:
        Tuple of (processed_audio, stats).

    Example:
        audio, stats = preprocess_for_asr(
            audio, sr, preset="mobile", enable_drc=False
        )
    """
    base = PRESETS.get(preset, PreprocessingConfig())

    # 只接受 PreprocessingConfig 真正定义的字段，并用 replace 复制一份，
    # 避免像原实现那样直接改到全局 PRESETS 里的共享对象。
    allowed = {f.name for f in fields(base)}
    overrides = {k: v for k, v in overrides.items() if k in allowed}
    config = replace(base, **overrides) if overrides else base

    pipeline = AudioPreprocessingPipeline(config)
    return pipeline.process(audio, sr)
