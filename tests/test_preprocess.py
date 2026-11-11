"""预处理服务与管道测试。"""

from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing import PRESETS, AudioPreprocessingPipeline
from app.services.preprocess import UnknownPresetError, build_pipeline, list_presets


def _noisy_tone(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    tone = 0.4 * np.sin(2 * np.pi * 220 * t)
    return (tone + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_build_pipeline_uses_preset_config():
    pipeline = build_pipeline("noisy_environment")
    preset = PRESETS["noisy_environment"]
    assert pipeline.config.noise_reduction_method == preset.noise_reduction_method
    assert pipeline.config.vad_threshold == preset.vad_threshold


def test_build_pipeline_applies_overrides():
    pipeline = build_pipeline("meeting_room", enable_vad=False, enable_nr=False, nr_strength=0.2)
    assert pipeline.config.enable_vad is False
    assert pipeline.config.enable_noise_reduction is False
    assert pipeline.config.noise_reduction_prop_decrease == pytest.approx(0.2)


def test_build_pipeline_clamps_strength():
    assert build_pipeline("mobile", nr_strength=5).config.noise_reduction_prop_decrease == 1.0
    assert build_pipeline("mobile", nr_strength=-1).config.noise_reduction_prop_decrease == 0.0


def test_build_pipeline_does_not_mutate_shared_preset():
    before = PRESETS["meeting_room"].enable_vad
    build_pipeline("meeting_room", enable_vad=False)
    assert PRESETS["meeting_room"].enable_vad is before


def test_build_pipeline_rejects_unknown_preset():
    with pytest.raises(UnknownPresetError):
        build_pipeline("studio")


def test_list_presets_shape():
    presets = list_presets()
    assert set(presets) == set(PRESETS)
    assert {"vad", "noise_reduction", "drc", "eq", "agc", "deess", "dereverb"} <= set(presets["mobile"])


@pytest.mark.parametrize("preset", ["lightweight", "meeting_room", "noisy_environment"])
def test_pipeline_process_returns_audio_and_stats(preset: str):
    audio = _noisy_tone()
    processed, stats = AudioPreprocessingPipeline.from_preset(preset).process(audio, 16000)

    assert processed.ndim == 1
    assert processed.dtype == np.float32
    assert len(processed) > 0
    payload = stats.to_dict()
    assert payload["input_duration_s"] == pytest.approx(1.0, abs=0.05)
    assert "output_rms_db" in payload


def test_preprocess_for_asr_does_not_leak_overrides():
    from app.preprocessing import preprocess_for_asr

    audio = _noisy_tone(0.2)
    preprocess_for_asr(audio, 16000, preset="mobile", enable_vad=False)
    assert PRESETS["mobile"].enable_vad is True


def test_stft_istft_roundtrip():
    """自实现的 STFT/ISTFT 必须能近似无损还原（重叠相加归一化正确）。"""
    from app.preprocessing.stft import istft, stft

    audio = _noisy_tone(0.5)
    spectrum = stft(audio, n_fft=512, hop_length=128)
    restored = istft(spectrum, hop_length=128, length=audio.size)

    assert spectrum.shape == (257, spectrum.shape[1])
    assert restored.shape == audio.shape
    # 两端有补零边界效应，只比较中间部分
    mse = float(np.mean((restored[1000:-1000] - audio[1000:-1000]) ** 2))
    assert mse < 1e-4, f"roundtrip mse={mse}"
