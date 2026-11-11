"""音频预处理子包（原 Topic 3 模块）。

对外只暴露这几项：预处理管道、配置、预设表与音频读写工具。
"""

from .pipeline import (
    PRESETS,
    AudioPreprocessingPipeline,
    NoiseReductionMethod,
    PreprocessingConfig,
    ProcessingStats,
    VadMethod,
    preprocess_for_asr,
)
from .audio_utils import load_audio, save_audio

__all__ = [
    "PRESETS",
    "AudioPreprocessingPipeline",
    "NoiseReductionMethod",
    "PreprocessingConfig",
    "ProcessingStats",
    "VadMethod",
    "preprocess_for_asr",
    "load_audio",
    "save_audio",
]
