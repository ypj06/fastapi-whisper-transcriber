"""预处理服务层：把"预设 + 用户覆盖"翻译成一条可执行的管道。

原实现在 API 函数里手写了一份属性拷贝清单（19 个字段逐条 setattr），
新增字段时极易漏改；这里直接用 ``dataclasses.replace`` 从预设派生副本。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.preprocessing import PRESETS, AudioPreprocessingPipeline, PreprocessingConfig


class UnknownPresetError(ValueError):
    """预设名不存在。"""


def build_pipeline(
    preset: str = "meeting_room",
    *,
    enable_vad: bool | None = None,
    enable_nr: bool | None = None,
    enable_se: bool | None = None,
    nr_strength: float | None = None,
) -> AudioPreprocessingPipeline:
    """按预设构建预处理管道，允许局部覆盖。

    Args:
        preset: 预设名，见 :func:`list_presets`。
        enable_vad: 覆盖是否启用 VAD。
        enable_nr: 覆盖是否启用降噪。
        enable_se: 覆盖是否启用信号增强。
        nr_strength: 降噪强度，会裁剪到 0.0~1.0。

    Raises:
        UnknownPresetError: 预设名不在 ``PRESETS`` 中。
    """
    if preset not in PRESETS:
        raise UnknownPresetError(
            f"未知预设 '{preset}'，可选：{', '.join(PRESETS)}"
        )

    overrides: dict[str, Any] = {}
    if enable_vad is not None:
        overrides["enable_vad"] = enable_vad
    if enable_nr is not None:
        overrides["enable_noise_reduction"] = enable_nr
    if enable_se is not None:
        overrides["enable_signal_enhancement"] = enable_se
    if nr_strength is not None:
        overrides["noise_reduction_prop_decrease"] = min(1.0, max(0.0, nr_strength))

    config: PreprocessingConfig = replace(PRESETS[preset], **overrides) if overrides else PRESETS[preset]
    return AudioPreprocessingPipeline(config)


def list_presets() -> dict[str, dict[str, Any]]:
    """列出所有预设的生效配置，供前端展示与调试。"""
    return {
        name: {
            "vad": cfg.vad_method.value if cfg.enable_vad else "off",
            "noise_reduction": cfg.noise_reduction_method.value if cfg.enable_noise_reduction else "off",
            "drc": cfg.enable_drc,
            "eq": cfg.eq_preset if cfg.enable_eq else "off",
            "agc": cfg.enable_agc,
            "deess": cfg.enable_deess,
            "dereverb": cfg.enable_dereverb,
        }
        for name, cfg in PRESETS.items()
    }
