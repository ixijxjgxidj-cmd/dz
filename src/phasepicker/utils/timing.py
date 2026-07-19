"""时间对齐工具（Time alignment）.

⚠️ 这是整个系统最容易"整盘皆输"的地方。
模型输出的是"第几个采样点"，必须精确换算成绝对 UTC 到时。任何一处
（重采样后采样率没更新、起点时间没同步、多段数据 gap）出错，都会让
所有到时发生系统性偏移——模型本身没问题，但分数直接归零。

因此这里的逻辑刻意做得极简且无重依赖（纯 float 运算），并配套单元测试
（tests/test_timing.py）反复验证。请勿在别处内联重复实现这段换算。

核心公式：
    绝对到时(秒) = 波形起点绝对时间(秒) + 采样点下标 / 采样率(Hz)
"""

from __future__ import annotations

from typing import Union


def sample_to_utc(
    sample_index: Union[int, float],
    starttime_utc: float,
    sampling_rate: float,
) -> float:
    """把采样点下标换算为绝对 UTC 时间戳（Unix epoch 秒）。

    Args:
        sample_index: 采样点下标（可为 float，模型可能输出亚采样点精度的峰值位置）。
        starttime_utc: 波形第一个采样点的绝对时间（Unix epoch 秒）。
        sampling_rate: 采样率 Hz，必须 > 0。

    Returns:
        绝对到时（Unix epoch 秒）。

    Raises:
        ValueError: 采样率非正。
    """
    if sampling_rate <= 0:
        raise ValueError(f"采样率必须为正，收到 {sampling_rate}")
    return starttime_utc + float(sample_index) / sampling_rate


def utc_to_sample(
    time_utc: float,
    starttime_utc: float,
    sampling_rate: float,
) -> float:
    """sample_to_utc 的逆运算：绝对时间 → 采样点下标（float）。

    主要用于本地评分时把"标准答案的绝对到时"映射回波形坐标做可视化，
    以及单元测试的往返一致性校验（round-trip）。
    """
    if sampling_rate <= 0:
        raise ValueError(f"采样率必须为正，收到 {sampling_rate}")
    return (time_utc - starttime_utc) * sampling_rate


def resample_index(
    sample_index: Union[int, float],
    original_rate: float,
    target_rate: float,
) -> float:
    """当波形被重采样后，把"原采样率下的下标"换算到"目标采样率下的下标"。

    注意：正确的做法通常是先重采样波形、更新 starttime/采样率，再让模型在新
    波形上出下标——那样就不需要这个函数。此函数仅用于需要在原始坐标系与模型
    坐标系之间来回映射的边缘场景，单独测试以防出错。
    """
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("采样率必须为正")
    return float(sample_index) * (target_rate / original_rate)
