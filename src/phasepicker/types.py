"""核心数据结构定义（Core data structures）.

这个模块定义了整个系统内部流转的"通用语言"：一个震相拾取结果 ``Pick``。
所有模块（数据处理、推理、后处理、评分、API）都围绕这些结构交互，
这样即使官方接口格式还没确定，内部逻辑也能先跑通、先测试。

设计要点：
- 内部一律使用 **绝对 UTC 时间戳（float 秒，Unix epoch）** 表示到时，避免时区/相对偏移的歧义。
- 与官方格式的对齐只发生在最外层的"适配层"（api/adapters.py），
  官方规范一旦拿到，只改适配层，核心逻辑零改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class PhaseType(str, Enum):
    """震相类型。继承 str 便于直接 JSON 序列化。"""

    P = "P"
    S = "S"


@dataclass
class Pick:
    """单个震相拾取结果。

    Attributes:
        phase: 震相类型（P 或 S）。
        time_utc: 到时的绝对时间，Unix epoch 秒（float）。选用绝对时间作为
            内部唯一真理来源，是为了避免"相对哪个起点"的歧义——不同 mseed
            文件、重采样、多段拼接都可能改变起点，绝对时间不受影响。
        confidence: 模型置信度 [0, 1]。用于阈值过滤与去重时择优。
        station: 台站标识（NET.STA），可选。多台站文件时用于分组。
        sample_index: 模型输出的原始采样点下标，仅用于调试/回溯，非必填。
    """

    phase: PhaseType
    time_utc: float
    confidence: float
    station: Optional[str] = None
    sample_index: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


@dataclass
class Waveform:
    """送入模型的单台站三分量波形（预处理后的统一格式）。

    Attributes:
        data: 形状 (3, n_samples) 的数组，通道顺序固定为 [Z, N, E]。
            固定顺序是硬性约定——SeisBench 模型对通道顺序敏感，
            顺序错了不会报错但精度会崩，这是隐蔽 bug 的高发区。
        sampling_rate: 采样率 Hz。
        starttime_utc: 波形第一个采样点的绝对时间（Unix epoch 秒）。
            这是把"采样点下标"换算回"绝对到时"的锚点，必须与 data 严格同步。
        station: 台站标识 NET.STA。
    """

    data: object  # numpy.ndarray, 避免在无 numpy 环境 import 失败
    sampling_rate: float
    starttime_utc: float
    station: str = ""

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[-1])

    @property
    def duration(self) -> float:
        """波形时长（秒）。"""
        return self.n_samples / self.sampling_rate


# 通道顺序的全局约定，供各模块引用，禁止在别处硬编码字符串。
CHANNEL_ORDER = ("Z", "N", "E")
