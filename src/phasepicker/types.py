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
from enum import Enum, IntEnum
from typing import List, Optional


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


# =========================================================================
# 官方比赛任务层（Official exam task layer）
# =========================================================================
# 去年官方三个任务（T1 震相到时 / T2 震级 / T3 事件分类）的输入输出各不相同，
# 但都以"单个 .mseed 文件"为最小评测单位。下面这些结构是官方任务与内部逻辑
# 之间的"通用语言"：解析层（io/official_answers.py）产出它们，写出层
# （io/submission_writer.py）消费它们，评估层（eval/official_eval.py）比对它们。
#
# 关键约定：T1 的到时用 **相对波形起点的秒**（float），与官方答案文件一致；
# 这与内部 Pick.time_utc 的"绝对 epoch 秒"是两套坐标，转换只发生在最外层。


class ExamTask(str, Enum):
    """官方比赛任务类型。继承 str 便于直接与路径/JSON 交互。"""

    T1 = "T1"  # 震相到时拾取（P/S，单位：相对波形起点秒）
    T2 = "T2"  # 震级估计（单个 float）
    T3 = "T3"  # 事件类型分类（1..5 整数）


class EventClass(IntEnum):
    """T3 事件类别。取值 1..5，含义由官方规定。

    继承 IntEnum：既能当整数直接写出/比较，又带可读名字便于调试与打印。
    """

    EARTHQUAKE = 1  # 地震
    EXPLOSION = 2   # 爆破
    COLLAPSE = 3    # 塌陷
    LANDSLIDE = 4   # 滑坡
    OTHER = 5       # 其它


@dataclass
class ExamSample:
    """一个官方评测样本（对应一个 .mseed 文件）的元信息。

    Attributes:
        file_id: 文件标识，通常取 basename（如 ``T1.A.Q0001.mseed``），
            作为预测与答案对齐的唯一键。
        task: 该样本属于哪个任务（T1/T2/T3）。
        source_path: 样本来源路径。可能是普通文件路径，也可能是 zip 内条目名。
        starttime_utc: 波形起点绝对时间（epoch 秒），可选。扫描阶段通常未读波形，
            故默认 None；真正读取 waveform 后可回填，用于相对秒↔绝对秒换算。
        duration_s: 波形时长（秒），可选。
        station: 台站标识（NET.STA），可选。
    """

    file_id: str
    task: ExamTask
    source_path: str
    starttime_utc: Optional[float] = None
    duration_s: Optional[float] = None
    station: Optional[str] = None


@dataclass
class Task1Result:
    """T1 结果：一个文件的 P/S 到时（相对波形起点，秒）。

    第1轮每文件恰好一个 P 和一个 S；第2轮允许多个。用列表统一承载两种情形，
    单值场景就是长度为 1 的列表，避免下游分叉。
    """

    file_id: str
    p_times_s: List[float] = field(default_factory=list)
    s_times_s: List[float] = field(default_factory=list)


@dataclass
class Task2Result:
    """T2 结果：一个文件的震级估计（单个 float）。"""

    file_id: str
    magnitude: float


@dataclass
class Task3Result:
    """T3 结果：一个文件的事件类别（1..5）。

    label 统一存为 int（EventClass 是 IntEnum，可直接赋入）；confidence 可选，
    仅供内部排序/复盘，不进入官方提交行。
    """

    file_id: str
    label: int
    confidence: Optional[float] = None
