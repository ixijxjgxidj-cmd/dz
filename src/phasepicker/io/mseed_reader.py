"""mseed 读取 + 校验 + 容错（Data ingestion）.

⚠️ 依赖 ObsPy，在本沙箱无法运行；请在你的机器上 `pip install obspy` 后运行。
本模块所有逻辑已按 ObsPy 官方 API 编写，函数签名与内部数据结构（types.py）
严格对齐，配套的时间对齐核心（utils/timing.py）已在纯 numpy 环境通过单元测试。

设计目标（对应赛题"数据处理模块"）：
1. 读取三分量 mseed，做基本校验：分量完整性、采样率、时间连续性。
2. 对异常数据（缺分量、非标准采样率、超短/超长波形、多台站混合）明确容错，
   任何异常都以"跳过该台站 + 结构化告警"收场，绝不让进程崩溃。
3. 输出统一的 Waveform 列表（每台站一个），通道顺序固定 [Z, N, E]。

容错哲学：**宁可跳过一个台站，也不让整个请求挂掉。** 每个可预见的异常都
被捕获并记录为 IngestWarning，调用方（API 层）据此决定是否降级返回空结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..types import Waveform, CHANNEL_ORDER

logger = logging.getLogger(__name__)

# ObsPy 延迟导入：让本模块在无 obspy 环境也能被 import（便于测试其它模块）。
try:
    from obspy import read, Stream, Trace, UTCDateTime  # type: ignore

    _OBSPY_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    _OBSPY_AVAILABLE = False


# 允许的采样率白名单（Hz）。非白名单会被重采样到 TARGET_RATE。
# 100Hz 是 SeisBench PhaseNet/EQTransformer 的原生采样率。
TARGET_RATE = 100.0
# 波形时长的合理边界（秒）。超短无法给模型足够上下文；超长做切窗（此处仅告警，
# 切窗在 preprocess 里做）。上界同时是"超时防护"的第一道闸——限制输入长度，
# 比指望中途 kill 正在跑的推理更可靠。
MIN_DURATION_S = 5.0
MAX_DURATION_S = 3600.0


@dataclass
class IngestWarning:
    """一条结构化告警，用于赛后复盘与 API 层降级决策。"""

    station: str
    reason: str
    detail: str = ""


@dataclass
class IngestResult:
    """读取结果：成功的波形 + 所有告警。"""

    waveforms: List[Waveform] = field(default_factory=list)
    warnings: List[IngestWarning] = field(default_factory=list)

    def add_warning(self, station: str, reason: str, detail: str = "") -> None:
        w = IngestWarning(station=station, reason=reason, detail=detail)
        self.warnings.append(w)
        logger.warning("摄入告警 [%s] %s: %s", station, reason, detail)


def _channel_key(channel_code: str) -> Optional[str]:
    """从 SEED 通道代码（如 'BHZ'/'HHN'/'EHE'）提取方向分量 Z/N/E。

    SEED 通道代码约定：末位是方向（Z/N/E，或 1/2/3 等价于 N/E）。
    这里做一个稳健映射，兼容常见的数字分量命名。
    """
    if not channel_code:
        return None
    last = channel_code[-1].upper()
    mapping = {"Z": "Z", "N": "N", "E": "E", "1": "N", "2": "E", "3": "Z"}
    return mapping.get(last)


def read_mseed_bytes(raw: bytes) -> "Stream":
    """从二进制字节读取 mseed 为 ObsPy Stream。API 层接收上传流后调用。

    Raises:
        RuntimeError: ObsPy 不可用。
        ValueError: 无法解析为有效 mseed。
    """
    if not _OBSPY_AVAILABLE:
        raise RuntimeError("ObsPy 未安装；请在部署环境 `pip install obspy`。")
    import io

    try:
        return read(io.BytesIO(raw), format="MSEED")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"mseed 解析失败：{exc}") from exc


def group_by_station(stream: "Stream") -> dict:
    """把 Stream 中的 Trace 按台站 NET.STA 分组，支持多台站混合文件。"""
    groups: dict = {}
    for tr in stream:
        key = f"{tr.stats.network}.{tr.stats.station}"
        groups.setdefault(key, []).append(tr)
    return groups


def _merge_gappy(traces: List["Trace"], station: str, result: IngestResult) -> Optional["Stream"]:
    """合并同一分量可能存在的多段（gap/overlap）。

    时间不连续（gap）是地震数据常态。用 ObsPy 的 merge 填充：
    - method=1：重叠段取插值；缺口用 fill_value 填 0（后续预处理会去均值，
      填 0 不会引入直流偏置到模型可感知的程度，且保持采样点与时间的严格对应）。
    保持时间连续性是"采样点↔绝对时间"换算成立的前提，这一步至关重要。
    """
    try:
        st = Stream(traces=traces)
        st.merge(method=1, fill_value=0, interpolation_samples=0)
        return st
    except Exception as exc:  # noqa: BLE001
        result.add_warning(station, "merge_failed", str(exc))
        return None


def build_waveform(
    traces: List["Trace"],
    station: str,
    result: IngestResult,
) -> Optional[Waveform]:
    """把某台站的一组 Trace 组装成统一的三分量 Waveform，带完整校验。

    返回 None 表示该台站不可用（原因已记入 result.warnings），调用方应跳过。
    """
    # 1) 按分量归类
    by_comp: dict = {}
    for tr in traces:
        comp = _channel_key(tr.stats.channel)
        if comp is None:
            result.add_warning(station, "unknown_channel", tr.stats.channel)
            continue
        by_comp.setdefault(comp, []).append(tr)

    # 2) 分量完整性校验
    missing = [c for c in CHANNEL_ORDER if c not in by_comp]
    if missing:
        result.add_warning(
            station, "missing_component", f"缺少分量 {missing}；仅 {list(by_comp)}"
        )
        return None

    # 3) 每个分量合并多段 + 采样率一致性
    merged: dict = {}
    rates = set()
    for comp in CHANNEL_ORDER:
        st = _merge_gappy(by_comp[comp], station, result)
        if st is None or len(st) == 0:
            result.add_warning(station, "empty_after_merge", comp)
            return None
        tr = st[0]
        merged[comp] = tr
        rates.add(round(float(tr.stats.sampling_rate), 6))

    if len(rates) > 1:
        result.add_warning(station, "inconsistent_sampling_rate", f"{rates}")
        return None
    sampling_rate = float(next(iter(rates)))
    if sampling_rate <= 0:
        result.add_warning(station, "nonpositive_sampling_rate", str(sampling_rate))
        return None

    # 4) 三分量对齐到公共时间窗（取交集），保证 (3, n) 严格同长且同起点
    starts = [merged[c].stats.starttime for c in CHANNEL_ORDER]
    ends = [merged[c].stats.endtime for c in CHANNEL_ORDER]
    common_start = max(starts)
    common_end = min(ends)
    if common_end <= common_start:
        result.add_warning(station, "no_time_overlap", "三分量时间窗无交集")
        return None

    arrays = []
    for c in CHANNEL_ORDER:
        tr = merged[c].copy()
        try:
            tr.trim(common_start, common_end, pad=False)
        except Exception as exc:  # noqa: BLE001
            result.add_warning(station, "trim_failed", f"{c}: {exc}")
            return None
        arrays.append(np.asarray(tr.data, dtype=np.float32))

    # 5) 长度对齐（trim 后可能差 1 个采样点，取最短，保证矩阵规整）
    n = min(a.shape[0] for a in arrays)
    if n <= 0:
        result.add_warning(station, "empty_data", "trim 后无采样点")
        return None
    data = np.stack([a[:n] for a in arrays], axis=0)  # (3, n) = [Z, N, E]

    # 6) 时长边界校验（超短拒绝；超长仅告警，交给预处理切窗）
    duration = n / sampling_rate
    if duration < MIN_DURATION_S:
        result.add_warning(
            station, "too_short", f"{duration:.2f}s < {MIN_DURATION_S}s"
        )
        return None
    if duration > MAX_DURATION_S:
        result.add_warning(
            station, "too_long", f"{duration:.2f}s > {MAX_DURATION_S}s（将切窗处理）"
        )

    starttime_utc = float(common_start.timestamp)  # UTCDateTime -> epoch 秒
    return Waveform(
        data=data,
        sampling_rate=sampling_rate,
        starttime_utc=starttime_utc,
        station=station,
    )


def load_waveforms(raw: bytes) -> IngestResult:
    """顶层入口：原始 mseed 字节 → 校验过的多台站 Waveform 列表。

    绝不抛出未捕获异常（除非 ObsPy 缺失，那是部署问题应尽早暴露）。
    任何数据层面的问题都以 IngestWarning 形式返回。
    """
    result = IngestResult()
    try:
        stream = read_mseed_bytes(raw)
    except ValueError as exc:
        result.add_warning("<file>", "parse_error", str(exc))
        return result

    groups = group_by_station(stream)
    if not groups:
        result.add_warning("<file>", "empty_stream", "文件中无任何 Trace")
        return result

    for station, traces in groups.items():
        try:
            wf = build_waveform(traces, station, result)
            if wf is not None:
                result.waveforms.append(wf)
        except Exception as exc:  # noqa: BLE001 —— 最后一道防线，绝不让单台站拖垮整体
            result.add_warning(station, "unexpected_error", repr(exc))
    return result
