"""波形预处理（Preprocessing）——把校验过的原始波形整理成模型输入格式.

⚠️ 与时间对齐强相关：**重采样会改变采样率，且可能改变起点时间**。
本模块的铁律是：任何改变采样率/起点的操作，都必须同步更新 Waveform 的
``sampling_rate`` 与 ``starttime_utc``，否则后续的采样点→绝对到时换算会系统性偏移。
（这正是 timing.py 头部反复强调的坑，此处是它的第一现场。）

处理链（顺序有讲究）：
    去均值 → 去线性趋势 → 带通滤波 → （必要时）重采样 → 幅值归一化

设计说明：
- 用 ObsPy 的 Trace/Stream 做 detrend/filter/resample，成熟稳定、边界处理正确，
  不自己手写滤波器（自写滤波器的相位延迟极易引入到时偏差）。
- 归一化不改变时间轴，仅稳定不同仪器/量纲（速度 vs 加速度）的幅值分布，
  帮助跨域泛化——这是应对"预训练模型域差异"的低成本手段之一。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..types import Waveform, CHANNEL_ORDER


@dataclass
class PreprocessConfig:
    """预处理参数。全部可配置，便于本地评分脚本做参数搜索。

    Attributes:
        target_sampling_rate: 目标采样率 Hz。SeisBench 预训练模型基本按 100Hz
            训练，非 100Hz 必须重采样到此值。设为 None 则不重采样（要求输入已是目标率）。
        bandpass_freqmin: 带通下限 Hz。地震体波能量主要在 1~20Hz，
            1Hz 高通可压制海浪/仪器低频漂移。
        bandpass_freqmax: 带通上限 Hz。需 < 奈奎斯特频率（采样率/2）。
        detrend_type: 去趋势方式，"linear" 或 "demean"（这里链式先 demean 再 linear）。
        normalize: 是否做逐通道幅值归一化。
        normalize_mode: "std"（除以标准差）或 "max"（除以峰值绝对值）。
    """

    target_sampling_rate: Optional[float] = 100.0
    bandpass_freqmin: float = 1.0
    bandpass_freqmax: float = 20.0
    detrend_type: str = "linear"
    normalize: bool = True
    normalize_mode: str = "std"


def _to_obspy_stream(wf: Waveform):
    """把内部 Waveform 转回 ObsPy Stream 以复用其成熟的信号处理。"""
    from obspy import Stream, Trace, UTCDateTime  # 延迟导入

    st = Stream()
    for i, comp in enumerate(CHANNEL_ORDER):
        tr = Trace(data=np.ascontiguousarray(wf.data[i], dtype=np.float64))
        tr.stats.sampling_rate = wf.sampling_rate
        tr.stats.starttime = UTCDateTime(wf.starttime_utc)
        tr.stats.channel = comp
        st.append(tr)
    return st


def preprocess(wf: Waveform, cfg: Optional[PreprocessConfig] = None) -> Waveform:
    """对单台站三分量波形执行完整预处理链，返回新的 Waveform。

    关键：重采样后从 ObsPy Trace 重新读取 sampling_rate 与 starttime，
    保证元数据与数据严格同步（不手动假设它们没变）。

    Args:
        wf: 输入波形（来自 mseed_reader，通道顺序已是 [Z, N, E]）。
        cfg: 预处理配置，None 用默认。

    Returns:
        预处理后的 Waveform，通道顺序仍为 [Z, N, E]。

    Raises:
        RuntimeError: ObsPy 未安装。
    """
    cfg = cfg or PreprocessConfig()
    try:
        st = _to_obspy_stream(wf)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("预处理需要 ObsPy，请先安装 obspy") from exc

    # 1) 去均值 + 去趋势
    st.detrend("demean")
    if cfg.detrend_type == "linear":
        st.detrend("linear")

    # 2) 带通滤波（zerophase=True 双向滤波，消除相位延迟——对到时精度至关重要）
    nyquist = wf.sampling_rate / 2.0
    fmax = min(cfg.bandpass_freqmax, nyquist * 0.95)
    if cfg.bandpass_freqmin < fmax:
        st.filter(
            "bandpass",
            freqmin=cfg.bandpass_freqmin,
            freqmax=fmax,
            corners=4,
            zerophase=True,
        )

    # 3) 重采样（若需要）。ObsPy 的 resample 会正确维护 starttime，
    #    我们随后从 trace 重新读取，绝不手动推算。
    if (
        cfg.target_sampling_rate is not None
        and abs(wf.sampling_rate - cfg.target_sampling_rate) > 1e-6
    ):
        # 下采样前 ObsPy 要求先做抗混叠（filter 已限带），再 resample
        st.resample(cfg.target_sampling_rate)

    # 4) 从处理后的 stream 重新组装数据与元数据
    st.sort()  # 保证顺序稳定
    by_comp = {tr.stats.channel: tr for tr in st}
    arrays = []
    for comp in CHANNEL_ORDER:
        tr = by_comp[comp]
        arrays.append(np.asarray(tr.data, dtype=np.float32))
    n = min(a.shape[0] for a in arrays)
    data = np.stack([a[:n] for a in arrays], axis=0)

    ref = by_comp[CHANNEL_ORDER[0]]
    new_rate = float(ref.stats.sampling_rate)          # 从 trace 读，不假设
    new_start = float(ref.stats.starttime.timestamp)   # 从 trace 读，不假设

    # 5) 幅值归一化（不改时间轴）
    if cfg.normalize:
        data = _normalize(data, cfg.normalize_mode)

    return Waveform(
        data=data,
        sampling_rate=new_rate,
        starttime_utc=new_start,
        station=wf.station,
    )


def _normalize(data: np.ndarray, mode: str) -> np.ndarray:
    """逐通道归一化。空/常数道用 eps 兜底，避免除零产生 NaN 污染模型输入。"""
    out = data.astype(np.float32, copy=True)
    eps = 1e-8
    for i in range(out.shape[0]):
        ch = out[i]
        if mode == "max":
            scale = np.max(np.abs(ch))
        else:  # "std"
            scale = np.std(ch)
        out[i] = ch / (scale + eps)
    return out
