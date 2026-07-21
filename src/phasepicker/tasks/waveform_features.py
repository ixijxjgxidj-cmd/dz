"""T2/T3 轻量基线的固定长度波形特征。

目标不是替代深度网络，而是先把“固定常数占位”升级为真正从波形学习的、
可复现且 CPU 可跑的基线。特征同时覆盖：
- 振幅/能量（对震级有用）；
- 频谱与持续时间（对地震、爆破、塌陷、滑坡区分有用）；
- 三分量比例与相关性（减少只看单通道造成的误判）。

输入可以是 ObsPy Stream，也可以是测试用的 ``[(channel, sampling_rate, data), ...]``。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


_BANDS = ((0.2, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 40.0))
_COMPONENTS = ("Z", "N", "E")
_PER_COMPONENT_NAMES = (
    "log_std",
    "log_peak",
    "log_p2p",
    "log_abs_q50",
    "log_abs_q90",
    "log_abs_q99",
    "crest_factor",
    "zero_cross_rate",
    "peak_position",
    "dominant_freq_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "bandpower_0p2_1",
    "bandpower_1_3",
    "bandpower_3_8",
    "bandpower_8_15",
    "bandpower_15_40",
)

FEATURE_NAMES: Tuple[str, ...] = tuple(
    [f"{comp}_{name}" for comp in _COMPONENTS for name in _PER_COMPONENT_NAMES]
    + [
        "duration_s",
        "sampling_rate_hz",
        "log_vector_rms",
        "log_vector_peak",
        "horizontal_vertical_rms_ratio",
        "vertical_total_rms_ratio",
        "corr_z_n",
        "corr_z_e",
        "corr_n_e",
    ]
)


def _log1p_abs(value: float) -> float:
    return float(np.log1p(max(0.0, abs(float(value)))))


def _component_features(data: np.ndarray, sampling_rate: float) -> List[float]:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return [0.0] * len(_PER_COMPONENT_NAMES)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    x = x - float(np.mean(x))
    abs_x = np.abs(x)
    std = float(np.std(x))
    peak = float(np.max(abs_x))
    p2p = float(np.ptp(x))
    q50, q90, q99 = (float(v) for v in np.quantile(abs_x, (0.50, 0.90, 0.99)))
    crest = peak / (std + 1e-12)
    if x.size >= 2:
        zero_cross = float(np.mean((x[:-1] >= 0.0) != (x[1:] >= 0.0)))
    else:
        zero_cross = 0.0
    peak_pos = float(np.argmax(abs_x) / max(1, x.size - 1))

    if x.size >= 4 and sampling_rate > 0:
        window = np.hanning(x.size)
        power = np.abs(np.fft.rfft(x * window)) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate)
        if power.size:
            power[0] = 0.0
        total_power = float(np.sum(power))
        if total_power > 0.0:
            dominant = float(freqs[int(np.argmax(power))])
            centroid = float(np.sum(freqs * power) / total_power)
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
            band_ratios = []
            for low, high in _BANDS:
                mask = (freqs >= low) & (freqs < min(high, sampling_rate / 2 + 1e-9))
                band_ratios.append(float(np.sum(power[mask]) / total_power))
        else:
            dominant = centroid = bandwidth = 0.0
            band_ratios = [0.0] * len(_BANDS)
    else:
        dominant = centroid = bandwidth = 0.0
        band_ratios = [0.0] * len(_BANDS)

    return [
        _log1p_abs(std),
        _log1p_abs(peak),
        _log1p_abs(p2p),
        _log1p_abs(q50),
        _log1p_abs(q90),
        _log1p_abs(q99),
        float(crest),
        zero_cross,
        peak_pos,
        dominant,
        centroid,
        bandwidth,
        *band_ratios,
    ]


def _canonical_component(channel: str) -> str:
    last = str(channel).upper()[-1:] if channel else ""
    if last == "Z":
        return "Z"
    if last in {"N", "1", "Y"}:
        return "N"
    if last in {"E", "2", "X"}:
        return "E"
    return ""


def stream_to_components(stream) -> Tuple[dict, float]:
    """把 ObsPy Stream 或测试三元组列表整理成 Z/N/E 三分量字典。"""
    components = {}
    sampling_rates: List[float] = []
    for item in stream:
        if hasattr(item, "stats") and hasattr(item, "data"):
            channel = getattr(item.stats, "channel", "")
            sr = float(getattr(item.stats, "sampling_rate", 0.0))
            data = np.asarray(item.data)
        else:
            channel, sr, data = item
            sr = float(sr)
            data = np.asarray(data)
        comp = _canonical_component(channel)
        if not comp or data.size == 0 or sr <= 0:
            continue
        # 同一分量若有多段，保留样本更多的一段；官方包当前每分量都是一条。
        if comp not in components or data.size > components[comp][1].size:
            components[comp] = (sr, data)
        sampling_rates.append(sr)
    sr = float(np.median(sampling_rates)) if sampling_rates else 0.0
    return components, sr


def extract_waveform_features(stream) -> np.ndarray:
    """从三分量波形提取与 ``FEATURE_NAMES`` 严格对齐的一维 float 特征。"""
    components, default_sr = stream_to_components(stream)
    values: List[float] = []
    demeaned = {}
    durations: List[float] = []
    for comp in _COMPONENTS:
        sr, data = components.get(comp, (default_sr, np.zeros(1, dtype=float)))
        values.extend(_component_features(data, sr))
        x = np.asarray(data, dtype=np.float64).reshape(-1)
        x = np.where(np.isfinite(x), x, 0.0)
        x = x - float(np.mean(x)) if x.size else x
        demeaned[comp] = x
        if sr > 0 and x.size:
            durations.append(x.size / sr)

    duration = float(max(durations)) if durations else 0.0
    sr = default_sr
    all_arrays = [demeaned[c] for c in _COMPONENTS if demeaned[c].size]
    rms = {c: float(np.sqrt(np.mean(demeaned[c] ** 2))) if demeaned[c].size else 0.0 for c in _COMPONENTS}
    vector_rms = float(np.sqrt(sum(v * v for v in rms.values())))
    vector_peak = max((float(np.max(np.abs(x))) for x in all_arrays), default=0.0)
    horizontal = float(np.sqrt(rms["N"] ** 2 + rms["E"] ** 2))
    hv_ratio = horizontal / (rms["Z"] + 1e-12)
    z_total_ratio = rms["Z"] / (vector_rms + 1e-12)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        n = min(a.size, b.size)
        if n < 2:
            return 0.0
        aa, bb = a[:n], b[:n]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        return float(np.clip(np.corrcoef(aa, bb)[0, 1], -1.0, 1.0))

    values.extend(
        [
            duration,
            sr,
            _log1p_abs(vector_rms),
            _log1p_abs(vector_peak),
            hv_ratio,
            z_total_ratio,
            corr(demeaned["Z"], demeaned["N"]),
            corr(demeaned["Z"], demeaned["E"]),
            corr(demeaned["N"], demeaned["E"]),
        ]
    )
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(f"特征长度错误：{out.shape}，期望 {(len(FEATURE_NAMES),)}")
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
