"""震级快速估计（EEW display layer）.

===== 原理与诚实边界（写给非 AI 背景的队友）=====
真正的地震预警系统会用 P 波最初几秒的振幅/位移特征快速估计震级（如 Pd 法：
用 P 波到达后 3 秒内的最大位移幅值，配合震中距，套一个经验公式估 M）。

**必须诚实的一点**：赛题波形"不含位置信息"，也没有台站响应/仪器增益标定，
所以这里给出的震级是**相对量级的演示性估计，不是科学定标后的绝对震级**。
我们在答辩中要明确说明：接入真实台网（有台站坐标、仪器响应、区域标定）后，
同一套公式即可产出可用的震级——现在展示的是"方法可行"，不是"数值精确"。

这种诚实反而加分：专家最反感"用无位置信息的单条波形硬吹精确震中和震级"。

===== 采用的经验形式 =====
Pd 型经验式（示意）：  M ≈ a * log10(Pd) + b * log10(R) + c
  Pd  = P 波到达后 τ 秒窗口内的最大位移幅值（对速度波形积分一次得位移）
  R   = 震中距（km），来自定位模块；无 R 时退化为仅用 Pd 的相对估计
  a,b,c = 经验系数，需用有标注震级的数据拟合；缺省给占位值并明确标注。

零重依赖（仅 numpy）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# 经验系数占位值：必须用"带真实震级标注的数据"重新拟合后替换。
# 未标定前，输出仅供相对比较（哪条更大），不代表绝对震级。
DEFAULT_COEF_A = 1.0
DEFAULT_COEF_B = 1.0
DEFAULT_COEF_C = 0.0


@dataclass
class MagnitudeResult:
    """震级估计结果，附带"是否已标定"的诚实标志。"""

    magnitude: float
    is_calibrated: bool
    """False 表示用的是未标定的占位系数，数值仅供相对比较。答辩必须如实说明。"""
    pd_displacement: float
    """P 波窗口内最大位移幅值（用于溯源/画图）。"""
    epicentral_distance_km: Optional[float]

    def summary(self) -> str:
        tag = "已标定" if self.is_calibrated else "未标定(仅相对)"
        r = (
            f"{self.epicentral_distance_km:.1f}km"
            if self.epicentral_distance_km is not None
            else "n/a"
        )
        return f"M≈{self.magnitude:.2f} [{tag}] | Pd={self.pd_displacement:.3e} | R={r}"


def _velocity_to_displacement(velocity: np.ndarray, dt: float) -> np.ndarray:
    """速度波形积分为位移（累积梯形积分）。dt = 1/采样率。

    赛题给的是速度波形，Pd 法需要位移，所以先积分一次。
    先去均值以抑制积分漂移（速度里的直流分量积分后会线性发散）。
    """
    v = np.asarray(velocity, dtype=float)
    v = v - np.mean(v)
    disp = np.cumsum(v) * dt
    # 再去一次线性趋势，进一步抑制积分漂移
    n = len(disp)
    if n >= 2:
        t = np.arange(n)
        coef = np.polyfit(t, disp, 1)
        disp = disp - (coef[0] * t + coef[1])
    return disp


def estimate_magnitude(
    velocity_z: np.ndarray,
    p_sample_index: int,
    sampling_rate: float,
    epicentral_distance_km: Optional[float] = None,
    tau_s: float = 3.0,
    coef: Optional[tuple] = None,
) -> MagnitudeResult:
    """基于 P 波后 τ 秒位移幅值的快速震级估计。

    Args:
        velocity_z: 垂直分量速度波形（1D）。竖直分量对 P 波最敏感。
        p_sample_index: P 波到时对应的采样点下标（来自震相拾取）。
        sampling_rate: 采样率 Hz。
        epicentral_distance_km: 震中距（来自定位模块）；None 时退化为仅 Pd 的相对估计。
        tau_s: P 波后取多长窗口算 Pd，默认 3 秒（EEW 常用）。
        coef: (a, b, c) 经验系数；None 时用未标定占位值并把 is_calibrated 设为 False。

    Returns:
        MagnitudeResult。
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate 必须为正")
    v = np.asarray(velocity_z, dtype=float)
    p_idx = int(max(0, p_sample_index))
    win = int(round(tau_s * sampling_rate))
    seg = v[p_idx : p_idx + win]
    if seg.size < 2:
        # P 到时太靠近末尾，取不到窗口：返回退化结果而非崩溃
        return MagnitudeResult(
            magnitude=float("nan"),
            is_calibrated=False,
            pd_displacement=0.0,
            epicentral_distance_km=epicentral_distance_km,
        )

    disp = _velocity_to_displacement(seg, 1.0 / sampling_rate)
    pd = float(np.max(np.abs(disp)))
    pd_safe = max(pd, 1e-12)  # 防 log10(0)

    a, b, c = coef if coef is not None else (DEFAULT_COEF_A, DEFAULT_COEF_B, DEFAULT_COEF_C)
    is_calibrated = coef is not None

    mag = a * np.log10(pd_safe)
    if epicentral_distance_km is not None and epicentral_distance_km > 0:
        mag += b * np.log10(epicentral_distance_km)
    mag += c

    return MagnitudeResult(
        magnitude=float(mag),
        is_calibrated=is_calibrated,
        pd_displacement=pd,
        epicentral_distance_km=epicentral_distance_km,
    )
