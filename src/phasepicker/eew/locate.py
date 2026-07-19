"""震中定位 + 发震时刻估计（EEW display layer）.

===== 原理（写给非 AI 背景的队友）=====
地震在震中发生后，P 波以速度 Vp 向四周传播。离震中越远的台站，P 波到得越晚。
如果我们有 N 个台站的经纬度，以及它们各自的 P 波到时，就能"反推"震中在哪、
地震什么时候发生的：

    对某个假设的震中 (lat, lon) 和发震时刻 t0，
    第 i 个台站的"理论 P 到时" = t0 + 震中到台站 i 的距离 / Vp。
    真实到时与理论到时的差，叫残差。
    我们在地图上撒一张网格，逐个格点试，找到"让所有台站残差平方和最小"的那个点，
    它就是最可能的震中；对应的 t0 就是发震时刻。

这叫**网格搜索定位（grid-search location）**。它简单、稳、可解释，
不需要梯度，非常适合答辩演示（可以画出残差热力图）。真实台网会用更复杂的
方法（如 Geiger 迭代、双差定位），但对展示层来说网格搜索足够且更直观。

===== 一个关键简化 =====
给定震中，最优 t0 有解析解：t0* = mean(观测到时 - 理论走时)。
所以网格搜索里每个格点不用再搜 t0，直接算均值即可，速度快很多。

零重依赖（仅 numpy），可直接测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 地球平均半径（km）。用球面距离而非平面近似，几十~几百 km 范围更准。
EARTH_RADIUS_KM = 6371.0

# 地壳 P 波典型速度（km/s）。真实应用应按区域速度模型调整；
# 这里给一个中地壳常用近似值，并允许调用方覆盖。
DEFAULT_VP_KM_S = 6.0
# S 波速度 ≈ Vp / 1.73（泊松体近似），用于估算 S 波到达 → 预警时间窗。
DEFAULT_VS_KM_S = DEFAULT_VP_KM_S / 1.73


@dataclass
class Station:
    """台站的地理信息。"""

    code: str
    latitude: float
    longitude: float
    elevation_km: float = 0.0


@dataclass
class StationArrival:
    """某台站观测到的 P 波绝对到时。"""

    station: Station
    p_time_utc: float
    """P 波到时，Unix epoch 秒。来自我们初赛的震相拾取结果。"""


@dataclass
class LocationResult:
    """定位结果。"""

    latitude: float
    longitude: float
    origin_time_utc: float
    """估计的发震时刻（Unix epoch 秒）。"""
    rms_residual_s: float
    """各台站到时残差的均方根（秒），越小越可信，可作为定位质量指标。"""
    n_stations: int
    used_vp_km_s: float

    def summary(self) -> str:
        return (
            f"震中≈({self.latitude:.3f}, {self.longitude:.3f}) | "
            f"发震时刻(epoch)={self.origin_time_utc:.2f} | "
            f"RMS残差={self.rms_residual_s:.3f}s | "
            f"台站数={self.n_stations} | Vp={self.used_vp_km_s}km/s"
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点球面距离（km）。经纬度输入为度。

    用 haversine 公式，避免平面近似在高纬度或大跨度时的误差。
    """
    r = EARTH_RADIUS_KM
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _best_origin_time(observed: np.ndarray, travel: np.ndarray) -> Tuple[float, float]:
    """给定观测到时与理论走时，求最优发震时刻及其 RMS 残差。

    最优 t0 = mean(observed - travel)（最小二乘解）。
    残差 = observed - (t0 + travel)，返回其 RMS。
    """
    diff = observed - travel
    t0 = float(np.mean(diff))
    resid = observed - (t0 + travel)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return t0, rms


def estimate_origin_and_location(
    arrivals: Sequence[StationArrival],
    vp_km_s: float = DEFAULT_VP_KM_S,
    grid_half_span_deg: float = 2.0,
    grid_step_deg: float = 0.05,
    refine: bool = True,
) -> LocationResult:
    """用多台站 P 到时做网格搜索定位 + 发震时刻估计。

    Args:
        arrivals: 各台站的 P 到时，至少 3 个台站才能稳定定位（少于 3 会抛错）。
        vp_km_s: P 波速度（km/s）。可按区域速度模型调整。
        grid_half_span_deg: 搜索网格以"台站质心"为中心，向四周延伸的半跨度（度）。
        grid_step_deg: 网格步长（度）。越小越精细、越慢。
        refine: 是否在粗网格最优点附近做一次更细的二级搜索（提升精度、成本很低）。

    Returns:
        LocationResult。

    Raises:
        ValueError: 台站数不足或 vp 非正。
    """
    if vp_km_s <= 0:
        raise ValueError("vp_km_s 必须为正")
    if len(arrivals) < 3:
        raise ValueError(
            f"定位至少需要 3 个台站，当前 {len(arrivals)} 个。"
            "台站太少无法唯一确定震中（这是物理限制，不是代码问题）。"
        )

    lats = np.array([a.station.latitude for a in arrivals], dtype=float)
    lons = np.array([a.station.longitude for a in arrivals], dtype=float)
    observed = np.array([a.p_time_utc for a in arrivals], dtype=float)

    def _search(center_lat: float, center_lon: float, half: float, step: float):
        best = None
        n = int(round(2 * half / step)) + 1
        grid_lat = np.linspace(center_lat - half, center_lat + half, n)
        grid_lon = np.linspace(center_lon - half, center_lon + half, n)
        for glat in grid_lat:
            for glon in grid_lon:
                # 该格点到各台站的距离 → 理论 P 走时
                dists = np.array(
                    [haversine_km(glat, glon, la, lo) for la, lo in zip(lats, lons)]
                )
                travel = dists / vp_km_s
                t0, rms = _best_origin_time(observed, travel)
                if best is None or rms < best[2]:
                    best = (glat, glon, rms, t0)
        return best

    # 一级搜索：以台站质心为中心
    c_lat, c_lon = float(np.mean(lats)), float(np.mean(lons))
    glat, glon, rms, t0 = _search(c_lat, c_lon, grid_half_span_deg, grid_step_deg)

    # 二级细化：在一级最优点附近用更小步长再搜一遍
    if refine:
        fine = _search(glat, glon, grid_step_deg * 2, grid_step_deg / 5)
        if fine is not None and fine[2] < rms:
            glat, glon, rms, t0 = fine

    return LocationResult(
        latitude=glat,
        longitude=glon,
        origin_time_utc=t0,
        rms_residual_s=rms,
        n_stations=len(arrivals),
        used_vp_km_s=vp_km_s,
    )


def estimate_back_azimuth(
    reference: Station,
    epicenter_lat: float,
    epicenter_lon: float,
) -> float:
    """从某参考台站看，震中位于哪个方位角（度，正北为0，顺时针）。

    用于"震中方向估计"的展示：即便定位不够精确，方向信息对预警也有意义。
    """
    lat1 = math.radians(reference.latitude)
    lat2 = math.radians(epicenter_lat)
    dlon = math.radians(epicenter_lon - reference.longitude)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def warning_time_at(
    target_lat: float,
    target_lon: float,
    location: LocationResult,
    vs_km_s: float = DEFAULT_VS_KM_S,
    processing_latency_s: float = 3.0,
) -> float:
    """估算某目标点能获得的预警时间（秒）。

    预警时间 = 破坏性 S 波到达目标点的时刻 − 系统发出预警的时刻。
      - S 波到达目标点时刻 = 发震时刻 + 震中到目标点距离 / Vs
      - 系统发出预警时刻 ≈ 发震时刻 + 处理延迟（拾取+定位+发布，这里给经验值）
    若结果为负，说明目标点太靠近震中，处于"预警盲区"（来不及预警）。

    这是把技术成果翻译成"为应急争取多少秒"的关键指标，答辩很有说服力。
    """
    if vs_km_s <= 0:
        raise ValueError("vs_km_s 必须为正")
    dist = haversine_km(
        location.latitude, location.longitude, target_lat, target_lon
    )
    s_arrival = location.origin_time_utc + dist / vs_km_s
    alert_time = location.origin_time_utc + processing_latency_s
    return s_arrival - alert_time
