"""EEW 展示层（定位 / 发震时刻 / 方位角 / 预警时间 / 震级）单元测试。

核心思路：**正演造真值，反演验精度**。
先人为设定一个"已知震中 + 已知发震时刻"，用物理公式正向算出各台站应当观测到的
P 到时（这就是无噪声的完美真值）；再把这些到时喂给定位模块，检查它能不能把
我们埋进去的震中和发震时刻反解回来。能反解回来，说明定位逻辑没写错。

两种运行方式：
    pytest tests/test_eew.py
    python  tests/test_eew.py     # 无 pytest 时 standalone
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from phasepicker.eew.locate import (
    Station,
    StationArrival,
    haversine_km,
    estimate_origin_and_location,
    estimate_back_azimuth,
    warning_time_at,
    DEFAULT_VP_KM_S,
)
from phasepicker.eew.magnitude import estimate_magnitude


def _forward_arrivals(true_lat, true_lon, true_t0, stations, vp):
    """正演：给定真震中/发震时刻，算各台站的完美 P 到时。"""
    arrivals = []
    for st in stations:
        d = haversine_km(true_lat, true_lon, st.latitude, st.longitude)
        arrivals.append(StationArrival(station=st, p_time_utc=true_t0 + d / vp))
    return arrivals


# ----------------------------- 距离基本量 -----------------------------

def test_haversine_zero():
    assert haversine_km(22.0, 108.0, 22.0, 108.0) == 0.0

def test_haversine_known_scale():
    # 纬度差 1 度 ≈ 111 km，容许 ±2km
    d = haversine_km(22.0, 108.0, 23.0, 108.0)
    assert abs(d - 111.19) < 2.0


# ----------------------------- 定位反演 -----------------------------

def _make_stations():
    # 围绕 (22.5, 108.5) 布 5 个台站
    return [
        Station("S1", 22.0, 108.0),
        Station("S2", 23.0, 108.0),
        Station("S3", 22.0, 109.0),
        Station("S4", 23.0, 109.0),
        Station("S5", 22.5, 108.5),
    ]

def test_location_recovers_known_epicenter():
    true_lat, true_lon, true_t0 = 22.4, 108.6, 1_600_000_000.0
    stations = _make_stations()
    arrivals = _forward_arrivals(true_lat, true_lon, true_t0, stations, DEFAULT_VP_KM_S)

    res = estimate_origin_and_location(arrivals, grid_step_deg=0.05)
    # 网格步长 0.05° ≈ 5.5km，加二级细化，反解震中应很接近真值
    assert abs(res.latitude - true_lat) < 0.1
    assert abs(res.longitude - true_lon) < 0.1
    # 发震时刻应基本还原
    assert abs(res.origin_time_utc - true_t0) < 1.0
    # 完美数据下 RMS 残差应接近 0
    assert res.rms_residual_s < 0.5

def test_location_requires_three_stations():
    stations = _make_stations()[:2]
    arrivals = _forward_arrivals(22.4, 108.6, 0.0, stations, DEFAULT_VP_KM_S)
    try:
        estimate_origin_and_location(arrivals)
        assert False, "少于3台站应抛错"
    except ValueError:
        pass

def test_location_rejects_bad_vp():
    stations = _make_stations()
    arrivals = _forward_arrivals(22.4, 108.6, 0.0, stations, DEFAULT_VP_KM_S)
    try:
        estimate_origin_and_location(arrivals, vp_km_s=0.0)
        assert False, "vp<=0 应抛错"
    except ValueError:
        pass


# ----------------------------- 方位角 -----------------------------

def test_back_azimuth_cardinal_directions():
    ref = Station("R", 22.0, 108.0)
    # 震中在正北 → 方位角≈0
    assert abs(estimate_back_azimuth(ref, 23.0, 108.0) - 0.0) < 2.0
    # 震中在正东 → 方位角≈90
    assert abs(estimate_back_azimuth(ref, 22.0, 109.0) - 90.0) < 2.0
    # 震中在正南 → 方位角≈180
    assert abs(estimate_back_azimuth(ref, 21.0, 108.0) - 180.0) < 2.0


# ----------------------------- 预警时间 -----------------------------

def test_warning_time_far_positive_near_negative():
    # 构造一个定位结果：震中在 (22.5,108.5)，发震时刻 0
    stations = _make_stations()
    arrivals = _forward_arrivals(22.5, 108.5, 0.0, stations, DEFAULT_VP_KM_S)
    loc = estimate_origin_and_location(arrivals, grid_step_deg=0.05)

    # 远处目标（约200km外）应有正的预警时间
    far = warning_time_at(24.5, 110.5, loc, processing_latency_s=3.0)
    # 极近目标（就在震中）应为负（预警盲区）
    near = warning_time_at(22.5, 108.5, loc, processing_latency_s=3.0)
    assert far > near
    assert near < 0.0  # 盲区


# ----------------------------- 震级（相对估计）-----------------------------

def test_magnitude_larger_amplitude_gives_larger_estimate():
    sr = 100.0
    n = 1000
    p_idx = 100
    # 两条信号：除幅值外完全相同，大幅值应给出更大的相对震级
    base = np.zeros(n)
    base[p_idx:p_idx + 300] = np.sin(np.linspace(0, 20, 300))
    small = estimate_magnitude(base * 1.0, p_idx, sr)
    large = estimate_magnitude(base * 10.0, p_idx, sr)
    assert large.magnitude > small.magnitude
    # 未提供系数 → 必须诚实标注未标定
    assert small.is_calibrated is False

def test_magnitude_handles_p_near_end():
    sr = 100.0
    v = np.ones(50)
    # P 到时在末尾，取不到窗口 → 返回 nan 而非崩溃
    res = estimate_magnitude(v, 49, sr)
    assert math.isnan(res.magnitude)


# ----------------------------- standalone runner -----------------------------

def _run_all():
    import io
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    lines = []
    for fn in fns:
        fn()
        lines.append(f"PASS {fn.__name__}")
    lines.append(f"SUMMARY {len(fns)}/{len(fns)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import base64
    out = _run_all()
    sys.stderr.write("B64:" + base64.b64encode(out.encode()).decode() + ":B64\n")
