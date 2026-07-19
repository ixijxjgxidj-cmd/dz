"""EEW 展示层端到端演示（决赛答辩用）.

用法::

    python scripts/run_eew_demo.py

这个脚本不依赖任何真实数据，用一组"自备台站 + 已知震中"的演示数据，
把决赛展示层的完整链路跑一遍并打印出来：
    多台站 P 到时 → 网格搜索定位 + 发震时刻 → 震中方位 → 各城市预警时间 → 相对震级

科学口径（答辩必须守住）：
    这是"震后秒级快速监测预警"演示，不是"震前预测"。
    赛题波形不含位置信息，所以定位/震级用的是自备台站表与演示数据，
    用于证明"方法可行、系统完整"，而非声称单条无位置信息波形能算震中。
"""

import os
import sys

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


def build_demo_arrivals():
    """用已知震中正演出各台站 P 到时（模拟初赛拾取模块的输出）。"""
    true_lat, true_lon, true_t0 = 22.8, 108.3, 1_600_000_000.0
    stations = [
        Station("NN01", 22.0, 108.0),
        Station("NN02", 23.2, 108.1),
        Station("NN03", 22.5, 109.0),
        Station("NN04", 23.5, 109.2),
        Station("NN05", 22.9, 108.4),
    ]
    arrivals = []
    for st in stations:
        d = haversine_km(true_lat, true_lon, st.latitude, st.longitude)
        # 加一点点噪声，模拟真实拾取误差（±0.05s）
        noise = np.random.RandomState(abs(hash(st.code)) % 2**32).normal(0, 0.05)
        arrivals.append(
            StationArrival(station=st, p_time_utc=true_t0 + d / DEFAULT_VP_KM_S + noise)
        )
    return true_lat, true_lon, true_t0, stations, arrivals


def main():
    print("=" * 70)
    print("地震早期预警(EEW)展示层 — 端到端演示")
    print("科学口径：震后秒级快速监测预警，非震前预测；定位/震级用自备台站演示数据")
    print("=" * 70)

    true_lat, true_lon, true_t0, stations, arrivals = build_demo_arrivals()
    print(f"\n[真值] 震中=({true_lat}, {true_lon})  发震时刻(epoch)={true_t0}")
    print(f"[输入] {len(arrivals)} 个台站的 P 到时（来自震相拾取模块）：")
    for a in arrivals:
        print(f"   {a.station.code}: P到时={a.p_time_utc:.3f}s "
              f"@({a.station.latitude},{a.station.longitude})")

    # 1) 定位 + 发震时刻
    loc = estimate_origin_and_location(arrivals, grid_step_deg=0.05)
    print("\n[定位结果] " + loc.summary())
    print(f"   定位误差：Δlat={abs(loc.latitude-true_lat):.3f}° "
          f"Δlon={abs(loc.longitude-true_lon):.3f}° "
          f"Δt0={abs(loc.origin_time_utc-true_t0):.3f}s")

    # 2) 震中方位（从第一个台站看）
    baz = estimate_back_azimuth(stations[0], loc.latitude, loc.longitude)
    print(f"\n[震中方向] 从台站 {stations[0].code} 看，震中方位角≈{baz:.1f}°（正北为0，顺时针）")

    # 3) 各目标城市的预警时间
    print("\n[预警时间] 破坏性 S 波到达前，各地可获得的预警窗口：")
    targets = {"目标城市A(近)": (22.9, 108.4), "目标城市B(中)": (24.0, 109.5),
               "目标城市C(远)": (25.5, 110.8)}
    for name, (tlat, tlon) in targets.items():
        wt = warning_time_at(tlat, tlon, loc, processing_latency_s=3.0)
        dist = haversine_km(loc.latitude, loc.longitude, tlat, tlon)
        status = f"可预警 {wt:.1f}s" if wt > 0 else "预警盲区(来不及)"
        print(f"   {name}: 震中距≈{dist:.0f}km → {status}")

    # 4) 相对震级（演示：合成一条 Z 分量速度波形）
    sr = 100.0
    n = 3000
    p_idx = 500
    wave = np.zeros(n)
    wave[p_idx:p_idx + 400] = np.sin(np.linspace(0, 30, 400)) * 5.0  # 模拟 P 波后振动
    dist_km = haversine_km(loc.latitude, loc.longitude,
                           stations[0].latitude, stations[0].longitude)
    mag = estimate_magnitude(wave, p_idx, sr, epicentral_distance_km=dist_km)
    print("\n[震级估计] " + mag.summary())
    print("   注：未标定系数，数值仅供相对比较；接入真实台网标定后可产出绝对震级。")

    print("\n" + "=" * 70)
    print("演示完成。以上链路即决赛'系统完整性'的核心叙事：")
    print("P/S拾取(初赛评分) → 多台站定位 → 发震时刻 → 方向 → 预警时间 → 震级")
    print("=" * 70)


if __name__ == "__main__":
    main()
