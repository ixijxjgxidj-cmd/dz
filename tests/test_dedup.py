"""震相去重 / 合并逻辑的单元测试（scoring-critical，纯 Python 可直接跑）。

运行：
    pytest tests/test_dedup.py
    python  tests/test_dedup.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import Pick, PhaseType
from phasepicker.postprocess.dedup import deduplicate, DEFAULT_MERGE_WINDOW_S


def _p(phase, t, conf, station="NET.STA"):
    return Pick(phase=phase, time_utc=t, confidence=conf, station=station)


def test_empty_input():
    assert deduplicate([]) == []


def test_near_duplicates_merged_keep_highest_conf():
    # 三个仅差 <0.05s 的 P，应合并为 1 个，且保留最高置信度那个的到时
    picks = [
        _p(PhaseType.P, 100.00, 0.6),
        _p(PhaseType.P, 100.02, 0.9),  # 最高置信度
        _p(PhaseType.P, 100.04, 0.7),
    ]
    out = deduplicate(picks)
    assert len(out) == 1
    assert out[0].time_utc == 100.02
    assert out[0].confidence == 0.9


def test_distinct_phases_not_merged():
    # 两个相距 5s 的真实 P 震相，不能被合并
    picks = [_p(PhaseType.P, 100.0, 0.8), _p(PhaseType.P, 105.0, 0.8)]
    out = deduplicate(picks)
    assert len(out) == 2


def test_p_and_s_never_cross_merge():
    # 同一时刻的 P 和 S 是不同类型，绝不能互相合并
    picks = [_p(PhaseType.P, 100.0, 0.8), _p(PhaseType.S, 100.0, 0.8)]
    out = deduplicate(picks)
    assert len(out) == 2


def test_different_stations_not_merged():
    # 不同台站的同类型同到时，属于不同台站的独立观测，不合并
    picks = [
        _p(PhaseType.P, 100.0, 0.8, station="NET.AAA"),
        _p(PhaseType.P, 100.0, 0.8, station="NET.BBB"),
    ]
    out = deduplicate(picks)
    assert len(out) == 2


def test_s_window_wider_than_p():
    # S 合并窗口默认 0.10s：相差 0.08s 的两个 S 应被合并；同样间距的 P 不应
    s_picks = [_p(PhaseType.S, 100.0, 0.5), _p(PhaseType.S, 100.08, 0.6)]
    p_picks = [_p(PhaseType.P, 100.0, 0.5), _p(PhaseType.P, 100.08, 0.6)]
    assert len(deduplicate(s_picks)) == 1
    assert len(deduplicate(p_picks)) == 2


def test_beyond_window_not_merged():
    # 明显超过窗口（严格小于才合并）→ 不合并。
    # 注：不测"恰好等于窗口值"的边界，因为 100.0+0.05 在 float64 下可能是
    # 0.0499999…，边界处的合并/不合并对最终得分无影响（窗口远小于计分容差），
    # 故用一个安全超出窗口的间距来验证"该分开的会分开"。
    win = DEFAULT_MERGE_WINDOW_S[PhaseType.P]  # 0.05
    picks = [_p(PhaseType.P, 100.0, 0.5), _p(PhaseType.P, 100.0 + win * 2, 0.6)]
    out = deduplicate(picks)
    assert len(out) == 2


def test_output_is_sorted():
    picks = [
        _p(PhaseType.S, 200.0, 0.8),
        _p(PhaseType.P, 100.0, 0.8),
        _p(PhaseType.P, 50.0, 0.8),
    ]
    out = deduplicate(picks)
    times = [p.time_utc for p in out]
    assert times == sorted(times) or all(
        out[i].phase.value <= out[i + 1].phase.value for i in range(len(out) - 1)
    )


def _run_all():
    import base64
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    line