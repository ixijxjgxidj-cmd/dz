"""时间对齐 + 评分逻辑的单元测试（两大胜负手的守门测试）。

可用两种方式运行：
    pytest tests/test_timing_and_scoring.py        # 有 pytest 时
    python  tests/test_timing_and_scoring.py       # 无 pytest 时，standalone
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.utils.timing import sample_to_utc, utc_to_sample, resample_index
from phasepicker.scoring.scorer import (
    phase_time_score,
    match_phases,
    count_error_penalty,
    score_file,
)

# ----------------------- 时间对齐 -----------------------

def test_sample_to_utc_basic():
    # 起点 1000.0s，100Hz，第 250 个采样点 → 1000 + 2.5 = 1002.5
    assert sample_to_utc(250, 1000.0, 100.0) == 1002.5

def test_sample_to_utc_zero_index():
    # 第 0 个采样点必须精确等于起点，不能有偏移
    assert sample_to_utc(0, 1_600_000_000.0, 100.0) == 1_600_000_000.0

def test_roundtrip_utc_sample():
    # 往返一致性：sample -> utc -> sample 应还原。
    #
    # ⚠️ 精度说明（重要）：以 Unix epoch 秒（~1.6e9）存绝对时间时，float64 在该
    # 量级的分辨率约 2.4e-7s（~240ns）。往返 + 乘以采样率会把该舍入放大到 ~1e-5s
    # 级别。这远小于计分容差（P=0.1s，S=0.2s），有约 4 个数量级的安全裕度，
    # 因此用 float64 epoch 秒做绝对到时对"得分"是安全的。
    # 但这提醒我们：残差类计算应尽量在"相对某参考点的偏移"坐标下做，别在 1.6e9
    # 量级上反复加减小量。测试容差按真实数值行为设为 1e-4s。
    st, sr = 1_600_000_000.0, 100.0
    for idx in [0, 1, 37, 1000, 999_999]:
        t = sample_to_utc(idx, st, sr)
        back = utc_to_sample(t, st, sr)
        assert abs(back - idx) < 1e-4, (idx, back)


def test_relative_offset_is_exact():
    # 对照：若以"相对波形起点的偏移秒"为参考（起点=0），float64 精度绰绰有余，
    # 往返可精确还原。这是推荐的内部计算坐标系。
    sr = 100.0
    for idx in [0, 1, 37, 1000, 999_999]:
        t = sample_to_utc(idx, 0.0, sr)
        back = utc_to_sample(t, 0.0, sr)
        assert abs(back - idx) < 1e-9, (idx, back)

def test_subsample_precision():
    # 亚采样点峰值：250.5 个点 @100Hz = 2.505s
    assert abs(sample_to_utc(250.5, 0.0, 100.0) - 2.505) < 1e-12

def test_resample_index():
    # 100Hz 下第 250 点，对应 50Hz 下第 125 点
    assert resample_index(250, 100.0, 50.0) == 125.0

def test_invalid_sampling_rate():
    for bad in (0, -1, -100.0):
        try:
            sample_to_utc(10, 0.0, bad)
            raise AssertionError("应对非正采样率抛错")
        except ValueError:
            pass

# ----------------------- 到时得分 -----------------------

def test_p_full_score_boundary():
    assert phase_time_score(0.0, "P") == 1.0
    assert phase_time_score(0.1, "P") == 1.0           # 边界含
    assert phase_time_score(0.10001, "P") < 1.0        # 刚过边界即递减

def test_p_zero_score():
    assert phase_time_score(1.0, "P") == 0.0
    assert phase_time_score(2.0, "P") == 0.0

def test_p_linear_midpoint():
    # P 波 0.1~1.0 线性；中点 0.55 应得 0.5
    assert abs(phase_time_score(0.55, "P") - 0.5) < 1e-9

def test_s_full_and_linear():
    assert phase_time_score(0.2, "S") == 1.0
    assert phase_time_score(2.0, "S") == 0.0
    # S 波 0.2~2.0 线性；中点 1.1 应得 0.5
    assert abs(phase_time_score(1.1, "S") - 0.5) < 1e-9

def test_symmetric_residual():
    # 正负残差应对称
    assert phase_time_score(-0.55, "P") == phase_time_score(0.55, "P")

# ----------------------- 匹配算法 -----------------------

def test_match_greedy_min_error():
    # 两个预测抢一个真值：应由更近的那个认领
    pred = [10.00, 10.30]
    true = [10.05]
    m = match_phases(pred, true, "P")
    assert len(m.matched) == 1
    i, j, r = m.matched[0]
    assert i == 0 and abs(r - 0.05) < 1e-9
    assert m.unmatched_pred == [1]      # 另一个成为误报

def test_match_beyond_tolerance_not_matched():
    # 残差 >= 零分上界的对不允许匹配
    pred = [100.0]
    true = [102.0]  # P 波 2s > 1s 上界
    m = match_phases(pred, true, "P")
    assert m.matched == []
    assert m.unmatched_pred == [0] and m.unmatched_true == [0]

def test_match_multiple_pairs():
    pred = [10.0, 20.0, 30.0]
    true = [10.02, 20.05, 30.5]
    m = match_phases(pred, true, "P")
    assert len(m.matched) == 3
    assert m.unmatched_pred == [] and m.unmatched_true == []

# ----------------------- 数量罚 -----------------------

def test_count_penalty_within_tolerance():
    # 真值100，5%容许=5个；预测105 → 不扣
    assert count_error_penalty(105, 100) == 0.0
    assert count_error_penalty(95, 100) == 0.0

def test_count_penalty_excess():
    # 真值100，预测108 → 超容许带3个 → 扣1.5
    assert count_error_penalty(108, 100) == 1.5

def test_count_penalty_true_zero():
    assert count_error_penalty(3, 0) == 1.5
    assert count_error_penalty(0, 0) == 0.0

# ----------------------- 端到端评分 -----------------------

def test_score_file_perfect():
    truth = [("P", 100.0), ("S", 105.0)]
    pred = [("P", 100.0), ("S", 105.0)]
    rep = score_file(pred, truth)
    assert abs(rep.total_score - 2.0) < 1e-9
    assert rep.n_false_pos == 0 and rep.n_false_neg == 0

def test_score_file_partial_and_penalty():
    # P 偏 0.55s(得0.5)，S 完美(得1.0)，且多报一个 P 造成数量误差
    truth = [("P", 100.0), ("S", 105.0)]
    pred = [("P", 100.55), ("S", 105.0), ("P", 999.0)]
    rep = score_file(pred, truth)
    # 时