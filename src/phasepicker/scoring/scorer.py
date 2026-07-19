"""本地评分脚本（Local scorer）——整个项目的"体温计"。

设计原则：
1. 严格复刻官方自动评分规则，作为阈值/参数调优的唯一客观依据。
2. 完全独立于 API 服务，可对一批样例快速跑批。
3. 无重依赖（仅 numpy），保证在任何机器上都能跑、都能测。

===== 官方评分规则（来自赛题） =====
P 波：误差 ≤0.1s 得 1 分；0.1s~1s 线性递减；≥1s 得 0 分。
S 波：误差 ≤0.2s 得 1 分；0.2s~2s 线性递减；≥2s 得 0 分。
数量误差：|预测数 - 真值数| / 真值数 ≤5% 不扣分；每超 1 个扣 0.5 分。

===== 预测↔真值 匹配算法（原方案缺失，此处补齐） =====
分数能不能算准，核心在于"把哪个预测配到哪个真值"。规则：
- 只在"同类型"（P 配 P，S 配 S）之间匹配。
- 采用"全局最小总误差"的贪心近似：把所有 (预测,真值) 候选对按时间差
  从小到大排序，依次认领，每个预测和每个真值最多被认领一次；只有落在
  该震相类型的"计分容差窗口上界"（P=1s, S=2s）内的候选对才允许匹配
  （超出上界得分必为 0，配了也没意义，还会污染数量统计）。
- 未被匹配的预测 = 误报（false positive）；未被匹配的真值 = 漏检
  （false negative）。二者都只通过"数量误差"项影响总分，不额外重复罚。

这是匈牙利算法的贪心近似。在震相稀疏、同类到时间隔远大于容差的地震场景
下，贪心解与最优解基本一致，且实现简单、行为可预测、便于团队理解。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 各震相类型的计分参数：(满分容差上界, 零分容差上界)
_PHASE_SCORING = {
    "P": (0.1, 1.0),
    "S": (0.2, 2.0),
}


def phase_time_score(residual_s: float, phase_type: str) -> float:
    """单个已匹配震相的到时得分（0~1），按官方线性递减规则。

    Args:
        residual_s: 到时绝对误差（秒），应 >= 0。
        phase_type: "P" 或 "S"。

    Returns:
        [0, 1] 区间得分。
    """
    if phase_type not in _PHASE_SCORING:
        raise ValueError(f"未知震相类型 {phase_type!r}，应为 'P' 或 'S'")
    full, zero = _PHASE_SCORING[phase_type]
    r = abs(residual_s)
    if r <= full:
        return 1.0
    if r >= zero:
        return 0.0
    # 在 (full, zero) 之间线性递减：full 处为 1，zero 处为 0
    return (zero - r) / (zero - full)


@dataclass
class MatchResult:
    """一次匹配的完整结果，供打分与误差分布统计使用。"""

    matched: List[Tuple[int, int, float]] = field(default_factory=list)
    """已匹配对列表：(预测下标, 真值下标, 到时残差秒)。"""
    unmatched_pred: List[int] = field(default_factory=list)
    """未匹配的预测下标（误报）。"""
    unmatched_true: List[int] = field(default_factory=list)
    """未匹配的真值下标（漏检）。"""


def match_phases(
    pred_times: Sequence[float],
    true_times: Sequence[float],
    phase_type: str,
) -> MatchResult:
    """同类型震相的贪心最小误差匹配。参见模块头部算法说明。

    Args:
        pred_times: 预测到时（Unix epoch 秒），仅含该类型。
        true_times: 真值到时（Unix epoch 秒），仅含该类型。
        phase_type: "P" 或 "S"，用于取容差上界。

    Returns:
        MatchResult。
    """
    if phase_type not in _PHASE_SCORING:
        raise ValueError(f"未知震相类型 {phase_type!r}")
    _, zero = _PHASE_SCORING[phase_type]

    # 构造所有落在容差上界内的候选对，按残差升序
    candidates: List[Tuple[float, int, int]] = []
    for i, pt in enumerate(pred_times):
        for j, tt in enumerate(true_times):
            r = abs(pt - tt)
            if r < zero:  # >= zero 得分必为 0，不纳入匹配
                candidates.append((r, i, j))
    candidates.sort(key=lambda x: x[0])

    used_pred: set[int] = set()
    used_true: set[int] = set()
    result = MatchResult()
    for r, i, j in candidates:
        if i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        result.matched.append((i, j, r))

    result.unmatched_pred = [i for i in range(len(pred_times)) if i not in used_pred]
    result.unmatched_true = [j for j in range(len(true_times)) if j not in used_true]
    return result


def count_error_penalty(n_pred: int, n_true: int) -> float:
    """数量误差扣分。误差率 ≤5% 不扣；每超 1 个扣 0.5 分。

    这里"超出个数"的定义：相对 5% 容许带边界，多出来的整数个震相。
    例如真值 100，5% 容许即 ±5 个；预测 108 → 超 3 个 → 扣 1.5 分。
    """
    if n_true == 0:
        # 真值为 0 的退化情况：任何预测都算超出，按预测数计罚
        return 0.5 * n_pred
    allowed = 0.05 * n_true
    diff = abs(n_pred - n_true)
    if diff <= allowed:
        return 0.0
    # 超出容许带的整数个数（向上取整以体现"每超1个"）
    excess = int(np.ceil(diff - allowed))
    return 0.5 * excess


@dataclass
class ScoreReport:
    """一个文件（或一批）的完整评分报告，可直接序列化进日志/文档。"""

    total_score: float
    p_time_score: float
    s_time_score: float
    count_penalty: float
    n_pred_p: int
    n_true_p: int
    n_pred_s: int
    n_true_s: int
    p_residuals: List[float]
    s_residuals: List[float]
    n_false_pos: int
    n_false_neg: int

    def summary(self) -> str:
        """人类可读的一行摘要，便于跑批时快速扫。"""
        def _stat(rs: List[float]) -> str:
            if not rs:
                return "n/a"
            a = np.asarray(rs)
            return f"mean={a.mean():.3f}s p50={np.median(a):.3f}s max={a.max():.3f}s"

        return (
            f"总分={self.total_score:.3f} | "
            f"P时序={self.p_time_score:.3f}({_stat(self.p_residuals)}) | "
            f"S时序={self.s_time_score:.3f}({_stat(self.s_residuals)}) | "
            f"数量罚={self.count_penalty:.2f} "
            f"[P {self.n_pred_p}/{self.n_true_p}, S {self.n_pred_s}/{self.n_true_s}] | "
            f"误报={self.n_false_pos} 漏检={self.n_false_neg}"
        )


def score_file(
    pred: Sequence[Tuple[str, float]],
    truth: Sequence[Tuple[str, float]],
) -> ScoreReport:
    """对单个文件评分。

    Args:
        pred: 预测震相列表，每项 (phase_type, time_utc_seconds)。
        truth: 真值震相列表，每项 (phase_type, time_utc_seconds)。

    Returns:
        ScoreReport。到时总分 = 各已匹配震相得分之和；数量罚单独计算后从中扣除，
        并 clip 到不小于 0（避免单文件出现负分干扰跨文件汇总口径）。
    """
    def _split(items: Sequence[Tuple[str, float]], t: str) -> List[float]:
        return [tt for pt, tt in items if pt == t]

    pred_p, pred_s = _split(pred, "P"), _split(pred, "S")
    true_p, true_s = _split(truth, "P"), _split(truth, "S")

    m_p = match_phases(pred_p, true_p, "P")
    m_s = match_phases(pred_s, true_s, "S")

    p_score = sum(phase_time_score(r, "P") for _, _, r in m_p.matched)
    s_score = sum(phase_time_score(r, "S") for _, _, r in m_s.matched)

    n_pred = len(pred_p) + len(pred_s)
    n_true = len(true_p) + len(true_s)
    penalty = count_error_penalty(n_pred, n_true)

    total = max(0.0, p_score + s_score - penalty)

    return ScoreReport(
        total_score=total,
        p_time_score=p_score,
        s_time_score=s_score,
        count_penalty=penalty,
        n_pred_p=len(pred_p),
        n_true_p=len(true_p),
        n_pred_s=len(pred_s),
        n_true_s=len(true_s),
        p_residuals=[r for _, _, r in m_p.matched],
        s_residuals=[r for _, _, r in m_s.matched],
        n_false_pos=len(m_p.unmatched_pred) + len(m_s.unmatched_pred),
        n_false_neg=len(m_p.unmatched_true) + len(m_s.unmatched_true),
    )
