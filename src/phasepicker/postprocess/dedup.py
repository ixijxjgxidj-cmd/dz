"""震相去重 / 合并（Deduplication & merging）.

===== 为什么需要这一步（写给非 AI 背景的队友）=====
逐窗（sliding-window）推理时，同一个真实震相可能被相邻窗口重复检测，
输出两三个仅差几十毫秒的到时。若直接上报，会同时踩两个坑：
  1) 数量误差超标——被"数量误差每超 1 个扣 0.5 分"的规则狠罚；
  2) 同一震相占用多个名额，挤占匹配、拉低整体表现。

因此上报前必须把"同台站、同类型、时间靠得很近"的多个 pick 合并为一个，
并保留其中置信度最高的那个到时。这是保护分数的关键一环。

===== 合并策略 =====
对每个 (station, phase) 分组，按到时排序，用"贪心聚类"：
相邻两个 pick 时间差 < merge_window_s 则并入同一簇；每个簇最终只输出
一个代表 pick——默认取簇内 **置信度最高** 者（而非平均），因为峰值位置
通常比平均更接近真实到时，且与模型的概率输出一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..types import Pick, PhaseType

# 默认合并窗口：P/S 分别取远小于各自计分容差（0.1s / 0.2s）的值，
# 保证"该合并的合并了，不该合并的（真的是两个震相）不会被误并"。
DEFAULT_MERGE_WINDOW_S = {
    PhaseType.P: 0.05,
    PhaseType.S: 0.10,
}


def _cluster_reduce(
    picks: List[Pick],
    merge_window_s: float,
) -> List[Pick]:
    """对已按 time_utc 升序排列的同组 picks 做贪心聚类，每簇取置信度最高者。"""
    if not picks:
        return []
    representatives: List[Pick] = []
    cluster: List[Pick] = [picks[0]]
    for p in picks[1:]:
        if p.time_utc - cluster[-1].time_utc < merge_window_s:
            cluster.append(p)
        else:
            representatives.append(max(cluster, key=lambda x: x.confidence))
            cluster = [p]
    representatives.append(max(cluster, key=lambda x: x.confidence))
    return representatives


def deduplicate(
    picks: Sequence[Pick],
    merge_window_s: dict | None = None,
) -> List[Pick]:
    """对一批 picks 去重合并。

    Args:
        picks: 待去重的震相列表（可跨台站、跨类型混合）。
        merge_window_s: 每种震相的合并窗口（秒），缺省用 DEFAULT_MERGE_WINDOW_S。

    Returns:
        去重后的 picks，按 (station, phase, time_utc) 稳定排序，便于复现与调试。
    """
    windows = dict(DEFAULT_MERGE_WINDOW_S)
    if merge_window_s:
        windows.update(merge_window_s)

    # 按 (station, phase) 分组
    groups: dict[tuple, List[Pick]] = {}
    for p in picks:
        groups.setdefault((p.station or "", p.phase), []).append(p)

    out: List[Pick] = []
    for (station, phase), group in groups.items():
        group.sort(key=lambda x: x.time_utc)
        win = windows.get(phase, 0.05)
        out.extend(_cluster_reduce(group, win))

    out.sort(key=lambda x: (x.station or "", x.phase.value, x.time_utc))
    return out


@dataclass
class DedupConfig:
    """去重配置对象（供推理入口 picker.py 注入）。

    把 P/S 各自的合并窗口收进一个配置对象，是为了让 SeisBenchPicker 能在构造时
    携带一份可配置的去重策略，而不必每次调用都传散参。字段与 DEFAULT_MERGE_WINDOW_S
    一一对应，缺省即沿用模块默认值，保证行为与旧代码完全一致。

    Attributes:
        p_merge_window_s: P 波合并窗口（秒）。
        s_merge_window_s: S 波合并窗口（秒）。
    """

    p_merge_window_s: float = DEFAULT_MERGE_WINDOW_S[PhaseType.P]
    s_merge_window_s: float = DEFAULT_MERGE_WINDOW_S[PhaseType.S]

    def to_windows(self) -> dict:
        """转成 deduplicate() 认识的 {PhaseType: 窗口秒} 映射。"""
        return {
            PhaseType.P: self.p_merge_window_s,
            PhaseType.S: self.s_merge_window_s,
        }


def dedup_picks(
    picks: Sequence[Pick],
    cfg: Optional[DedupConfig] = None,
) -> List[Pick]:
    """去重合并的配置化包装。

    这是 deduplicate() 的薄封装：cfg 为 None 时行为与 deduplicate(picks) 完全一致；
    给定 cfg 时用其窗口覆盖默认值。推理入口（inference/picker.py）用它，从而把
    "去重策略"变成可注入的配置项，同时不改动 deduplicate() 的原有语义。

    Args:
        picks: 待去重的震相列表。
        cfg: 去重配置；None 时使用模块默认窗口。

    Returns:
        去重后的 picks（排序规则同 deduplicate）。
    """
    windows = cfg.to_windows() if cfg is not None else None
    return deduplicate(picks, merge_window_s=windows)
