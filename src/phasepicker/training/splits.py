"""无泄漏训练/验证切分（Leak-free train/val split）.

===== 为什么这是"防止本地分虚高"的关键（写给非 AI 背景的队友）=====
地震波形数据有一个隐蔽陷阱：**同一个地震事件、同一个台站的相邻窗口高度相似**。
如果随机按"窗口"切分，很可能把同一事件的两个几乎一样的窗口，一个分进训练集、
一个分进验证集。模型在训练集见过后，在验证集上自然"认得"，验证分虚高——
等到官方用全新数据评分时，分数直接跳水。这叫**数据泄漏（data leakage）**。

正确做法：**按"分组键"切分，而不是按单个样本切分。** 常见分组键优先级：
  event_id（同一次地震）> station（同一台站）> 文件名。
同一组的所有样本要么全在训练集、要么全在验证集，绝不跨越。

这个模块只做一件事，但做对：给定每个样本的分组键，输出无泄漏的
train/val 下标划分，且**由随机种子决定、完全可复现**。

无重依赖（仅标准库），因此可在任何机器上直接跑、直接测。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SplitResult:
    """一次切分的结果。"""

    train_idx: List[int]
    """训练集样本下标（相对输入序列）。"""
    val_idx: List[int]
    """验证集样本下标。"""
    train_groups: List[str]
    """落入训练集的分组键，便于人工核查无重叠。"""
    val_groups: List[str]
    """落入验证集的分组键。"""

    def assert_no_leak(self) -> None:
        """断言训练/验证的分组键零交集。切分后必调，作为最后一道保险。"""
        overlap = set(self.train_groups) & set(self.val_groups)
        if overlap:
            raise AssertionError(
                f"数据泄漏！以下分组同时出现在训练与验证集：{sorted(overlap)[:10]}"
            )

    def summary(self) -> str:
        n = len(self.train_idx) + len(self.val_idx)
        frac = len(self.val_idx) / n if n else 0.0
        return (
            f"切分：训练 {len(self.train_idx)} 样本/{len(self.train_groups)} 组，"
            f"验证 {len(self.val_idx)} 样本/{len(self.val_groups)} 组，"
            f"验证占比 {frac:.1%}（按组无泄漏）"
        )


def _stable_hash_fraction(key: str, seed: int) -> float:
    """把分组键 + 种子映射到 [0,1) 的稳定浮点数。

    用 md5 而非 Python 内置 hash()，因为内置 hash 对 str 会随进程随机化
    （PYTHONHASHSEED），无法跨机器/跨运行复现。md5 是确定性的。
    """
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    # 取前 8 位十六进制 → 整数 → 归一化到 [0,1)
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_group(
    group_keys: Sequence[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """按分组键做无泄漏切分，结果由种子唯一确定、完全可复现。

    实现要点——**哈希切分而非洗牌切分**：
    对每个不同的分组键算一个稳定哈希分数 h∈[0,1)，h < val_fraction 的组整体
    进验证集，其余进训练集。这样做的好处是：
      1) 无需洗牌，天然可复现，且与样本出现顺序无关；
      2) 未来数据增量时，老样本的归属不变（同一 key 哈希不变），
         便于持续迭代时保持验证集稳定，避免"换一批数据验证分就抖"。

    Args:
        group_keys: 与样本一一对应的分组键序列（如每个训练窗口所属的 event_id
            或 station）。长度即样本数。
        val_fraction: 验证集目标占比（按"组"近似，非按样本精确）。
        seed: 随机种子，决定切分。

    Returns:
        SplitResult。

    Raises:
        ValueError: 入参非法。
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction 必须在 (0,1)，收到 {val_fraction}")
    if len(group_keys) == 0:
        raise ValueError("group_keys 为空，无法切分")

    # 每个唯一组只算一次哈希，决定其整体归属
    unique_keys = sorted(set(group_keys))
    val_group_set = {
        k for k in unique_keys if _stable_hash_fraction(k, seed) < val_fraction
    }

    # 退化保护：若因组数太少导致某一侧为空，回退为"至少留一组给验证"
    if not val_group_set and unique_keys:
        # 选哈希分数最小的那组进验证，保证验证集非空
        smallest = min(unique_keys, key=lambda k: _stable_hash_fraction(k, seed))
        val_group_set = {smallest}
    if len(val_group_set) == len(unique_keys):
        # 全进了验证集：回退为把哈希分数最大的一组留给训练
        largest = max(unique_keys, key=lambda k: _stable_hash_fraction(k, seed))
        val_group_set.discard(largest)

    train_idx, val_idx = [], []
    for i, k in enumerate(group_keys):
        (val_idx if k in val_group_set else train_idx).append(i)

    train_groups = sorted(set(group_keys) - val_group_set)
    val_groups = sorted(val_group_set)

    result = SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        train_groups=train_groups,
        val_groups=val_groups,
    )
    result.assert_no_leak()  # 自检，绝不返回带泄漏的结果
    return result


def kfold_by_group(
    group_keys: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[SplitResult]:
    """按组的 K 折交叉验证切分（可选，用于数据量较小时更稳的评估）。

    同样保证：同一分组键的样本永远在同一折内，跨折不泄漏。
    每折的验证集 = 该折分到的组；训练集 = 其余所有组。

    数据量小的时候（比如主办方只给几百条标注），K 折能显著降低
    "验证集偶然好/坏"的方差，评估更可信——这正是我们本地评分脚本
    做阈值搜索时想要的稳定性。
    """
    if n_folds < 2:
        raise ValueError(f"n_folds 必须 >= 2，收到 {n_folds}")

    unique_keys = sorted(set(group_keys))
    if len(unique_keys) < n_folds:
        raise ValueError(
            f"唯一分组数 {len(unique_keys)} 少于折数 {n_folds}，无法切分"
        )

    # 按稳定哈希把每个组分配到 0..n_folds-1 号折
    fold_of_key: Dict[str, int] = {}
    for k in unique_keys:
        bucket = int(_stable_hash_fraction(k, seed) * n_folds)
        fold_of_key[k] = min(bucket, n_folds - 1)  # 防边界 ==1.0

    results: List[SplitResult] = []
    for f in range(n_folds):
        val_group_set = {k for k, fold in fold_of_key.items() if fold == f}
        if not val_group_set:
            # 某折恰好没分到组，跳过（组数接近折数时可能发生）
            continue
        train_idx, val_idx = [], []
        for i, k in enumerate(group_keys):
            (val_idx if k in val_group_set else train_idx).append(i)
        res = SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            train_groups=sorted(set(group_keys) - val_group_set),
            val_groups=sorted(val_group_set),
        )
        res.assert_no_leak()
        results.append(res)
    return results
