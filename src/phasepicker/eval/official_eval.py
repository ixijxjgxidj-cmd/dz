"""官方任务评估（Official task evaluation）——把预测与官方答案比对出分.

三个任务各有各的评估口径：
- T1：复用现有官方评分器 scoring.scorer.score_file（P/S 到时得分 + 数量罚）。
  这里只做"结构转换"：把 Task1Result 的 P/S 列表拍平成 score_file 认识的
  ``[("P", t), ("S", t), ...]``，然后逐文件汇总。
- T2：震级回归，报 MAE（平均绝对误差）+ 已匹配数 + 缺失/多余数。
- T3：事件分类，报 accuracy + 混淆矩阵 + 缺失/多余数。

"缺失/多余"指预测集合与答案集合的 file_id 差集：answer 有而 pred 没有 = missing；
pred 有而 answer 没有 = extra。只对交集文件计入分数，避免键不齐时口径混乱。

返回普通 dataclass（字段皆为内置类型），便于测试断言与直接 print。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..types import Task1Result, Task2Result, Task3Result
from ..scoring.scorer import score_file


def _common_keys(pred: Dict[str, object], truth: Dict[str, object]) -> Tuple[List[str], List[str], List[str]]:
    """返回 (共有键 已排序, 缺失键 truth有pred无, 多余键 pred有truth无)。"""
    pk, tk = set(pred), set(truth)
    common = sorted(pk & tk)
    missing = sorted(tk - pk)
    extra = sorted(pk - tk)
    return common, missing, extra


# ============================ T1 ============================

@dataclass
class Task1EvalReport:
    """T1 评估汇总。"""

    total_score: float
    mean_score: float
    n_files: int
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    per_file: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"T1: 均分={self.mean_score:.3f} 总分={self.total_score:.3f} "
            f"文件数={self.n_files} 缺失={len(self.missing)} 多余={len(self.extra)}"
        )


def _task1_to_pairs(r: Task1Result) -> List[Tuple[str, float]]:
    """Task1Result → score_file 认识的 [("P", t), ("S", t), ...]。"""
    pairs: List[Tuple[str, float]] = [("P", float(t)) for t in r.p_times_s]
    pairs += [("S", float(t)) for t in r.s_times_s]
    return pairs


def evaluate_task1(
    predictions: Dict[str, Task1Result],
    answers: Dict[str, Task1Result],
) -> Task1EvalReport:
    """对 T1 逐文件评分并汇总。仅对共有 file_id 计分。"""
    common, missing, extra = _common_keys(predictions, answers)
    per_file: Dict[str, float] = {}
    total = 0.0
    for fid in common:
        rep = score_file(_task1_to_pairs(predictions[fid]), _task1_to_pairs(answers[fid]))
        per_file[fid] = rep.total_score
        total += rep.total_score
    n = len(common)
    return Task1EvalReport(
        total_score=total,
        mean_score=(total / n) if n else 0.0,
        n_files=n,
        missing=missing,
        extra=extra,
        per_file=per_file,
    )


# ============================ T2 ============================

@dataclass
class Task2EvalReport:
    """T2 评估汇总（震级 MAE）。"""

    mae: float
    count: int
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    max_abs_error: float = 0.0

    def summary(self) -> str:
        return (
            f"T2: MAE={self.mae:.4f} maxAE={self.max_abs_error:.4f} "
            f"count={self.count} 缺失={len(self.missing)} 多余={len(self.extra)}"
        )


def evaluate_task2(
    predictions: Dict[str, Task2Result],
    answers: Dict[str, Task2Result],
) -> Task2EvalReport:
    """对 T2 计算 MAE、最大绝对误差、计数与缺失/多余。"""
    common, missing, extra = _common_keys(predictions, answers)
    abs_errors: List[float] = []
    for fid in common:
        err = abs(float(predictions[fid].magnitude) - float(answers[fid].magnitude))
        abs_errors.append(err)
    count = len(abs_errors)
    mae = (sum(abs_errors) / count) if count else 0.0
    return Task2EvalReport(
        mae=mae,
        count=count,
        missing=missing,
        extra=extra,
        max_abs_error=max(abs_errors) if abs_errors else 0.0,
    )


# ============================ T3 ============================

@dataclass
class Task3EvalReport:
    """T3 评估汇总（分类 accuracy + 混淆矩阵）。"""

    accuracy: float
    correct: int
    count: int
    confusion: Dict[Tuple[int, int], int] = field(default_factory=dict)
    """混淆矩阵：{(真值类别, 预测类别): 计数}。"""
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"T3: acc={self.accuracy:.4f} ({self.correct}/{self.count}) "
            f"缺失={len(self.missing)} 多余={len(self.extra)}"
        )


def evaluate_task3(
    predictions: Dict[str, Task3Result],
    answers: Dict[str, Task3Result],
) -> Task3EvalReport:
    """对 T3 计算 accuracy、混淆矩阵与缺失/多余。"""
    common, missing, extra = _common_keys(predictions, answers)
    confusion: Dict[Tuple[int, int], int] = {}
    correct = 0
    for fid in common:
        true_label = int(answers[fid].label)
        pred_label = int(predictions[fid].label)
        key = (true_label, pred_label)
        confusion[key] = confusion.get(key, 0) + 1
        if true_label == pred_label:
            correct += 1
    count = len(common)
    return Task3EvalReport(
        accuracy=(correct / count) if count else 0.0,
        correct=correct,
        count=count,
        confusion=confusion,
        missing=missing,
        extra=extra,
    )
