"""训练数据集（Dataset）—— 把"波形 + 标注"变成模型能吃的 (输入, 目标) 张量。

===== 写给非 AI 背景的队友 =====
PhaseNet 是"逐采样点分类"模型：它不直接输出"P 波在第几秒"，而是对波形上
每一个采样点，输出三条概率曲线——"这里是 P""这里是 S""这里是噪声"。
训练时，我们要把标注里的"P 波到时"转成这样一条概率曲线作为学习目标：
在到时那个采样点附近，用一个窄高斯"鼓包"表示"这里很可能是 P"。

所以本模块干两件事：
  1) 把每个波形切成固定长度的窗口（window_samples），并保证窗口里包含到时；
  2) 为每个窗口生成 (3, N) 的输入 和 (3, N) 的目标概率图（P/S/噪声三通道）。

⚠️ 依赖 torch。本文件在无 torch 的环境无法运行，属于"在你机器上跑"的部分。
纯逻辑（高斯标签生成、窗口裁剪的下标计算）已抽成不依赖 torch 的纯函数，
并在 tests 里单独验证，避免"标签错位"这类隐蔽 bug（它和推理侧的时间对齐
是同一类高危问题：目标偏一个采样点，模型就学歪了）。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .label_adapter import LabelSet
from ..utils.timing import utc_to_sample


# ----------------------------------------------------------------------------
# 纯逻辑（不依赖 torch）——可单测
# ----------------------------------------------------------------------------

def gaussian_label(
    n_samples: int,
    center_samples: Sequence[float],
    sigma_samples: float,
) -> np.ndarray:
    """在长度 n_samples 的一维轴上，为每个到时中心生成高斯"鼓包"，取逐点最大值叠加。

    这是 PhaseNet 系列的标准做法：用窄高斯而非单点 one-hot，能让模型对
    "差几个采样点"更鲁棒，也更好收敛。

    Args:
        n_samples: 窗口长度（采样点）。
        center_samples: 若干到时中心（采样点下标，可为 float 亚采样点精度）。
        sigma_samples: 高斯标准差（采样点）。约 10~20 对 100Hz 常用（即 0.1~0.2s）。

    Returns:
        形状 (n_samples,) 的 float32 概率轴，峰值为 1。
    """
    axis = np.zeros(n_samples, dtype=np.float32)
    if sigma_samples <= 0:
        raise ValueError("sigma_samples 必须为正")
    idx = np.arange(n_samples, dtype=np.float32)
    for c in center_samples:
        if c < 0 or c >= n_samples:
            continue  # 中心落在窗口外则跳过（该到时不在此窗口）
        bump = np.exp(-0.5 * ((idx - float(c)) / sigma_samples) ** 2)
        axis = np.maximum(axis, bump)
    return axis


def make_target(
    n_samples: int,
    p_centers: Sequence[float],
    s_centers: Sequence[float],
    sigma_samples: float,
) -> np.ndarray:
    """生成 PhaseNet 的三通道目标概率图：[P, S, 噪声]。

    噪声通道 = 1 - P - S（逐点 clip 到 [0,1]），保证三通道构成对每个采样点的
    软分类分布。通道顺序 [P, S, N] 与 SeisBench PhaseNet 的输出约定一致。

    Returns:
        形状 (3, n_samples) 的 float32 数组。
    """
    p = gaussian_label(n_samples, p_centers, sigma_samples)
    s = gaussian_label(n_samples, s_centers, sigma_samples)
    noise = np.clip(1.0 - p - s, 0.0, 1.0).astype(np.float32)
    return np.stack([p, s, noise], axis=0)


def compute_window_bounds(
    total_samples: int,
    window_samples: int,
    anchor_sample: float,
    jitter: int = 0,
    rng: "np.random.Generator | None" = None,
) -> Tuple[int, int]:
    """计算一个包含 anchor（某到时）的窗口 [start, end)。

    保证：窗口长度恰为 window_samples；anchor 落在窗口内；不越界。
    jitter>0 时在合法范围内随机平移窗口起点（数据增强：让到时不总在窗口正中，
    提升模型对到时位置的鲁棒性）。

    Returns:
        (start, end)，end - start == window_samples（当 total >= window 时）。
    """
    if window_samples >= total_samples:
        return 0, total_samples  # 波形比窗口还短，整段返回（调用方负责 pad）

    # 理想：anchor 居中
    ideal_start = int(round(anchor_sample - window_samples / 2))
    if jitter > 0:
        r = rng if rng is not None else np.random.default_rng()
        ideal_start += int(r.integers(-jitter, jitter + 1))

    # 夹紧到合法区间，保证 anchor 仍在 [start, start+window) 内
    max_start = total_samples - window_samples
    lo = max(0, int(math.ceil(anchor_sample)) - window_samples + 1)
    hi = min(max_start, int(math.floor(anchor_sample)))
    start = max(lo, min(ideal_start, hi))
    start = max(0, min(start, max_start))
    return start, start + window_samples


# ----------------------------------------------------------------------------
# torch Dataset（在你机器上跑）
# ----------------------------------------------------------------------------

def build_torch_dataset(*args, **kwargs):  # pragma: no cover - 需要 torch
    """惰性构造 torch Dataset，避免在无 torch 环境 import 即失败。

    真正实现放在内部函数里，只有调用时才 import torch。
    """
    import torch
    from torch.utils.data import Dataset

    class PhaseDataset(Dataset):
        """把 (Waveform, LabelSet) 列表转成训练样本。

        每个样本 = 一个包含某到时的窗口 + 其三通道高斯目标。
        一个波形可能产出多个窗口（每个到时一个），实现"数据放大"。
        """

        def __init__(
            self,
            waveforms: list,            # List[phasepicker.types.Waveform]
            labelsets: list,           # 与 waveforms 对齐的 List[LabelSet]
            window_samples: int,
            sigma_samples: float,
            augment: bool = True,
            seed: int = 42,
        ) -> None:
            assert len(waveforms) == len(labelsets), "波形与标注数量必须一致"
            self.window_samples = window_samples
            self.sigma_samples = sigma_samples
            self.augment = augment
            self._rng = np.random.default_rng(seed)
            # 预先把 (波形下标, 到时采样点, [P中心], [S中心]) 展开成样本索引
            self._index: List[Tuple[int, float]] = []
            self._wf = waveforms
            self._ls = labelsets
            for wi, (wf, ls) in enumerate(zip(waveforms, labelsets)):
                for p in ls.picks:
                    anchor = utc_to_sample(p.time_utc, wf.starttime_utc, wf.sampling_rate)
                    if 0 <= anchor < wf.n_samples:
                        self._index.append((wi, anchor))

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, i: int):
            wi, anchor = self._index[i]
            wf, ls = self._wf[wi], self._ls[wi]
            jitter = int(0.2 * self.window_samples) if self.augment else 0
            start, end = compute_window_bounds(
                wf.n_samples, self.window_samples, anchor, jitter, self._rng
            )
            x = np.asarray(wf.data[:, start:end], dtype=np.float32)
            # 逐通道归一化（去均值、除以标准差），与推理预处理保持一致口径
            x = _normalize_window(x)

            # 收集落在该窗口内的所有 P/S 到时（可能不止 anchor 一个）
            p_centers, s_centers = [], []
            for p in ls.picks:
                a = utc_to_sample(p.time_utc, wf.starttime_utc, wf.sampling_rate) - start
                if 0 <= a < (end - start):
                    (p_centers if p.phase == "P" else s_centers).append(a)
            y = make_target(end - start, p_centers, s_centers, self.sigma_samples)

            # pad 到固定长度（波形短于窗口时）
            if x.shape[-1] < self.window_samples:
                x = _pad_last(x, self.window_samples)
                y = _pad_last(y, self.window_samples, pad_value_last_channel=1.0)

            return torch.from_numpy(x), torch.from_numpy(y)

    return PhaseDataset(*args, **kwargs)


def _normalize_window(x: np.ndarray) -> np.ndarray:
    """逐通道 z-score 归一化，数值稳定处理零方差通道。"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _pad_last(a: np.ndarray, target: int, pad_value_last_channel: float = 0.0) -> np.ndarray:
    """在最后一维右侧 pad 到 target 长度。目标图的噪声通道用 1.0 填充
    （pad 出来的区域视为纯噪声）。"""
    pad = target - a.shape[-1]
    if pad <= 0:
        return a
    if a.ndim == 1:
        return np.pad(a, (0, pad))
    out = np.pad(a, ((0, 0), (0, pad)))
    if pad_value_last_channel and a.shape[0] == 3:
        out[2, -pad:] = pad_value_last_channel
    return out
