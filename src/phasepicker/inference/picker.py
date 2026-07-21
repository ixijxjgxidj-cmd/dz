"""震相拾取推理封装（Inference）——SeisBench 模型的统一接口.

设计目标（对应原方案"可扩展、不硬编码模型结构"）：
- 用一个抽象基类 ``BasePicker`` 定义统一契约：输入 Waveform，输出 List[Pick]。
- ``SeisBenchPicker`` 是默认实现，通过"模型名 + 权重名"从 SeisBench 加载，
  **不硬编码网络结构**。要换 PhaseNet→EQTransformer，或换成微调后的权重，
  只改配置，不改代码。
- 设备（cpu/cuda）可配置：本地用 cuda，云上若是 CPU 主机自动退化到 cpu。
  PhaseNet 很小，单条波形 CPU 推理完全可接受，这正是可用性架构的底气。

显存说明（RTX4060 8GB）：
- PhaseNet 参数量极小（~几 MB），推理显存主要由输入长度决定。
- 采用 SeisBench 的 ``annotate`` 滑窗机制处理长波形，避免一次性喂入超长序列爆显存。
- 微调时 batch_size 建议从 64 起（100Hz、30s 窗口），8GB 足够；OOM 则减半。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..types import Pick, PhaseType, Waveform, CHANNEL_ORDER
from ..utils.timing import sample_to_utc
from ..postprocess.dedup import dedup_picks, DedupConfig


@dataclass
class PickerConfig:
    """推理配置。全部可配置，供参数搜索与部署环境切换。

    Attributes:
        model_name: SeisBench 模型类名，如 "PhaseNet" / "EQTransformer"。
        pretrained: 预训练权重名，如 "original" / "stead" / "instance"，
            或指向本地微调权重的标识（见 from_config 的加载逻辑）。
        device: "cuda" 或 "cpu"。None 表示自动探测。
        p_threshold: P 波触发概率阈值。**调高偏向高精确率**（少误报），
            这是应对"数量误差每超1个扣0.5分"的关键旋钮。
        s_threshold: S 波触发概率阈值。
        batch_size: annotate 滑窗的 batch 大小，受显存限制。
        overlap: 滑窗重叠比例，边界震相不漏检。
        local_weights_path: 本地微调权重 (.pt) 路径。给定则优先加载，
            这就是"数据一到位即可切换到微调模型"的可插拔入口。
    """

    model_name: str = "PhaseNet"
    pretrained: str = "original"
    device: Optional[str] = None
    p_threshold: float = 0.3
    s_threshold: float = 0.3
    batch_size: int = 256
    overlap: float = 0.5
    local_weights_path: Optional[str] = None


class BasePicker(ABC):
    """拾取器抽象契约。任何实现只要吃 Waveform、吐 List[Pick] 即可接入系统。"""

    @abstractmethod
    def pick(self, wf: Waveform) -> List[Pick]:
        """对单台站波形做拾取，返回去重后的震相列表。"""
        raise NotImplementedError


def _resolve_device(requested: Optional[str]) -> str:
    """决定推理设备。请求 cuda 但不可用时安全退化到 cpu 并保持可用。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("推理需要 PyTorch，请先安装 torch") from exc
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class SeisBenchPicker(BasePicker):
    """基于 SeisBench 的默认拾取器。"""

    def __init__(self, model, cfg: PickerConfig, dedup_cfg: Optional[DedupConfig] = None):
        """通常不直接用构造函数，用 ``SeisBenchPicker.from_config(cfg)``。"""
        self._model = model
        self._cfg = cfg
        self._dedup_cfg = dedup_cfg or DedupConfig()

    @classmethod
    def from_config(cls, cfg: PickerConfig, dedup_cfg: Optional[DedupConfig] = None) -> "SeisBenchPicker":
        """按配置加载模型权重（预训练或本地微调），移到目标设备并置 eval。"""
        import seisbench.models as sbm
        import torch

        device = _resolve_device(cfg.device)

        model_cls = getattr(sbm, cfg.model_name, None)
        if model_cls is None:
            raise ValueError(
                f"SeisBench 中不存在模型 {cfg.model_name!r}，"
                f"可选如 'PhaseNet' / 'EQTransformer'"
            )

        if cfg.local_weights_path:
            # 微调权重加载路径：先用同一个预训练配置实例化，确保标签顺序、
            # 归一化方式和网络结构与训练时完全一致，再灌入本地 state_dict。
            model = model_cls.from_pretrained(cfg.pretrained)
            try:
                checkpoint = torch.load(
                    cfg.local_weights_path,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:  # 兼容较老 PyTorch（没有 weights_only 参数）
                checkpoint = torch.load(cfg.local_weights_path, map_location=device)
            if isinstance(checkpoint, dict):
                # 本仓库两套训练代码分别使用 model / model_state_dict；同时兼容
                # 常见第三方 checkpoint 的 state_dict 键以及裸 state_dict。
                state = checkpoint.get("model_state_dict")
                if state is None:
                    state = checkpoint.get("model")
                if state is None:
                    state = checkpoint.get("state_dict")
                if state is None:
                    state = checkpoint
            else:
                state = checkpoint
            model.load_state_dict(state)
        else:
            model = model_cls.from_pretrained(cfg.pretrained)

        model.to(device)
        model.eval()
        return cls(model, cfg, dedup_cfg)

    def pick(self, wf: Waveform) -> List[Pick]:
        """对单台站波形做拾取。

        流程：Waveform → ObsPy Stream → model.annotate（滑窗概率）→
        classify（阈值触发出 picks）→ 采样点/相对时间换算为绝对 UTC →
        构造 Pick → 去重合并。
        """
        import torch

        stream = self._to_stream(wf)

        # classify 内部会 annotate 并按阈值挑峰，返回带绝对时间的 picks。
        # 用官方 API 而非自己挑峰，避免重复造轮子且行为与 SeisBench 一致。
        with torch.no_grad():
            outputs = self._model.classify(
                stream,
                batch_size=self._cfg.batch_size,
                overlap=self._cfg.overlap,
                P_threshold=self._cfg.p_threshold,
                S_threshold=self._cfg.s_threshold,
            )

        picks: List[Pick] = []
        for p in getattr(outputs, "picks", outputs):
            phase = self._normalize_phase(p.phase)
            if phase is None:
                continue
            # SeisBench pick 的 peak_time 是 ObsPy UTCDateTime（绝对时间）。
            # 直接取其 epoch 秒，与内部 time_utc 约定一致；不经过手动采样点换算，
            # 消除一次潜在的对齐误差来源。
            time_utc = float(p.peak_time.timestamp)
            picks.append(
                Pick(
                    phase=phase,
                    time_utc=time_utc,
                    confidence=float(getattr(p, "peak_value", 0.0)),
                    station=wf.station,
                )
            )

        return dedup_picks(picks, self._dedup_cfg)

    def _to_stream(self, wf: Waveform):
        from obspy import Stream, Trace, UTCDateTime

        st = Stream()
        for i, comp in enumerate(CHANNEL_ORDER):
            tr = Trace(data=np.ascontiguousarray(wf.data[i], dtype=np.float32))
            tr.stats.sampling_rate = wf.sampling_rate
            tr.stats.starttime = UTCDateTime(wf.starttime_utc)
            tr.stats.channel = f"HH{comp}"  # SeisBench 按通道码识别分量
            tr.stats.station = wf.station or "STA"
            st.append(tr)
        return st

    @staticmethod
    def _normalize_phase(raw: str) -> Optional[PhaseType]:
        """把模型的相位标签统一到 PhaseType；非 P/S（如噪声/检测）返回 None。"""
        s = str(raw).upper()
        if s == "P":
            return PhaseType.P
        if s == "S":
            return PhaseType.S
        return None
