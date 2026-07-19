"""训练配置（Training config）——所有超参与路径的唯一真理来源。

===== 设计意图（写给非 AI 背景的队友）=====
把训练要调的所有旋钮集中到一个 dataclass，好处有三：
  1) 可复现——每次训练把这份 config 连同随机种子一起存进 checkpoint，
     日后能精确重跑同一次实验（决赛技术文档最看重"可复现"）。
  2) 可跑批——改几个字段就是一组新实验，配合本地评分脚本挑最优。
  3) 显存安全——默认值全部按 RTX4060 8GB 拧到"稳妥不炸显存"，
     换更大的卡（如 Cloud Studio 的 T4 16GB）只需调大 batch_size。

这个模块**零重依赖**（只用标准库），因此能在任何机器上 import、被单测覆盖。
torch/seisbench 只在真正训练时（trainer.py）才 import。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """一次微调实验的完整配置。

    字段按"最常调 → 很少动"排序，方便跑批时快速改上面几个。
    """

    # ---------- 实验标识 ----------
    experiment_name: str = "phasenet_ft_v1"
    """实验名。checkpoint 目录、日志、评分归档都用它，跑批时务必每组唯一。"""

    # ---------- 模型 ----------
    base_model: str = "PhaseNet"
    """基座模型，对应 SeisBench 的模型类名。可换 'EQTransformer' 做对比。"""
    pretrained_weights: Optional[str] = "instance"
    """SeisBench 预训练权重名（如 'instance'/'stead'/'scedc'）。
    None 表示从头训练（不推荐，除非有大规模自有数据）。微调时从这个权重出发。"""

    # ---------- 数据 ----------
    data_root: str = ""
    """训练数据根目录。**留空是有意为之**——数据还没到位，一到就填这里，
    其余代码零改动。这就是"赌数据很快到位"的可插拔点。"""
    sampling_rate: float = 100.0
    """统一采样率 Hz。SeisBench 模型默认 100Hz，与预处理模块保持一致。"""
    window_samples: int = 3001
    """送入模型的窗口长度（采样点）。PhaseNet 默认 3001（约 30s @100Hz）。"""

    # ---------- 训练循环（最常调）----------
    batch_size: int = 64
    """批大小。默认 64 是 4060 8GB 上 PhaseNet 的稳妥值；
    T4 16GB 可上到 128~256。若出现 CUDA out of memory，第一件事就是调小它。"""
    epochs: int = 30
    """训练轮数。PhaseNet 微调通常 10~30 轮内收敛，配合早停不必设太大。"""
    learning_rate: float = 1e-3
    """学习率。微调可比从头训练略小（如 1e-3 → 5e-4）以免破坏预训练特征。"""
    weight_decay: float = 0.0

    # ---------- 早停与验证 ----------
    val_fraction: float = 0.15
    """从训练集中切出的验证集比例。这个集合**只用于早停/选模型，绝不参与调参**，
    否则会对验证集过拟合、本地分虚高（详见 splits.py 的防泄漏设计）。"""
    early_stop_patience: int = 5
    """验证指标连续多少轮不提升就早停。省算力、防过拟合。"""

    # ---------- 数据增强（影响跨域泛化，值得跑批对比）----------
    augment: bool = True
    """是否开启数据增强（加噪、随机平移窗口等）。跨数据域时通常能提升泛化。"""

    # ---------- 设备与 IO ----------
    device: str = "auto"
    """'auto' / 'cuda' / 'cpu'。auto 时有 GPU 用 GPU，否则 CPU。
    对应你"本地 4060 训练、也可能在 CPU 上跑通流程"的需求。"""
    num_workers: int = 4
    """DataLoader 进程数。Cloud Studio / 本地按 CPU 核数调。"""
    output_root: str = "runs"
    """所有产物（checkpoint、日志、config 快照、指标）的根目录。"""

    # ---------- 可复现 ----------
    seed: int = 42
    """全局随机种子。固定它 + 存 config，才能精确重跑实验。"""

    # ---------- 云端持久化（对付 Cloud Studio 硬盘 15 天清理 + 关页面关机）----------
    oss_sync_uri: str = ""
    """checkpoint 实时同步到的 OSS 地址（如 'oss://your-bucket/runs/'）。
    留空则不同步（本地训练时）。**在 Cloud Studio 上必须填**——机器随时释放，
    权重必须在它死之前就已经躺在 OSS 里。见 checkpoint.py 的 sync 钩子。"""
    sync_every_n_steps: int = 200
    """每多少 step 触发一次 checkpoint + OSS 同步。越小越安全、IO 越频繁。"""

    def run_dir(self) -> Path:
        """本次实验的产物目录：output_root/experiment_name。"""
        return Path(self.output_root) / self.experiment_name

    def resolve_device(self) -> str:
        """把 'auto' 解析成实际设备字符串。仅在此处 import torch，
        保证无 torch 环境也能 import 本模块。"""
        if self.device != "auto":
            return self.device
        try:
            import torch  # 局部 import，隔离重依赖

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 —— 无 torch 时安全退回 CPU
            return "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """把 config 存成 JSON（随 checkpoint 一起归档，保证可复现）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        """从 JSON 恢复 config。断点续训 / 复现实验时用。

        对未知字段容错：只挑 dataclass 里存在的字段，避免旧 checkpoint
        因新增字段而加载失败。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def validate(self) -> list[str]:
        """轻量自检，返回问题列表（空列表=通过）。在训练开始前调用，
        尽早暴露"配置写错"这类低级但致命的问题。"""
        problems: list[str] = []
        if self.batch_size <= 0:
            problems.append("batch_size 必须为正")
        if self.epochs <= 0:
            problems.append("epochs 必须为正")
        if not (0.0 < self.val_fraction < 1.0):
            problems.append("val_fraction 必须在 (0,1) 之间")
        if self.learning_rate <= 0:
            problems.append("learning_rate 必须为正")
        if self.window_samples <= 0:
            problems.append("window_samples 必须为正")
        if self.sampling_rate <= 0:
            problems.append("sampling_rate 必须为正")
        return problems
