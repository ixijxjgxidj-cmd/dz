"""Checkpoint 管理与断点续训（Checkpoint & resume）.

===== 为什么这是训练脚手架的心脏（写给非 AI 背景的队友）=====
你的 Cloud Studio 免费 GPU 有三个"随时会弄丢成果"的特性：
  1) 关闭编辑器网页标签 → 10 分钟后自动关机；
  2) 赠送的 80GB 云硬盘免费保留期只有 15 天，超期清理；
  3) 单个 session 最长 24 小时，到点释放。

所以训练不能"跑完才存"，必须"边跑边存、存完立刻外送"。本模块的职责：
  - 每隔 N 步/每个 epoch 存一次 checkpoint（含模型权重 + 优化器状态 +
    随机数生成器状态 + 当前步数），保证机器随时没了都能从最近一次接着跑；
  - 维护 last.ckpt（最近）和 best.ckpt（验证分最高）两个指针；
  - 存档后可选地触发一个"外送钩子"（把文件推到阿里云 OSS），
    真正做到"权重在机器死之前已经在别处"。

===== 与训练框架解耦 =====
本模块只负责"存什么、怎么组织、什么时候提示外送"，不 import torch。
真正的 torch.save/torch.load 由调用方（trainer.py）传入的两个回调完成，
这样：
  - 无 torch 环境（比如本地快速跑通、CI）也能测本模块的簿记逻辑；
  - 未来换框架（PyTorch Lightning 等）只改 trainer，本模块零改动。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# checkpoint 目录下的固定文件名约定，别处禁止硬编码这些字符串
LAST_NAME = "last.ckpt"
BEST_NAME = "best.ckpt"
MANIFEST_NAME = "manifest.json"


@dataclass
class CheckpointRecord:
    """单个 checkpoint 的元信息（不含权重本身，权重在 .ckpt 文件里）。"""

    step: int
    epoch: int
    val_score: Optional[float]
    path: str
    wall_time: float
    is_best: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "val_score": self.val_score,
            "path": os.path.basename(self.path),  # manifest 只存文件名，便于整目录搬运
            "wall_time": self.wall_time,
            "is_best": self.is_best,
        }


class CheckpointManager:
    """管理一次训练运行的所有 checkpoint。

    典型用法（在 trainer 里）::

        cm = CheckpointManager(
            ckpt_dir="runs/exp1/ckpts",
            save_fn=lambda path, payload: torch.save(payload, path),
            higher_is_better=True,
            keep_last_n=3,
            upload_hook=oss_upload,   # 可选：存完立刻推 OSS
        )
        # 训练中：
        cm.save(step, epoch, payload=build_payload(), val_score=score)
        # 断点续训（开机第一件事）：
        payload = cm.load_latest(load_fn=lambda path: torch.load(path))
        if payload: resume_from(payload)

    Args:
        ckpt_dir: checkpoint 存放目录，不存在会自动建。
        save_fn: 回调 (path, payload) -> None，实际写盘（通常是 torch.save）。
        higher_is_better: 验证指标是否越大越好。我们的本地评分是"总分"，
            越大越好，故默认 True；若换成 loss 之类应设 False。
        keep_last_n: 最多保留几个"按步存的"历史 checkpoint（best 和 last 不计入
            清理），控制磁盘占用——Cloud Studio 只有 80GB，别把盘写爆。
        upload_hook: 可选回调 (local_path) -> None，每次存档后调用，用于外送 OSS。
            设计成"尽力而为"：钩子内部异常只告警、不中断训练。
    """

    def __init__(
        self,
        ckpt_dir: str,
        save_fn: Callable[[str, dict], None],
        higher_is_better: bool = True,
        keep_last_n: int = 3,
        upload_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.save_fn = save_fn
        self.higher_is_better = higher_is_better
        self.keep_last_n = max(1, keep_last_n)
        self.upload_hook = upload_hook

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.records: List[CheckpointRecord] = []
        self.best_score: Optional[float] = None
        self._load_manifest()

    # ---------------- 内部工具 ----------------

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.ckpt_dir, MANIFEST_NAME)

    def _load_manifest(self) -> None:
        """开机时恢复历史簿记（若 manifest 存在），支撑跨 session 续训。"""
        if not os.path.exists(self.manifest_path):
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # manifest 损坏不应让训练无法启动，忽略即可重建
            return
        self.best_score = data.get("best_score")
        for r in data.get("records", []):
            self.records.append(
                CheckpointRecord(
                    step=r["step"],
                    epoch=r["epoch"],
                    val_score=r.get("val_score"),
                    path=os.path.join(self.ckpt_dir, r["path"]),
                    wall_time=r.get("wall_time", 0.0),
                    is_best=r.get("is_best", False),
                )
            )

    def _write_manifest(self) -> None:
        data = {
            "best_score": self.best_score,
            "higher_is_better": self.higher_is_better,
            "records": [r.to_dict() for r in self.records],
        }
        # 先写临时文件再原子改名，避免写一半被关机导致 manifest 损坏
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.manifest_path)

    def _is_better(self, score: Optional[float]) -> bool:
        if score is None:
            return False
        if self.best_score is None:
            return True
        return (
            score > self.best_score
            if self.higher_is_better
            else score < self.best_score
        )

    def _fire_upload(self, path: str) -> None:
        """尽力而为地外送；钩子异常绝不中断训练。"""
        if self.upload_hook is None:
            return
        try:
            self.upload_hook(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] ⚠️ 外送失败（训练继续）: {path} -> {exc!r}")

    def _prune(self) -> None:
        """清理超过 keep_last_n 的历史按步 checkpoint，保护 80GB 磁盘。

        保护规则：best 和 最新一个 last 永不删；其余按步数从旧到新删到只剩 keep_last_n。
        """
        # 候选：既不是 best、也不是最新的那个
        deletable = [r for r in self.records if not r.is_best]
        deletable.sort(key=lambda r: r.step)
        # 最新的保留（last），其余多出来的删
        while len(deletable) > self.keep_last_n:
            victim = deletable.pop(0)
            if os.path.exists(victim.path) and os.path.basename(victim.path) not in (
                LAST_NAME,
                BEST_NAME,
            ):
                try:
                    os.remove(victim.path)
                except OSError:
                    pass
            self.records = [r for r in self.records if r is not victim]

    # ---------------- 对外 API ----------------

    def save(
        self,
        step: int,
        epoch: int,
        payload: dict,
        val_score: Optional[float] = None,
    ) -> CheckpointRecord:
        """存一个 checkpoint。

        payload 应由 trainer 组装，至少包含（约定俗成，本模块不校验内容）:
            {
              "model_state": ...,        # 模型权重
              "optimizer_state": ...,    # 优化器状态（续训必需）
              "scheduler_state": ...,    # 学习率调度器（可选）
              "rng_state": ...,          # 随机数状态（见 seed.py，保证严格可复现续训）
              "step": step, "epoch": epoch,
              "config": ...,             # 训练配置快照（便于溯源）
            }

        Returns:
            本次 CheckpointRecord。
        """
        step_name = f"step_{step:08d}.ckpt"
        step_path = os.path.join(self.ckpt_dir, step_name)

        # 1) 写按步存档
        self.save_fn(step_path, payload)
        rec = CheckpointRecord(
            step=step,
            epoch=epoch,
            val_score=val_score,
            path=step_path,
            wall_time=time.time(),
        )
        self.records.append(rec)

        # 2) 更新 last 指针（拷贝一份为 last.ckpt，方便"开机直接找 last"）
        last_path = os.path.join(self.ckpt_dir, LAST_NAME)
        shutil.copyfile(step_path, last_path)

        # 3) 若刷新最佳，更新 best 指针
        if self._is_better(val_score):
            self.best_score = val_score
            best_path = os.path.join(self.ckpt_dir, BEST_NAME)
            shutil.copyfile(step_path, best_path)
            rec.is_best = True
            self._fire_upload(best_path)  # 最佳权重优先外送

        # 4) 清理旧档 + 落 manifest
        self._prune()
        self._write_manifest()

        # 5) 外送本次存档与 last（尽力而为）
        self._fire_upload(step_path)
        self._fire_upload(last_path)
        return rec

    def latest_path(self) -> Optional[str]:
        """返回可用于续训的最近 checkpoint 路径；没有则 None。"""
        last_path = os.path.join(self.ckpt_dir, LAST_NAME)
        if os.path.exists(last_path):
            return last_path
        if self.records:
            newest = max(self.records, key=lambda r: r.step)
            if os.path.exists(newest.path):
                return newest.path
        return None

    def best_path(self) -> Optional[str]:
        """返回验证分最高的 checkpoint 路径；没有则 None。"""
        p = os.path.join(self.ckpt_dir, BEST_NAME)
        return p if os.path.exists(p) else None

    def load_latest(self, load_fn: Callable[[str], dict]) -> Optional[dict]:
        """加载最近 checkpoint 用于续训。开机后第一件事应调用它。

        Args:
            load_fn: 回调 (path) -> payload，通常是 torch.load。

        Returns:
            payload dict；若没有任何 checkpoint 返回 None（表示全新开训）。
        """
        p = self.latest_path()
        if p is None:
            return None
        return load_fn(p)
