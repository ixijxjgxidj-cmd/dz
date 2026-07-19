"""训练循环（Trainer）—— 把所有部件串成一次可续训、抗释放的微调。

===== 这个文件如何对付 Cloud Studio 的三个"释放"特性 =====
1) 关页面 10 分钟自动关机  → 训练用 nohup 后台跑（见 scripts/），且每
   sync_every_n_steps 步就存一次 checkpoint，关机时最多丢几百步。
2) 硬盘 15 天清理          → 每次存 checkpoint 立刻通过 CheckpointManager
   的 OSS 钩子推到对象存储，权重在机器死之前就已经在别处。
3) 单 session 24h 上限     → 存优化器状态 + 随机种子 + 已完成 epoch/step，
   下次开机 `--resume` 即可无缝续训，跨 session 累积训练。

⚠️ 依赖 torch/seisbench，属于"在你机器/Cloud Studio 上跑"的部分。
本文件被设计成：无 torch 时 import 不炸（重依赖全部延迟到函数内），
因此 CLI 的 --help、config 校验等仍可在任意环境验证。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .config import TrainConfig
from .seed import set_global_seed
from .checkpoint import CheckpointManager


def build_model(cfg: TrainConfig):
    """按 config 从 SeisBench 构造（并可加载预训练权重的）模型。

    可扩展点：这里不硬编码模型结构，而是按 cfg.base_model 动态取类，
    以后换 EQTransformer 或换权重只改 config，不改代码。
    """
    import seisbench.models as sbm  # 延迟导入

    model_cls = getattr(sbm, cfg.base_model, None)
    if model_cls is None:
        raise ValueError(
            f"SeisBench 中找不到模型 {cfg.base_model!r}；"
            f"可选如 PhaseNet / EQTransformer"
        )
    if cfg.pretrained_weights:
        # 从预训练权重出发做微调（迁移学习），这是本比赛的主线策略
        model = model_cls.from_pretrained(cfg.pretrained_weights)
    else:
        model = model_cls()  # 从头训练（少用）
    return model


def _phase_loss():
    """PhaseNet 的目标是三通道概率图，用逐点交叉熵/KL 的软标签形式。
    这里用带 log 的 KL 散度近似（软标签交叉熵），数值稳定。"""
    import torch
    import torch.nn as nn

    logsm = nn.LogSoftmax(dim=1)

    def loss_fn(pred, target):
        # pred: (B,3,N) 原始 logits；target: (B,3,N) 概率(每列和≈1)
        return -(target * logsm(pred)).sum(dim=1).mean()

    return loss_fn


def train(
    cfg: TrainConfig,
    train_loader,
    val_loader,
    resume: bool = False,
    oss_sync_fn: Optional[Callable[[Path], None]] = None,
    log_fn: Callable[[str], None] = print,
) -> dict:
    """执行微调训练。返回最终指标字典（也会写入 run_dir）。

    Args:
        cfg: 训练配置。
        train_loader / val_loader: torch DataLoader。
        resume: True 则尝试从 run_dir 的 last checkpoint 续训。
        oss_sync_fn: 可选的 OSS 同步回调，签名 fn(local_dir)。由 CLI 注入
            （通常包一层 `ossutil cp -r`）。留空则只存本地。
        log_fn: 日志输出函数，便于重定向到文件。

    Returns:
        {"best_val_loss": ..., "epochs_run": ..., "run_dir": ...}
    """
    import torch

    problems = cfg.validate()
    if problems:
        raise ValueError("配置校验未通过：" + "；".join(problems))

    set_global_seed(cfg.seed)
    device = cfg.resolve_device()
    log_fn(f"[trainer] 设备={device} 模型={cfg.base_model} "
           f"预训练权重={cfg.pretrained_weights} batch={cfg.batch_size}")

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = _phase_loss()

    ckpt = CheckpointManager(
        run_dir=cfg.run_dir(),
        oss_sync_fn=oss_sync_fn,
        keep_last=3,
    )

    start_epoch, global_step, best_val = 0, 0, float("inf")
    no_improve = 0

    if resume:
        state = ckpt.load_last(map_location=device)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            global_step = state["global_step"]
            best_val = state.get("best_val", best_val)
            log_fn(f"[trainer] 已从 checkpoint 续训：epoch={start_epoch} "
                   f"step={global_step} best_val={best_val:.4f}")
        else:
            log_fn("[trainer] 未找到可续训的 checkpoint，从头开始")

    cfg.save(cfg.run_dir() / "config.json")  # 归档配置，保证可复现

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1

            # —— 抗释放核心：定期存档 + OSS 同步 ——
            if global_step % cfg.sync_every_n_steps == 0:
                ckpt.save(
                    tag="last",
                    payload={
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                        "config": cfg.to_dict(),
                    },
                )
                log_fn(f"[trainer] step={global_step} 已存档并同步（抗释放）")

        train_loss = epoch_loss / max(1, n_batches)
        val_loss = _evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        log_fn(f"[epoch {epoch}] train_loss={train_loss:.4f} "
               f"val_loss={val_loss:.4f} 用时={dt:.1f}s")

        # 早停 + 保存最优
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            no_improve = 0
            ckpt.save(
                tag="best",
                payload={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val": best_val,
                    "config": cfg.to_dict(),
                },
            )
            log_fn(f"[epoch {epoch}] ✅ 新最优 val_loss={best_val:.4f}，已存 best")
        else:
            no_improve += 1
            log_fn(f"[epoch {epoch}] 未提升（{no_improve}/{cfg.early_stop_patience}）")
            if no_improve >= cfg.early_stop_patience:
                log_fn(f"[trainer] 触发早停于 epoch {epoch}")
                break

        # 每个 epoch 末也存一次 last，保证 epoch 边界可续训
        ckpt.save(
            tag="last",
            payload={
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "config": cfg.to_dict(),
            },
        )

    metrics = {
        "best_val_loss": best_val,
        "epochs_run": epoch + 1 if cfg.epochs > start_epoch else start_epoch,
        "run_dir": str(cfg.run_dir()),
    }
    import json
    (cfg.run_dir() / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_fn(f"[trainer] 训练结束：{metrics}")
    return metrics


def _evaluate(model, loader, loss_fn, device) -> float:
    """在验证集上算平均 loss（仅用于早停/选模型，不参与调参）。"""
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += float(loss_fn(model(x), y).item())
            n += 1
    return total / max(1, n)
