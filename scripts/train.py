#!/usr/bin/env python3
"""训练 / 微调 CLI 入口。

用法（本地或 Cloud Studio 通用）：
    # 用默认配置微调 PhaseNet（数据目录必须先填）
    python scripts/train.py --data-root /path/to/data --exp phasenet_ft_v1

    # 断点续训（机器被自动关机后重开，接着上次跑）
    python scripts/train.py --data-root /path/to/data --exp phasenet_ft_v1 --resume

    # 跑一组新超参实验（改 exp 名，避免覆盖）
    python scripts/train.py --data-root /path/to/data --exp phasenet_lr5e4 --lr 5e-4

    # 从 JSON 配置文件加载（跑批时推荐，把每组实验存成一个 json）
    python scripts/train.py --config configs/exp_a.json

设计说明（写给非 AI 背景的队友）：
- 这个脚本本身不含训练逻辑，只负责"拼配置 + 调用 trainer"，方便阅读。
- 所有真正的训练细节都在 src/phasepicker/training/ 里，且已模块化、可单测。
- 命令行参数只覆盖最常调的几个；完整配置见 training/config.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在仓库根目录下直接运行即可 import 到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phasepicker.training.config import TrainConfig  # noqa: E402


def build_config(args: argparse.Namespace) -> TrainConfig:
    """把命令行参数合并成一个 TrainConfig。

    优先级：--config 文件 < 命令行显式参数（命令行覆盖文件）。
    这样既能用 json 存实验，又能临时在命令行微调某个字段。
    """
    if args.config:
        cfg = TrainConfig.load(args.config)
    else:
        cfg = TrainConfig()

    # 命令行显式给出的字段才覆盖（None 表示用户没传）
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.exp is not None:
        cfg.experiment_name = args.exp
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.base_model is not None:
        cfg.base_model = args.base_model
    if args.pretrained is not None:
        # 允许用 "none" 表示从头训练
        cfg.pretrained_weights = None if args.pretrained.lower() == "none" else args.pretrained
    if args.device is not None:
        cfg.device = args.device
    if args.oss_uri is not None:
        cfg.oss_sync_uri = args.oss_uri
    if args.output_root is not None:
        cfg.output_root = args.output_root
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhaseNet/EQTransformer 微调训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="从 JSON 配置文件加载（跑批推荐）")
    p.add_argument("--data-root", type=str, default=None, help="训练数据根目录（数据一到就填这里）")
    p.add_argument("--exp", type=str, default=None, help="实验名（每组实验唯一，用于 checkpoint 目录）")
    p.add_argument("--batch-size", type=int, default=None, help="批大小（OOM 就调小；T4 16G 可调大）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="学习率")
    p.add_argument("--base-model", type=str, default=None, help="PhaseNet / EQTransformer")
    p.add_argument("--pretrained", type=str, default=None, help="预训练权重名，或 'none' 从头训练")
    p.add_argument("--device", type=str, default=None, help="auto / cuda / cpu")
    p.add_argument("--oss-uri", type=str, default=None, help="checkpoint 同步到的 OSS 地址")
    p.add_argument("--output-root", type=str, default=None, help="产物根目录")
    p.add_argument("--resume", action="store_true", help="断点续训（从最近 checkpoint 继续）")
    p.add_argument("--dry-run", action="store_true", help="只校验配置与数据是否就绪，不真正训练")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    # 训练前自检：尽早暴露"配置写错 / 数据没填"这类致命低级错误
    problems = cfg.validate()
    if not cfg.data_root:
        problems.append("data_root 为空——训练数据还没到位。数据一到，用 --data-root 指定即可。")
    if problems:
        print("[配置自检未通过] 请先修正以下问题：", file=sys.stderr)
        for pb in problems:
            print(f"  - {pb}", file=sys.stderr)
        return 2

    print("========== 本次训练配置 ==========")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  解析后设备: {cfg.resolve_device()}")
    print("==================================")

    if args.dry_run:
        print("[dry-run] 配置与数据路径校验通过，未执行真正训练。")
        return 0

    # 真正的训练在此触发。trainer 依赖 torch/seisbench，仅在此处 import，
    # 保证 --dry-run / 单元测试在无重依赖环境也能跑。
    from phasepicker.training.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg)
    trainer.run(resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
