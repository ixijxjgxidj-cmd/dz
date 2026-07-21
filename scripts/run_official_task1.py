#!/usr/bin/env python3
"""官方 T1 端到端 CLI —— 从官方 .mseed 输入产出 Task1Result/T1.an（相对秒）.

用法：
    python scripts/run_official_task1.py --input exam2025/ --output T1.an
    python scripts/run_official_task1.py --input round1.zip --output T1.an --answer answers.txt
    python scripts/run_official_task1.py --input in/ --output T1.an \\
        --weights ckpts/phasenet_ft.pt --device cuda --p-threshold 0.4 --s-threshold 0.3

流程（复用已测通的各层，不重复造轮子）：
    scan_exam_input → 过滤 T1 → run_task1_samples(load_waveforms, picker) →
    write_task1_results → （可选）evaluate_task1 打分。

依赖说明：
- 扫描 / 换算 / 写出 / 评估都是纯标准库或纯 numpy，随时可跑。
- **真实推理**需要 ObsPy（读 mseed）+ SeisBench + PyTorch（PhaseNet）。缺任一
  都会在构建 picker 或读波形时给出清晰的中文报错，指明缺哪个包，而不是隐晦栈回溯。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

# 让脚本无需安装即可 import 到包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamSample, ExamTask, Waveform  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.submission_writer import write_task1_results  # noqa: E402
from phasepicker.tasks.task1_runner import run_task1_samples  # noqa: E402


def _read_sample_bytes(sample: ExamSample) -> bytes:
    """把 ExamSample.source_path 读成原始 mseed 字节，兼容普通文件与 zip 内条目。

    official_exam 对 zip 内条目记 source_path 为 ``<zip_path>!<entry_name>``；
    这里据此分流：含 ``!`` 且前段是 zip → 从 zip 读该条目（不解压落盘），
    否则按普通文件路径读。
    """
    from phasepicker.io.official_waveforms import read_source_bytes

    return read_source_bytes(sample.source_path)


def _make_load_waveforms_fn():
    """构造 load_waveforms_fn；ObsPy 缺失时给清晰报错。

    返回一个吃 ExamSample、吐 List[Waveform] 的闭包。内部用 mseed_reader
    （依赖 ObsPy）把字节解成校验过的多台站波形。
    """
    try:
        from phasepicker.io.mseed_reader import load_waveforms
    except Exception as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            f"读取 mseed 需要 ObsPy，导入失败：{exc!r}\n"
            "请先安装：pip install obspy"
        )

    def _load(sample: ExamSample) -> List[Waveform]:
        raw = _read_sample_bytes(sample)
        result = load_waveforms(raw)
        for w in result.warnings:
            print(f"[warn] {sample.file_id} [{w.station}] {w.reason}: {w.detail}", file=sys.stderr)
        return result.waveforms

    return _load


def _make_picker(
    weights: str | None,
    device: str,
    p_threshold: float,
    s_threshold: float,
    pretrained: str,
):
    """按参数构建 SeisBenchPicker；torch/seisbench 缺失时给清晰报错。"""
    try:
        from phasepicker.inference.picker import PickerConfig, SeisBenchPicker
    except Exception as exc:  # pragma: no cover - 环境相关
        raise SystemExit(f"构建 picker 失败（import 阶段）：{exc!r}")

    cfg = PickerConfig(
        device=device,
        pretrained=pretrained,
        p_threshold=p_threshold,
        s_threshold=s_threshold,
        local_weights_path=weights,
    )
    try:
        return SeisBenchPicker.from_config(cfg)
    except ImportError as exc:
        raise SystemExit(
            f"真实推理需要 SeisBench + PyTorch，加载失败：{exc!r}\n"
            "请先安装：pip install seisbench torch"
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"加载模型失败：{exc!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="官方 T1 端到端拾取（输出相对秒 T1.an）")
    ap.add_argument("--input", required=True, help="官方输入目录或 zip")
    ap.add_argument("--output", required=True, help="输出提交文件（T1.an）")
    ap.add_argument("--weights", default=None, help="可选：本地微调权重 (.pt) 路径")
    ap.add_argument("--pretrained", default="stead",
                    help="SeisBench 基础权重名；默认 stead（本地微调权重也按此结构加载）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    ap.add_argument("--p-threshold", type=float, default=0.3, help="P 波触发概率阈值")
    ap.add_argument("--s-threshold", type=float, default=0.3, help="S 波触发概率阈值")
    ap.add_argument("--prefix", default="", help="写出行的路径前缀（如 exam2025/TASK01/）")
    ap.add_argument("--answer", default=None, help="可选：官方答案文件，提供则跑 official_eval 打分")
    ap.add_argument("--answer-package", default=None,
                    help="可选：直接从官方 zip（含嵌套 zip）读取 T1 答案并打分")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅调试：只跑前 N 个 T1 文件；正式提交不要设置")
    args = ap.parse_args(argv)

    # 1) 扫描输入并过滤出 T1 样本
    samples = [s for s in scan_exam_input(args.input) if s.task == ExamTask.T1]
    if args.limit is not None:
        if args.limit <= 0:
            ap.error("--limit 必须是正整数")
        samples = samples[: args.limit]
    if not samples:
        print(f"未在 {args.input!r} 找到任何 T1 样本（.mseed）", file=sys.stderr)
        return 1
    print(f"扫描到 {len(samples)} 个 T1 样本")

    # 2) 构建依赖（缺 obspy/seisbench/torch 会在此给出清晰报错）
    load_waveforms_fn = _make_load_waveforms_fn()
    picker = _make_picker(
        args.weights,
        args.device,
        args.p_threshold,
        args.s_threshold,
        args.pretrained,
    )

    # 3) 端到端推理 → 相对秒 Task1Result
    results_map = run_task1_samples(samples, load_waveforms_fn, picker)

    # 4) 写出（保持输入扫描顺序，便于复现）
    ordered = [results_map[s.file_id] for s in samples if s.file_id in results_map]
    write_task1_results(ordered, args.output, prefix=args.prefix)
    n_p = sum(len(r.p_times_s) for r in ordered)
    n_s = sum(len(r.s_times_s) for r in ordered)
    print(f"已写出 {len(ordered)} 行到 {args.output}（P 到时 {n_p} 个，S 到时 {n_s} 个）")

    # 5) 可选打分
    if args.answer and args.answer_package:
        ap.error("--answer 与 --answer-package 只能选一个")
    if args.answer or args.answer_package:
        from phasepicker.io.official_answers import parse_task1_answer_lines
        from phasepicker.eval.official_eval import evaluate_task1

        if args.answer_package:
            from phasepicker.io.official_waveforms import read_package_answers

            answers = read_package_answers(args.answer_package, ExamTask.T1)
        else:
            with open(args.answer, "r", encoding="utf-8", errors="replace") as f:
                answers = parse_task1_answer_lines(f.read().splitlines())
        report = evaluate_task1(results_map, answers)
        print(report.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
