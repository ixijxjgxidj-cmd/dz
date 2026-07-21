#!/usr/bin/env python
"""用往届官方包训练 T2 震级与 T3 事件分类基线。

推荐的诚实验证方式：第 1 轮训练、第 2 轮评估。

示例：
  python scripts/train_official_baselines.py \
    --train-zip "第1轮.zip" --eval-zip "第2轮.zip" \
    --out-dir weights/official_r1_to_r2

最终模型可把两轮都放进 --train-zip；同名文件不会冲突，因为逐包读取与配对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.eval.official_eval import evaluate_task2, evaluate_task3  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_mseed_stream, read_package_answers  # noqa: E402
from phasepicker.tasks.baseline_models import (  # noqa: E402
    BaselineModelBundle,
    save_bundle,
    train_event_baseline,
    train_magnitude_baseline,
)
from phasepicker.tasks.waveform_features import FEATURE_NAMES, extract_waveform_features  # noqa: E402
from phasepicker.types import ExamTask, Task2Result, Task3Result  # noqa: E402


def _collect_package(zip_path: str, task: ExamTask, metadata_encoding: str = "gbk"):
    answers = read_package_answers(zip_path, task, metadata_encoding=metadata_encoding)
    samples = [s for s in scan_exam_input(zip_path, metadata_encoding=metadata_encoding) if s.task is task]
    X: List[np.ndarray] = []
    y: List[float] = []
    ids: List[str] = []
    failures: List[str] = []
    matched = [s for s in samples if s.file_id in answers]
    print(f"[{task.value}] {os.path.basename(zip_path)}：找到 {len(samples)} 个波形，答案匹配 {len(matched)} 个")
    for i, sample in enumerate(matched, 1):
        try:
            stream = read_mseed_stream(sample.source_path, metadata_encoding=metadata_encoding)
            X.append(extract_waveform_features(stream))
            target = answers[sample.file_id]
            y.append(float(target.magnitude) if task is ExamTask.T2 else int(target.label))
            ids.append(sample.file_id)
        except Exception as exc:  # 单个坏文件不让整批中止，但必须显式记录
            failures.append(f"{sample.file_id}: {type(exc).__name__}: {exc}")
        if i == 1 or i % 25 == 0 or i == len(matched):
            print(f"    已提取 {i}/{len(matched)}，成功 {len(X)}，失败 {len(failures)}")
    if failures:
        print(f"    警告：失败 {len(failures)} 个，前 5 个：")
        for line in failures[:5]:
            print(f"      - {line}")
    return np.asarray(X, dtype=np.float64), np.asarray(y), ids, answers, failures


def _concat_packages(paths: Sequence[str], task: ExamTask, metadata_encoding: str):
    xs, ys = [], []
    counts: Dict[str, int] = {}
    failures: Dict[str, List[str]] = {}
    for path in paths:
        X, y, _ids, _answers, failed = _collect_package(path, task, metadata_encoding)
        if len(X):
            xs.append(X)
            ys.append(y)
        counts[os.path.abspath(path)] = int(len(X))
        failures[os.path.abspath(path)] = failed
    if not xs:
        raise RuntimeError(f"{task.value} 没有可训练样本")
    return np.concatenate(xs), np.concatenate(ys), counts, failures


def _predict_package(bundle: BaselineModelBundle, zip_path: str, task: ExamTask, metadata_encoding: str):
    answers = read_package_answers(zip_path, task, metadata_encoding=metadata_encoding)
    samples = [s for s in scan_exam_input(zip_path, metadata_encoding=metadata_encoding) if s.task is task]
    predictions = {}
    failures = []
    for i, sample in enumerate(samples, 1):
        if sample.file_id not in answers:
            continue
        try:
            stream = read_mseed_stream(sample.source_path, metadata_encoding=metadata_encoding)
            feat = extract_waveform_features(stream)
            value = bundle.predict_one(feat)
            if task is ExamTask.T2:
                predictions[sample.file_id] = Task2Result(sample.file_id, float(value))
            else:
                predictions[sample.file_id] = Task3Result(sample.file_id, int(value))
        except Exception as exc:
            failures.append(f"{sample.file_id}: {type(exc).__name__}: {exc}")
        if i % 50 == 0 or i == len(samples):
            print(f"    评估提取 {i}/{len(samples)}")
    return predictions, answers, failures


def _evaluate(bundle, paths: Sequence[str], task: ExamTask, train_y, metadata_encoding: str):
    reports = {}
    if task is ExamTask.T2:
        constant = float(np.median(train_y))
    else:
        constant = int(Counter(int(v) for v in train_y).most_common(1)[0][0])
    for path in paths:
        print(f"[{task.value}] 跨包评估：{os.path.basename(path)}")
        pred, truth, failures = _predict_package(bundle, path, task, metadata_encoding)
        if task is ExamTask.T2:
            report = evaluate_task2(pred, truth)
            const_pred = {fid: Task2Result(fid, constant) for fid in truth}
            const_report = evaluate_task2(const_pred, truth)
            result = {
                "model_mae": report.mae,
                "model_max_abs_error": report.max_abs_error,
                "model_count": report.count,
                "constant_value": constant,
                "constant_mae": const_report.mae,
                "failed": failures,
            }
            print(f"    模型 {report.summary()}")
            print(f"    常数基线 MAE={const_report.mae:.4f}（常数={constant:.3f}）")
        else:
            report = evaluate_task3(pred, truth)
            const_pred = {fid: Task3Result(fid, constant) for fid in truth}
            const_report = evaluate_task3(const_pred, truth)
            result = {
                "model_accuracy": report.accuracy,
                "model_correct": report.correct,
                "model_count": report.count,
                "constant_class": constant,
                "constant_accuracy": const_report.accuracy,
                "confusion": {f"{a}->{b}": n for (a, b), n in report.confusion.items()},
                "failed": failures,
            }
            print(f"    模型 {report.summary()}")
            print(f"    多数类基线 acc={const_report.accuracy:.4f}（类别={constant}）")
        reports[os.path.abspath(path)] = result
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练官方 T2/T3 轻量可提交基线")
    ap.add_argument("--train-zip", action="append", required=True, help="训练官方包；可重复传入")
    ap.add_argument("--eval-zip", action="append", default=[], help="独立评估包；可重复传入")
    ap.add_argument("--out-dir", required=True, help="模型与 manifest 输出目录")
    ap.add_argument("--task", choices=["all", "t2", "t3"], default="all")
    ap.add_argument("--metadata-encoding", default="gbk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    for path in args.train_zip + args.eval_zip:
        if not os.path.isfile(path):
            print(f"找不到文件：{path}", file=sys.stderr)
            return 2
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "train_zips": [os.path.abspath(p) for p in args.train_zip],
        "eval_zips": [os.path.abspath(p) for p in args.eval_zip],
        "tasks": {},
    }

    tasks = []
    if args.task in {"all", "t2"}:
        tasks.append(ExamTask.T2)
    if args.task in {"all", "t3"}:
        tasks.append(ExamTask.T3)

    for task in tasks:
        X, y, counts, failures = _concat_packages(args.train_zip, task, args.metadata_encoding)
        print(f"[{task.value}] 开始训练：X={X.shape} y={y.shape}")
        if task is ExamTask.T2:
            bundle = train_magnitude_baseline(X, y, random_state=args.seed)
            model_name = "t2_magnitude_baseline.joblib"
            train_summary = {
                "samples": int(len(y)),
                "target_min": float(np.min(y)),
                "target_max": float(np.max(y)),
                "target_mean": float(np.mean(y)),
            }
        else:
            bundle = train_event_baseline(X, y, random_state=args.seed)
            model_name = "t3_event_baseline.joblib"
            train_summary = {
                "samples": int(len(y)),
                "class_distribution": {str(k): int(v) for k, v in sorted(Counter(int(v) for v in y).items())},
            }
        eval_reports = _evaluate(bundle, args.eval_zip, task, y, args.metadata_encoding)
        bundle.trained_on = [os.path.abspath(p) for p in args.train_zip]
        bundle.metrics = {"training": train_summary, "evaluation": eval_reports}
        model_path = os.path.join(args.out_dir, model_name)
        save_bundle(bundle, model_path)
        print(f"[{task.value}] 已保存模型：{model_path}")
        manifest["tasks"][task.value] = {
            "model": os.path.abspath(model_path),
            "training": train_summary,
            "samples_per_package": counts,
            "training_failures": failures,
            "evaluation": eval_reports,
        }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"完成。清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
