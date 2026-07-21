#!/usr/bin/env python
"""用已训练模型生成官方 T2.an 与 T3.an。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.submission_writer import write_task2_submission, write_task3_submission  # noqa: E402
from phasepicker.tasks.event_classifier import TrainedEventClassifier  # noqa: E402
from phasepicker.tasks.magnitude_task import TrainedMagnitudePredictor  # noqa: E402
from phasepicker.types import ExamTask  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成官方 T2/T3 提交答案")
    ap.add_argument("--input", required=True, help="官方输入目录或 zip（支持嵌套 zip）")
    ap.add_argument("--t2-model", help="t2_magnitude_baseline.joblib")
    ap.add_argument("--t3-model", help="t3_event_baseline.joblib")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--t2-prefix", default="./T2-Q/")
    ap.add_argument("--t3-prefix", default="./T3-Q/")
    ap.add_argument("--metadata-encoding", default="gbk")
    args = ap.parse_args(argv)

    if not args.t2_model and not args.t3_model:
        ap.error("--t2-model 与 --t3-model 至少给一个")
    if not os.path.exists(args.input):
        print(f"找不到输入：{args.input}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    samples = scan_exam_input(args.input, metadata_encoding=args.metadata_encoding)

    if args.t2_model:
        predictor = TrainedMagnitudePredictor(args.t2_model)
        t2_samples = [s for s in samples if s.task is ExamTask.T2]
        results = []
        failures: List[str] = []
        for i, sample in enumerate(t2_samples, 1):
            try:
                results.append(predictor.predict(sample))
            except Exception as exc:
                failures.append(f"{sample.file_id}: {type(exc).__name__}: {exc}")
            if i % 50 == 0 or i == len(t2_samples):
                print(f"T2 {i}/{len(t2_samples)}")
        out = os.path.join(args.output_dir, "T2.an")
        write_task2_submission(sorted(results, key=lambda r: r.file_id), out, prefix=args.t2_prefix)
        print(f"T2 写出 {len(results)} 条：{out}；失败 {len(failures)}")
        for line in failures[:5]:
            print(f"  - {line}")

    if args.t3_model:
        classifier = TrainedEventClassifier(args.t3_model)
        t3_samples = [s for s in samples if s.task is ExamTask.T3]
        results = []
        failures = []
        for i, sample in enumerate(t3_samples, 1):
            try:
                results.append(classifier.predict(sample))
            except Exception as exc:
                failures.append(f"{sample.file_id}: {type(exc).__name__}: {exc}")
            if i % 50 == 0 or i == len(t3_samples):
                print(f"T3 {i}/{len(t3_samples)}")
        out = os.path.join(args.output_dir, "T3.an")
        write_task3_submission(sorted(results, key=lambda r: r.file_id), out, prefix=args.t3_prefix)
        print(f"T3 写出 {len(results)} 条：{out}；失败 {len(failures)}")
        for line in failures[:5]:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
