#!/usr/bin/env python3
"""本地评分 CLI —— 独立于 API 的"体温计"，用于跑批测试与阈值调优。

用法：
    python scripts/run_local_scoring.py --pred preds.json --truth truth.json
    python scripts/run_local_scoring.py --pred-dir out/ --truth-dir answers/

输入 JSON 格式（每个文件一个对象，或按文件名索引的字典）：
    单文件：  [{"phase": "P", "time_utc": 1600000000.12}, {"phase": "S", ...}]
    多文件：  {"filename1": [ {...}, ... ], "filename2": [ ... ]}

设计原则见 src/phasepicker/scoring/scorer.py 头部。此脚本只做 I/O 与汇总，
所有计分逻辑复用已被单元测试覆盖的 scorer 模块，避免重复实现导致口径漂移。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

# 让脚本无需安装即可 import 到包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.scoring.scorer import score_file, ScoreReport  # noqa: E402


def _load_picks(path: str) -> List[Tuple[str, float]]:
    """把一个 JSON 文件解析为 [(phase, time_utc), ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _coerce_picks(data)


def _coerce_picks(data) -> List[Tuple[str, float]]:
    """兼容多种字段命名，容错地抽出 (phase, time)。

    官方到时格式尚未确定，这里对常见字段名都做兼容：
    phase / type / label ；time_utc / time / arrival / t 。
    官方规范一到，只需在这里（或 api/adapters.py）加一个映射即可。
    """
    out: List[Tuple[str, float]] = []
    for item in data:
        phase = item.get("phase") or item.get("type") or item.get("label")
        t = (
            item.get("time_utc")
            if item.get("time_utc") is not None
            else item.get("time", item.get("arrival", item.get("t")))
        )
        if phase is None or t is None:
            continue
        phase = str(phase).upper()
        if phase not in ("P", "S"):
            continue
        out.append((phase, float(t)))
    return out


def _load_map(path_or_dir: str, is_dir: bool) -> Dict[str, List[Tuple[str, float]]]:
    """加载为 {文件名: picks} 的字典，统一单文件 / 目录 / 多文件字典三种情况。"""
    result: Dict[str, List[Tuple[str, float]]] = {}
    if is_dir:
        for p in sorted(glob.glob(os.path.join(path_or_dir, "*.json"))):
            result[os.path.splitext(os.path.basename(p))[0]] = _load_picks(p)
        return result

    with open(path_or_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            result[k] = _coerce_picks(v)
    else:  # 单文件的 list
        result["__single__"] = _coerce_picks(data)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本地震相评分（严格复刻官方规则）")
    ap.add_argument("--pred", help="预测 JSON 文件")
    ap.add_argument("--truth", help="真值 JSON 文件")
    ap.add_argument("--pred-dir", help="预测 JSON 目录（每文件一个）")
    ap.add_argument("--truth-dir", help="真值 JSON 目录（每文件一个）")
    ap.add_argument("--json-out", help="把逐文件明细写出到该 JSON 路径", default=None)
    args = ap.parse_args(argv)

    if args.pred_dir and args.truth_dir:
        preds = _load_map(args.pred_dir, is_dir=True)
        truths = _load_map(args.truth_dir, is_dir=True)
    elif args.pred and args.truth:
        preds = _load_map(args.pred, is_dir=False)
        truths = _load_map(args.truth, is_dir=False)
    else:
        ap.error("请提供 (--pred 与 --truth) 或 (--pred-dir 与 --truth-dir)")
        return 2

    keys = sorted(set(preds) | set(truths))
    reports: Dict[str, ScoreReport] = {}
    total = 0.0
    print("=" * 88)
    for k in keys:
        rep = score_file(preds.get(k, []), truths.get(k, []))
        reports[k] = rep
        total += rep.total_score
        print(f"[{k}] {rep.summary()}")
    print("=" * 88)

    n = len(keys) or 1
    print(f"文件数={len(keys)}  总分合计={total:.3f}  平均分={total / n:.4f}")

    # 汇总误差分布，供调阈值参考
    all_p = [r for rep in reports.values() for r in rep.p_residuals]
    all_s = [r for rep in reports.values() for r in rep.s_residuals]
    if all_p:
        import numpy as np

        ap_arr = np.asarray(all_p)
        print(
            f"P 残差: n={len(all_p)} mean={ap_arr.mean():.3f}s "
            f"p50={np.median(ap_arr):.3f}s p90={np.percentile(ap_arr, 90):.3f}s "
            f"命中0.1s比例={(ap_arr <= 0.1).mean():.1%}"
        )
    if all_s:
        import numpy as np

        as_arr = np.asarray(all_s)
        print(
            f"S 残差: n={len(all_s)} mean={as_arr.mean():.3f}s "
            f"p50={np.median(as_arr):.3f}s p90={np.percentile(as_arr, 90):.3f}s "
            f"命中0.2s比例={(as_arr <= 0.2).mean():.1%}"
        )

    if args.json_out:
        detail = {
            k: {
                "total_score": rep.total_score,
                "p_time_score": rep.p_time_score,
                "s_time_score": rep.s_time_score,
                "count_penalty": rep.count_penalty,
                "p_residuals": rep.p_residuals,
                "s_residuals": rep.s_residuals,
                "n_false_pos": rep.n_false_pos,
                "n_false_neg": rep.n_false_neg,
            }
            for k, rep in reports.items()
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"total": total, "files": detail}, f, ensure_ascii=False, indent=2)
        print(f"逐文件明细已写出：{args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
