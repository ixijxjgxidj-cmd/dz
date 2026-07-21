#!/usr/bin/env python3
"""真实官方 zip 只读校验工具（Official package validator）——纯标准库.

对着真实的官方比赛 zip，验证 official_exam（输入扫描）与 official_answers
（答案解析）在真实包上确实能工作：不依赖 obspy/seisbench，不解压落盘，只读
扫描 zip 条目名、按 T1/T2/T3 归类、解析答案文本、按 basename 对齐 input vs
answer，报出 missing/extra 与各任务的分布统计。

两轮包结构不同（都已在真实包上确认）：
  第1轮：单层 zip。答案 ``T1.an.txt / T2.an.txt / T3.an.txt`` 与 mseed
        （``TASKxx/`` 目录）同在一个 zip 里，中文根目录名多为 GBK。
  第2轮：外层 zip 内含两个子 zip——
        ``*answer*.zip``（内含 ``T1.an / T2.an / T3.an``）与
        ``*data*.zip``（内含 ``Tx-Q/`` 下的 mseed）。
        故需要**嵌套 zip 只读扫描**：把子 zip 的字节读进内存再当 zip 打开。

用法：
    python scripts/validate_official_package.py --zip <官方zip> [--round auto|round1|round2]
    python scripts/validate_official_package.py --zip r1.zip --answers-only
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 让脚本无需安装即可 import 到包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamTask, Task1Result, Task2Result, Task3Result  # noqa: E402
from phasepicker.io.official_exam import classify_task, _is_mseed  # noqa: E402
from phasepicker.io.official_answers import (  # noqa: E402
    normalize_file_id,
    parse_task1_answer_lines,
    parse_task2_answer_lines,
    parse_task3_answer_lines,
)


# =========================================================================
# 只读 zip 遍历：把（可能嵌套的）zip 拍平成条目列表
# =========================================================================
# 内存纪律：mseed 只需要计数与归类，绝不把它的内容读进来（第2轮 data zip 有
# 66MB+ 波形，全量固化会撑爆内存）。只有**答案文本**这类小条目才在 zip 仍打开
# 的作用域内 eager 读出字节存进 _Entry.data；mseed 的 data 恒为 None。


@dataclass
class _Entry:
    """zip 内一个文件条目的只读视图。"""

    display_name: str              # 人类可读的完整路径（含嵌套 zip 前缀）
    basename: str                  # normalize 后的 basename，用于对齐
    data: Optional[bytes] = None   # 仅小文本条目会固化字节；mseed 恒为 None


def _decode_entry_name(info: zipfile.ZipInfo, metadata_encoding: str) -> str:
    """还原可能被 CP437 误解的中文条目名（未标记 UTF-8 时按 metadata_encoding 试）。"""
    name = info.filename
    if info.flag_bits & 0x800 == 0:
        try:
            return name.encode("cp437").decode(metadata_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            return info.filename
    return name


def _iter_zip_entries(
    zf: zipfile.ZipFile,
    prefix: str,
    metadata_encoding: str,
    depth: int,
) -> List[_Entry]:
    """遍历一个已打开的 zip，产出文件条目；遇到内嵌 .zip 则递归下钻（只读）。

    对非 mseed 的普通文件（答案文本）当场读出字节固化，因为调用方 with 关闭后
    就读不到了；mseed 只留元信息（data=None），避免把大波形读进内存。
    depth 是防御性递归上限，官方包最多两层，给到 3 足够且能挡住异常自引用。
    """
    entries: List[_Entry] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = _decode_entry_name(info, metadata_encoding)
        display = f"{prefix}{name}" if prefix else name

        if name.lower().endswith(".zip") and depth > 0:
            # 嵌套 zip：读进内存再当 zip 打开（不落盘），下钻后即释放该字节
            raw = zf.read(info)
            try:
                with zipfile.ZipFile(io.BytesIO(raw), "r") as inner:
                    entries.extend(
                        _iter_zip_entries(
                            inner,
                            prefix=f"{display}!",
                            metadata_encoding=metadata_encoding,
                            depth=depth - 1,
                        )
                    )
                continue
            except zipfile.BadZipFile:
                # 不是真 zip（罕见），按普通文件对待，字节已在手
                entries.append(_Entry(display, normalize_file_id(name), raw))
                continue

        base = normalize_file_id(name)
        # mseed 不读内容（可能极大）；其余小文件（答案文本）eager 固化字节
        data = None if base.lower().endswith(".mseed") else zf.read(info)
        entries.append(_Entry(display, base, data))
    return entries


def collect_entries(zip_path: str, metadata_encoding: str = "gbk", max_depth: int = 3) -> List[_Entry]:
    """打开外层 zip，拍平出所有（含嵌套）文件条目（mseed 只留元信息，不读内容）。"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return _iter_zip_entries(zf, prefix="", metadata_encoding=metadata_encoding, depth=max_depth)


# =========================================================================
# 答案文件识别
# =========================================================================
# 两轮的答案文件名分别是 T1.an.txt / T1.an（T2/T3 同理）。用 basename 前缀匹配，
# 允许 .an 或 .an.txt 结尾，且不与 mseed 混淆。

_ANSWER_MARKERS = (
    ("t1.an", ExamTask.T1),
    ("t2.an", ExamTask.T2),
    ("t3.an", ExamTask.T3),
)


def find_answer_entries(entries: List[_Entry]) -> Dict[ExamTask, _Entry]:
    """在条目里找 T1/T2/T3 答案文件；每任务取第一个命中。"""
    out: Dict[ExamTask, _Entry] = {}
    for e in entries:
        low = e.basename.lower()
        if low.endswith(".mseed"):
            continue
        for marker, task in _ANSWER_MARKERS:
            if task in out:
                continue
            if low.startswith(marker):
                out[task] = e
    return out


def _read_lines(entry: _Entry, metadata_encoding: str = "gbk") -> List[str]:
    """把答案条目字节按 UTF-8→GBK→replace 解码成行。"""
    data = entry.data or b""
    for enc in ("utf-8", metadata_encoding):
        try:
            return data.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").splitlines()


# =========================================================================
# 各任务统计
# =========================================================================


@dataclass
class TaskStat:
    """单任务的输入/答案对齐统计。"""

    task: ExamTask
    n_input: int = 0
    n_answer: int = 0
    missing: List[str] = field(default_factory=list)   # answer 有、input 无
    extra: List[str] = field(default_factory=list)     # input 有、answer 无
    matched: int = 0
    detail: Dict[str, object] = field(default_factory=dict)  # 任务特有指标

    def diff(self, input_bases: set, answer_bases: set) -> None:
        self.matched = len(input_bases & answer_bases)
        self.missing = sorted(answer_bases - input_bases)
        self.extra = sorted(input_bases - answer_bases)


def _t1_detail(results: Dict[str, Task1Result]) -> Dict[str, object]:
    p_counts = [len(r.p_times_s) for r in results.values()]
    s_counts = [len(r.s_times_s) for r in results.values()]
    multi = sum(1 for r in results.values() if len(r.p_times_s) > 1 or len(r.s_times_s) > 1)
    return {
        "p_per_file_min": min(p_counts) if p_counts else 0,
        "p_per_file_max": max(p_counts) if p_counts else 0,
        "s_per_file_min": min(s_counts) if s_counts else 0,
        "s_per_file_max": max(s_counts) if s_counts else 0,
        "files_with_multi_ps": multi,
    }


def _t2_detail(results: Dict[str, Task2Result]) -> Dict[str, object]:
    mags = [r.magnitude for r in results.values()]
    if not mags:
        return {"mag_min": None, "mag_max": None, "mag_mean": None}
    return {
        "mag_min": min(mags),
        "mag_max": max(mags),
        "mag_mean": sum(mags) / len(mags),
    }


def _t3_detail(results: Dict[str, Task3Result]) -> Dict[str, object]:
    dist: Dict[int, int] = {}
    for r in results.values():
        dist[r.label] = dist.get(r.label, 0) + 1
    return {"class_dist": dict(sorted(dist.items()))}


# =========================================================================
# 主校验流程
# =========================================================================


@dataclass
class ValidationReport:
    zip_path: str
    round_kind: str
    n_entries: int
    n_mseed: int
    stats: Dict[ExamTask, TaskStat] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def detect_round(entries: List[_Entry]) -> str:
    """按条目布局猜测是第几轮：出现 ``Tx-Q`` 目录标记 → round2，否则 round1。"""
    for e in entries:
        low = e.display_name.replace("\\", "/").lower()
        if "t1-q" in low or "t2-q" in low or "t3-q" in low:
            return "round2"
    return "round1"


def validate_package(
    zip_path: str,
    round_kind: str = "auto",
    metadata_encoding: str = "gbk",
    answers_only: bool = False,
) -> ValidationReport:
    """对单个官方 zip 做只读校验，返回 ValidationReport。"""
    entries = collect_entries(zip_path, metadata_encoding=metadata_encoding)
    if round_kind == "auto":
        round_kind = detect_round(entries)

    n_mseed = sum(1 for e in entries if _is_mseed(e.basename))
    report = ValidationReport(
        zip_path=zip_path,
        round_kind=round_kind,
        n_entries=len(entries),
        n_mseed=n_mseed,
    )

    # ---- 输入 mseed 按任务归类 ----
    input_bases: Dict[ExamTask, set] = {t: set() for t in ExamTask}
    unclassified: List[str] = []
    for e in entries:
        if not _is_mseed(e.basename):
            continue
        task = classify_task(e.display_name)
        if task is None:
            unclassified.append(e.basename)
            continue
        input_bases[task].add(e.basename)
    if unclassified:
        report.notes.append(f"{len(unclassified)} 个 mseed 无法判定任务（示例：{unclassified[:3]}）")

    # ---- 答案文件解析 ----
    answer_entries = find_answer_entries(entries)
    for task in ExamTask:
        stat = TaskStat(task=task, n_input=len(input_bases[task]))

        ans_entry = answer_entries.get(task)
        if ans_entry is None:
            report.notes.append(f"{task.value} 未找到答案文件（T{task.value[-1]}.an[.txt]）")
            report.stats[task] = stat
            continue

        lines = _read_lines(ans_entry, metadata_encoding=metadata_encoding)
        if task is ExamTask.T1:
            parsed = parse_task1_answer_lines(lines)
            stat.detail = _t1_detail(parsed)
        elif task is ExamTask.T2:
            parsed = parse_task2_answer_lines(lines)
            stat.detail = _t2_detail(parsed)
        else:
            parsed = parse_task3_answer_lines(lines)
            stat.detail = _t3_detail(parsed)

        stat.n_answer = len(parsed)
        n_nonempty = sum(1 for ln in lines if ln.strip())
        if stat.n_answer < n_nonempty:
            report.notes.append(
                f"{task.value} 答案 {n_nonempty} 非空行只解析出 {stat.n_answer} 条（有行未匹配格式）"
            )

        if not answers_only:
            stat.diff(input_bases[task], set(parsed.keys()))
        report.stats[task] = stat

    return report


# =========================================================================
# 打印
# =========================================================================


def _fmt_num(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def print_report(report: ValidationReport, max_list: int = 5) -> None:
    print("=" * 72)
    print(f"包：{report.zip_path}")
    print(f"识别轮次：{report.round_kind}   条目总数：{report.n_entries}   mseed 总数：{report.n_mseed}")
    print("-" * 72)
    for task in ExamTask:
        stat = report.stats.get(task)
        if stat is None:
            continue
        print(f"[{task.value}] 输入={stat.n_input}  答案={stat.n_answer}  "
              f"匹配={stat.matched}  缺失={len(stat.missing)}  多余={len(stat.extra)}")
        for k, v in stat.detail.items():
            print(f"       {k} = {_fmt_num(v)}")
        if stat.missing:
            print(f"       missing(answer有input无) 示例: {stat.missing[:max_list]}")
        if stat.extra:
            print(f"       extra(input有answer无) 示例:   {stat.extra[:max_list]}")
    if report.notes:
        print("-" * 72)
        print("格式边界 / 备注：")
        for n in report.notes:
            print(f"  * {n}")
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="真实官方 zip 只读校验工具（不解压落盘）")
    ap.add_argument("--zip", required=True, help="官方 zip 路径")
    ap.add_argument("--round", dest="round_kind", default="auto",
                    choices=["auto", "round1", "round2"], help="轮次（默认 auto 自动识别）")
    ap.add_argument("--metadata-encoding", default="gbk", help="zip 条目名推测编码（默认 gbk）")
    ap.add_argument("--answers-only", action="store_true", help="只解析答案、跳过 input↔answer 对齐")
    ap.add_argument("--max-list", type=int, default=5, help="missing/extra 示例最多打印几个")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.zip):
        print(f"找不到 zip：{args.zip}", file=sys.stderr)
        return 2
    if not zipfile.is_zipfile(args.zip):
        print(f"不是合法 zip：{args.zip}", file=sys.stderr)
        return 2

    report = validate_package(
        args.zip,
        round_kind=args.round_kind,
        metadata_encoding=args.metadata_encoding,
        answers_only=args.answers_only,
    )
    print_report(report, max_list=args.max_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
