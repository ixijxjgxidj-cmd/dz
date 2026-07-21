"""官方提交文件写出（Submission writing）——纯标准库.

把 types.py 里的 Task{1,2,3}Result 写成官方要求的文本行。写出格式与
official_answers.py 的解析格式严格互逆——任何一条写出的行都应能被对应的
parser 原样读回（这一点由测试 round-trip 保证）。

关键约定：
- T1 输出的是**相对波形起点的秒**（float），不是 epoch 秒；上游负责换算。
- 每行前缀 prefix 是官方要求的路径前缀（如 ``exam2025/TASK01/`` 或 ``./T1-Q/``），
  与 file_id 拼成完整路径；prefix 为空时直接用 file_id。
"""

from __future__ import annotations

from typing import Iterable, List

from ..types import Task1Result, Task2Result, Task3Result


def _join_prefix(prefix: str, file_id: str) -> str:
    """拼接路径前缀与 file_id，规范化斜杠，避免出现双斜杠或缺斜杠。"""
    if not prefix:
        return file_id
    p = prefix.replace("\\", "/")
    if not p.endswith("/"):
        p = p + "/"
    return p + file_id


def _fmt_times(times: Iterable[float], ndigits: int = 2) -> str:
    """把到时列表格式化为分号分隔串，如 ``35.04;135.76``；空列表返回空串。"""
    return ";".join(f"{float(t):.{ndigits}f}" for t in times)


def format_task1_line(result: Task1Result, prefix: str = "", ndigits: int = 2) -> str:
    """格式化一条 T1 行：``<path> : P : p1;p2 : S : s1;s2``。

    采用第2轮的带空格冒号风格（``: P :``），它同时能被 parser 兼容第1轮读回。
    到时保留 ndigits 位小数（默认 2，与官方样例一致）。
    """
    path = _join_prefix(prefix, result.file_id)
    p_str = _fmt_times(result.p_times_s, ndigits)
    s_str = _fmt_times(result.s_times_s, ndigits)
    return f"{path} : P : {p_str} : S : {s_str}"


def format_task2_line(result: Task2Result, prefix: str = "", ndigits: int = 1) -> str:
    """格式化一条 T2 行：``<path>\\t<magnitude>``，震级默认一位小数。"""
    path = _join_prefix(prefix, result.file_id)
    return f"{path}\t{result.magnitude:.{ndigits}f}"


def format_task3_line(result: Task3Result, prefix: str = "") -> str:
    """格式化一条 T3 行：``<path>\\t<label>``，类别为整数。"""
    path = _join_prefix(prefix, result.file_id)
    return f"{path}\t{int(result.label)}"


def _write_lines(out_path: str, lines: List[str]) -> None:
    """把若干行写入文本文件（UTF-8，行尾统一 \\n）。"""
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def write_task1_results(
    results: Iterable[Task1Result],
    out_path: str,
    prefix: str = "",
    ndigits: int = 2,
) -> List[str]:
    """把一批 Task1Result 写成提交文件，返回写出的行（便于测试/日志）。"""
    lines = [format_task1_line(r, prefix, ndigits) for r in results]
    _write_lines(out_path, lines)
    return lines


# 别名：题面同时提到 write_task1_submission 与 write_task1_results，二者等价。
write_task1_submission = write_task1_results


def write_task2_submission(
    results: Iterable[Task2Result],
    out_path: str,
    prefix: str = "",
    ndigits: int = 1,
) -> List[str]:
    """把一批 Task2Result 写成提交文件，返回写出的行。"""
    lines = [format_task2_line(r, prefix, ndigits) for r in results]
    _write_lines(out_path, lines)
    return lines


def write_task3_submission(
    results: Iterable[Task3Result],
    out_path: str,
    prefix: str = "",
) -> List[str]:
    """把一批 Task3Result 写成提交文件，返回写出的行。"""
    lines = [format_task3_line(r, prefix) for r in results]
    _write_lines(out_path, lines)
    return lines
