"""官方比赛包中的波形与答案读取。

支持三种来源：
1. 普通 ``.mseed`` 文件；
2. 外层 zip 里的 mseed；
3. 外层 zip → 内层 zip → mseed（第 2 轮真实布局）。

``source_path`` 使用 ``!`` 分隔归档层级，例如：
``round2.zip!exam-data07.zip!exam-data07/T2-Q/T2.Q0001.mseed``。
所有读取均为只读，不把官方大包解压到磁盘。
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, Iterable, Optional, Union

from ..types import ExamTask, Task1Result, Task2Result, Task3Result
from .official_answers import (
    normalize_file_id,
    parse_task1_answer_lines,
    parse_task2_answer_lines,
    parse_task3_answer_lines,
)


def _decode_entry_name(info: zipfile.ZipInfo, metadata_encoding: str) -> str:
    name = info.filename
    if info.flag_bits & 0x800 == 0:
        try:
            return name.encode("cp437").decode(metadata_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def _find_info(zf: zipfile.ZipFile, wanted: str, metadata_encoding: str) -> zipfile.ZipInfo:
    """按原始名或还原后的名字查找条目。"""
    try:
        return zf.getinfo(wanted)
    except KeyError:
        pass
    wanted_norm = wanted.replace("\\", "/")
    for info in zf.infolist():
        decoded = _decode_entry_name(info, metadata_encoding)
        if info.filename.replace("\\", "/") == wanted_norm:
            return info
        if decoded.replace("\\", "/") == wanted_norm:
            return info
    raise KeyError(f"zip 中找不到条目：{wanted}")


def read_source_bytes(source_path: str, metadata_encoding: str = "gbk") -> bytes:
    """读取普通文件或 ``!`` 嵌套归档链指向的最终文件字节。"""
    parts = str(source_path).split("!")
    if len(parts) == 1:
        with open(source_path, "rb") as f:
            return f.read()

    outer, chain = parts[0], parts[1:]
    with zipfile.ZipFile(outer, "r") as zf:
        current: Union[zipfile.ZipFile, None] = zf
        owned: list[zipfile.ZipFile] = []
        try:
            for index, name in enumerate(chain):
                assert current is not None
                info = _find_info(current, name, metadata_encoding)
                raw = current.read(info)
                if index == len(chain) - 1:
                    return raw
                inner = zipfile.ZipFile(io.BytesIO(raw), "r")
                owned.append(inner)
                current = inner
        finally:
            for inner in reversed(owned):
                inner.close()
    raise ValueError(f"无效 source_path：{source_path}")


def read_mseed_stream(source_path: str, metadata_encoding: str = "gbk"):
    """用 ObsPy 读取 source_path 指向的 mseed，返回 ``obspy.Stream``。

    ObsPy 延迟导入：纯格式解析/单元测试环境没有安装 ObsPy 时，其他模块仍能 import。
    """
    try:
        from obspy import read
    except ImportError as exc:
        raise RuntimeError("读取真实 mseed 需要安装 obspy：python -m pip install obspy") from exc

    if "!" not in str(source_path):
        return read(source_path, format="MSEED")
    return read(io.BytesIO(read_source_bytes(source_path, metadata_encoding)), format="MSEED")


_ANSWER_PREFIX = {
    ExamTask.T1: "t1.an",
    ExamTask.T2: "t2.an",
    ExamTask.T3: "t3.an",
}


def _iter_small_entries(
    zf: zipfile.ZipFile,
    metadata_encoding: str,
    depth: int,
) -> Iterable[tuple[str, bytes]]:
    """递归遍历答案等小文件；mseed 不读入内存。"""
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = _decode_entry_name(info, metadata_encoding)
        if info.filename.lower().endswith(".zip") and depth > 0:
            try:
                with zipfile.ZipFile(io.BytesIO(zf.read(info)), "r") as inner:
                    yield from _iter_small_entries(inner, metadata_encoding, depth - 1)
            except zipfile.BadZipFile:
                pass
            continue
        if not normalize_file_id(name).lower().endswith(".mseed"):
            yield name, zf.read(info)


def read_package_answers(
    zip_path: str,
    task: ExamTask,
    metadata_encoding: str = "gbk",
    max_depth: int = 3,
) -> Dict[str, Union[Task1Result, Task2Result, Task3Result]]:
    """从单层或嵌套官方 zip 中找到指定任务答案并解析。"""
    marker = _ANSWER_PREFIX[task]
    chosen: Optional[bytes] = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name, raw in _iter_small_entries(zf, metadata_encoding, max_depth):
            if normalize_file_id(name).lower().startswith(marker):
                chosen = raw
                break
    if chosen is None:
        raise KeyError(f"官方包中找不到 {task.value} 答案文件（{marker}[.txt]）")

    text: Optional[str] = None
    for enc in ("utf-8", metadata_encoding):
        try:
            text = chosen.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = chosen.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if task is ExamTask.T1:
        return parse_task1_answer_lines(lines)
    if task is ExamTask.T2:
        return parse_task2_answer_lines(lines)
    return parse_task3_answer_lines(lines)
