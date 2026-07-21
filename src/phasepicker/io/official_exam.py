"""官方输入扫描（Official exam input scanning）——纯标准库.

从官方给定的输入（普通目录 或 zip）里枚举 ``.mseed`` 文件，并根据路径判定
它属于哪个任务（T1/T2/T3），产出 ExamSample 列表。本模块**不读取 waveform
数据**——只做结构扫描与任务归类，波形读取交给 mseed_reader（依赖 ObsPy）。

任务判定规则（大小写不敏感，任一命中即可）：
    TASK01 / T1-Q  => T1
    TASK02 / T2-Q  => T2
    TASK03 / T3-Q  => T3
两轮比赛的目录命名不同（第1轮 TASKxx，第2轮 Tx-Q），这里同时覆盖。

zip 读取只枚举条目名（不解压落盘），并对可能的 GBK 条目名做还原。
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import List, Optional

from ..types import ExamSample, ExamTask
from .official_answers import normalize_file_id

# 路径关键字 → 任务。顺序无关，命中即返回。
_TASK_MARKERS = (
    (("task01", "t1-q"), ExamTask.T1),
    (("task02", "t2-q"), ExamTask.T2),
    (("task03", "t3-q"), ExamTask.T3),
)


def classify_task(path: str) -> Optional[ExamTask]:
    """根据路径判定任务类型；无法判定返回 None。

    先按目录关键字（TASKxx / Tx-Q）匹配；都不中时，退化为看 basename 前缀
    （``T1.``/``T2.``/``T3.``），覆盖答案样例里 ``T1.A.Q0001.mseed`` 这类命名。
    """
    low = str(path).replace("\\", "/").lower()
    for markers, task in _TASK_MARKERS:
        if any(mk in low for mk in markers):
            return task
    base = normalize_file_id(path).lower()
    for prefix, task in (("t1.", ExamTask.T1), ("t2.", ExamTask.T2), ("t3.", ExamTask.T3)):
        if base.startswith(prefix):
            return task
    return None


def _is_mseed(name: str) -> bool:
    return normalize_file_id(name).lower().endswith(".mseed")


def _decode_entry_name(info: zipfile.ZipInfo, metadata_encoding: str) -> str:
    """还原可能被 CP437 误解的中文 zip 条目名。"""
    name = info.filename
    if info.flag_bits & 0x800 == 0:
        try:
            return name.encode("cp437").decode(metadata_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def _scan_zip_file(
    zf: zipfile.ZipFile,
    outer_zip_path: str,
    raw_chain: List[str],
    display_chain: List[str],
    metadata_encoding: str,
    depth: int,
) -> List[ExamSample]:
    """递归扫描一个已打开的 zip；source_path 用 ``!`` 保存嵌套条目链。"""
    samples: List[ExamSample] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        raw_name = info.filename
        display_name = _decode_entry_name(info, metadata_encoding)

        if raw_name.lower().endswith(".zip") and depth > 0:
            try:
                raw = zf.read(info)
                with zipfile.ZipFile(io.BytesIO(raw), "r") as inner:
                    samples.extend(
                        _scan_zip_file(
                            inner,
                            outer_zip_path=outer_zip_path,
                            raw_chain=raw_chain + [raw_name],
                            display_chain=display_chain + [display_name],
                            metadata_encoding=metadata_encoding,
                            depth=depth - 1,
                        )
                    )
            except zipfile.BadZipFile:
                pass
            continue

        if not _is_mseed(display_name):
            continue
        display_path = "!".join(display_chain + [display_name])
        task = classify_task(display_path)
        if task is None:
            continue
        source_path = "!".join([outer_zip_path] + raw_chain + [raw_name])
        samples.append(
            ExamSample(
                file_id=normalize_file_id(display_name),
                task=task,
                source_path=source_path,
            )
        )
    return samples


def scan_directory(root: str) -> List[ExamSample]:
    """递归扫描目录下所有 ``.mseed``，产出 ExamSample 列表。

    Args:
        root: 输入根目录。

    Returns:
        ExamSample 列表（按 source_path 排序，便于复现）。无法判定任务的文件被跳过。
    """
    samples: List[ExamSample] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not _is_mseed(fn):
                continue
            full = os.path.join(dirpath, fn)
            task = classify_task(full)
            if task is None:
                continue
            samples.append(
                ExamSample(
                    file_id=normalize_file_id(fn),
                    task=task,
                    source_path=full,
                )
            )
    samples.sort(key=lambda s: s.source_path)
    return samples


def scan_zip(
    zip_path: str,
    metadata_encoding: str = "gbk",
    max_depth: int = 3,
) -> List[ExamSample]:
    """枚举 zip 内所有 ``.mseed`` 条目，产出 ExamSample 列表（**不解压落盘**）。

    对可能被 CP437 误解的中文条目名，用 metadata_encoding（默认 gbk）还原后
    再判定任务与取 basename。source_path 记为 ``zip_path!entry_name`` 形式，
    表明来源于 zip 内条目。

    Args:
        zip_path: zip 文件路径（只读）。
        metadata_encoding: 条目名推测编码，默认 "gbk"。

    Returns:
        ExamSample 列表（按 source_path 排序）。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        samples = _scan_zip_file(
            zf,
            outer_zip_path=zip_path,
            raw_chain=[],
            display_chain=[],
            metadata_encoding=metadata_encoding,
            depth=max_depth,
        )
    samples.sort(key=lambda s: s.source_path)
    return samples


def scan_exam_input(path: str, metadata_encoding: str = "gbk") -> List[ExamSample]:
    """统一入口：给定目录或 zip 路径，返回 ExamSample 列表。

    自动分流：``.zip`` 走 scan_zip，目录走 scan_directory。
    """
    if str(path).lower().endswith(".zip") and zipfile.is_zipfile(path):
        return scan_zip(path, metadata_encoding=metadata_encoding)
    return scan_directory(path)
