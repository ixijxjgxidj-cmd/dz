"""官方答案文件解析（Official answer parsing）——纯标准库，零第三方依赖.

===== 为什么单独一层 =====
去年官方两轮比赛的答案是**纯文本行**，每行一个文件的结果，但格式在两轮之间
有差异（尤其 T1 的分隔符与 P/S 数量）。把"文本 → 结构化 Result"的解析集中在
这里，下游评估/写出都吃 types.py 里的结构，官方格式一变只改这一层。

===== 已确认的官方格式 =====
T1（震相到时，单位=相对波形起点秒）：
  第1轮，每文件一个 P、一个 S，冒号无空格：
    ``exam2025/TASK01/T1.A.Q0001.mseed :P:   17.28 :S:   26.96``
  第2轮，允许多个 P/S 用分号分隔，冒号带空格：
    ``./T1-Q/T1.A.Q0001.mseed : P : 35.04;135.76 : S : 41.78;142.42``
  → 解析必须同时兼容 ``:P:`` 与 ``: P :``，以及单/多值。

T2（震级）：
    ``exam2025/TASK02/T2.A.Q0001.mseed       4.3``

T3（事件类别，1..5 整数）：
    ``exam2025/TASK03/T3.A.Q0001.mseed       1``

===== 编码坑 =====
第1轮 zip 的中文根目录可能是 GBK 编码，读文本时给 read_text_from_zip 传
``metadata_encoding="gbk"``；文件名匹配统一走 basename，绕开根目录中文差异。
"""

from __future__ import annotations

import re
import zipfile
from typing import Dict, Iterable, List, Optional

from ..types import EventClass, Task1Result, Task2Result, Task3Result


def normalize_file_id(path: str) -> str:
    """把任意路径统一成对齐用的 file_id，默认取 basename。

    答案路径（如 ``exam2025/TASK01/T1.A.Q0001.mseed``）与输入扫描到的路径
    往往根目录不同，但文件名一致。取 basename 作为唯一键最稳。同时兼容
    Windows 反斜杠与 POSIX 正斜杠，并去除首尾空白。

    Args:
        path: 原始路径字符串（可能含目录、正/反斜杠、首尾空白）。

    Returns:
        basename，如 ``T1.A.Q0001.mseed``。
    """
    s = str(path).strip().replace("\\", "/")
    # 去掉可能的尾部斜杠后取最后一段
    s = s.rstrip("/")
    return s.rsplit("/", 1)[-1] if "/" in s else s


def _split_semicolon_floats(field: str) -> List[float]:
    """把 ``35.04;135.76`` 这样的分号串解析为 float 列表；空串返回空列表。"""
    field = field.strip()
    if not field:
        return []
    out: List[float] = []
    for tok in field.split(";"):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


# T1 行的宽松正则：
#   group('path'): 冒号/P 标签之前的路径部分
#   group('p'):    P 与 S 标签之间的到时串
#   group('s'):    S 标签之后的到时串
# 同时兼容 ``:P:`` 与 ``: P :``（标签左右可有任意空白），P/S 大小写不敏感。
_T1_RE = re.compile(
    r"^\s*(?P<path>.+?)\s*:\s*P\s*:\s*(?P<p>[^:]*?)\s*:\s*S\s*:\s*(?P<s>[^:]*?)\s*$",
    re.IGNORECASE,
)


def parse_task1_answer_lines(lines: Iterable[str]) -> Dict[str, Task1Result]:
    """解析 T1 答案文本行为 {file_id: Task1Result}。

    兼容第1轮（单 P/S、``:P:`` 无空格）与第2轮（多 P/S 分号分隔、``: P :`` 带空格）。
    空行与无法匹配的行被跳过（不抛异常，答案文件里可能夹杂注释/空行）。

    Args:
        lines: 可迭代的文本行。

    Returns:
        以 basename 为键的 Task1Result 字典。
    """
    results: Dict[str, Task1Result] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _T1_RE.match(line)
        if not m:
            continue
        file_id = normalize_file_id(m.group("path"))
        results[file_id] = Task1Result(
            file_id=file_id,
            p_times_s=_split_semicolon_floats(m.group("p")),
            s_times_s=_split_semicolon_floats(m.group("s")),
        )
    return results


# T2/T3 行：``<path><空白><数值>``。路径可能含空格？官方样例路径不含空格，
# 故采用"最后一个空白分隔的 token 是数值，其余是路径"的稳健切法。
def _split_path_and_value(line: str) -> Optional[tuple]:
    """把 ``path   value`` 切成 (path, value_str)；切不出则返回 None。"""
    parts = line.rsplit(None, 1)  # 从右按任意空白切一次
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_task2_answer_lines(lines: Iterable[str]) -> Dict[str, Task2Result]:
    """解析 T2 答案文本行为 {file_id: Task2Result}。

    每行形如 ``exam2025/TASK02/T2.A.Q0001.mseed  4.3``；数值解析为 float。
    非法行（缺数值/数值非法）被跳过。
    """
    results: Dict[str, Task2Result] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        pair = _split_path_and_value(line)
        if pair is None:
            continue
        path, value = pair
        try:
            mag = float(value)
        except ValueError:
            continue
        file_id = normalize_file_id(path)
        results[file_id] = Task2Result(file_id=file_id, magnitude=mag)
    return results


def parse_task3_answer_lines(lines: Iterable[str]) -> Dict[str, Task3Result]:
    """解析 T3 答案文本行为 {file_id: Task3Result}。

    两轮格式差异（均已在真实官方包上确认）：
      第1轮：``exam2025/TASK03/T3.A.Q0001.mseed    2``（路径 + 数字类别）
      第2轮：``./T3-Q/T3.A.Q0001.mseed  1 earquake``（数字类别后**多一个文字标签**）
    因此不能简单取"最后一个 token"当类别（第2轮会取到 ``earquake``）。策略：
    第一个 token 作路径，其余 token 里取**第一个落在 1..5 的整数**当类别；
    找不到合法类别的行被跳过（宽松容错，避免脏行污染整批）。
    """
    valid = {int(c) for c in EventClass}
    results: Dict[str, Task3Result] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        path = tokens[0]
        label: Optional[int] = None
        for tok in tokens[1:]:
            try:
                cand = int(tok)
            except ValueError:
                continue
            if cand in valid:
                label = cand
                break
        if label is None:
            continue
        file_id = normalize_file_id(path)
        results[file_id] = Task3Result(file_id=file_id, label=label)
    return results


def read_text_from_zip(
    zip_path: str,
    entry_name: str,
    metadata_encoding: str = "gbk",
) -> List[str]:
    """从 zip 中读取一个文本条目并按行返回（**只读，不解压落盘**）。

    第1轮 zip 的中文条目名可能是 GBK 编码，ZipFile 默认按 CP437 解释非 UTF-8
    文件名，会导致按名查找失败。这里的策略：
      1. 先尝试 metadata_encoding（默认 gbk）重新解释每个条目名做匹配；
      2. 再退化为 basename 匹配（绕开根目录中文差异）。
    文本内容本身按 UTF-8 读、失败退 GBK，最后 errors='replace' 兜底。

    Args:
        zip_path: zip 文件路径（不会被修改或解压到磁盘）。
        entry_name: 想读取的条目名（可只给 basename）。
        metadata_encoding: 条目名的推测编码，默认 "gbk"。

    Returns:
        文本按行拆分后的列表（不含换行符）。

    Raises:
        KeyError: 找不到匹配条目。
    """
    target = normalize_file_id(entry_name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        chosen = None
        for info in zf.infolist():
            raw_name = info.filename
            # 尝试用 metadata_encoding 还原可能被 CP437 误解的中文名
            decoded_name = raw_name
            if info.flag_bits & 0x800 == 0:  # 未标记 UTF-8，可能是 GBK
                try:
                    decoded_name = raw_name.encode("cp437").decode(metadata_encoding)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    decoded_name = raw_name
            if normalize_file_id(decoded_name) == target or normalize_file_id(raw_name) == target:
                chosen = info
                break
        if chosen is None:
            raise KeyError(f"zip 中找不到条目：{entry_name!r}（basename={target!r}）")
        data = zf.read(chosen)

    for enc in ("utf-8", metadata_encoding):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    return text.splitlines()
