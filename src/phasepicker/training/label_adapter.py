"""标注格式适配层（Label adapter）—— 吸收"训练数据格式还不确定"的唯一入口。

===== 这个模块存在的理由 =====
到目前为止我们还不知道主办方给的标注长什么样：可能是每个 mseed 配一个
CSV，可能是一个大 CSV 列出所有 (文件, 台站, 震相, 到时)，到时可能是绝对
UTC 字符串、Unix 秒、也可能是"相对波形起点的秒数"。

如果把"读标注"的逻辑散落在数据集/训练循环里，格式一变就要改很多地方。
所以这里立一条铁律：**所有外部标注，先经过本模块归一化成内部统一的
LabelSet，训练代码只认 LabelSet。** 数据格式一旦明确，只改这一个文件。

===== 内部统一表示 =====
和推理侧保持一致：内部一律用"绝对 UTC 时间戳（float 秒）"表示到时，
与 phasepicker.types.Pick 对齐，这样训练标签和模型输出可以直接对比、
直接喂给同一个本地评分脚本。

===== 已内置两种最常见格式的解析器 =====
1. long_csv   —— 一个大 CSV，每行一条震相：file_id, station, phase, time
2. per_file   —— 每个波形一个同名 .csv/.json，内含若干震相

新格式只需再写一个 parse_xxx 函数并注册到 _PARSERS，其余代码零改动。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ----------------------------------------------------------------------------
# 内部统一标注表示
# ----------------------------------------------------------------------------

@dataclass
class LabelPick:
    """一条标注震相（训练真值）。

    Attributes:
        phase: "P" 或 "S"。
        time_utc: 绝对到时，Unix epoch 秒。与推理侧 Pick.time_utc 同口径。
        station: 台站标识 NET.STA，可空。
    """

    phase: str
    time_utc: float
    station: str = ""

    def __post_init__(self) -> None:
        if self.phase not in ("P", "S"):
            raise ValueError(f"标注震相类型必须是 P 或 S，收到 {self.phase!r}")


@dataclass
class LabelSet:
    """一个波形文件对应的全部标注震相。

    file_id 用于把标注和波形文件对应起来（通常是不含扩展名的文件名），
    是"标注↔波形"配对的主键。
    """

    file_id: str
    picks: List[LabelPick] = field(default_factory=list)

    def by_phase(self, phase: str) -> List[LabelPick]:
        return [p for p in self.picks if p.phase == phase]


# ----------------------------------------------------------------------------
# 到时字段的归一化：吸收"绝对 UTC / Unix 秒 / 相对偏移"三种可能
# ----------------------------------------------------------------------------

def normalize_time(
    raw_value: str | float,
    time_mode: str,
    starttime_utc: Optional[float] = None,
) -> float:
    """把标注里的到时字段统一成绝对 UTC 秒。

    Args:
        raw_value: 原始到时，可能是 ISO 字符串、Unix 秒、或相对偏移秒。
        time_mode: 三选一：
            - "utc_iso"   : ISO8601 字符串，如 "2024-01-01T00:00:03.5Z"
            - "unix"      : 已经是 Unix epoch 秒（float/字符串）
            - "relative"  : 相对波形起点的秒数，需配合 starttime_utc 换算
        starttime_utc: 当 time_mode="relative" 时必填，波形第一个采样点的
            绝对时间（Unix 秒）。这一步就是把相对偏移锚回绝对时间，避免后续
            所有到时系统性偏移——和推理侧强调的时间对齐是同一件事。

    Returns:
        绝对到时，Unix epoch 秒。

    Raises:
        ValueError: time_mode 非法，或 relative 模式下缺 starttime_utc。
    """
    if time_mode == "unix":
        return float(raw_value)

    if time_mode == "relative":
        if starttime_utc is None:
            raise ValueError("relative 模式必须提供 starttime_utc 才能换算为绝对时间")
        return starttime_utc + float(raw_value)

    if time_mode == "utc_iso":
        # 延迟导入，避免在纯逻辑测试环境强依赖 obspy
        try:
            from obspy import UTCDateTime  # type: ignore
        except ImportError as exc:  # pragma: no cover - 部署问题应尽早暴露
            raise ImportError(
                "utc_iso 模式需要 obspy（UTCDateTime）解析 ISO 时间字符串"
            ) from exc
        return float(UTCDateTime(raw_value).timestamp)

    raise ValueError(f"未知 time_mode={time_mode!r}，应为 utc_iso/unix/relative")


# ----------------------------------------------------------------------------
# 具体格式解析器（新增格式在此扩展）
# ----------------------------------------------------------------------------

def parse_long_csv(
    csv_path: str,
    time_mode: str = "unix",
    columns: Optional[Dict[str, str]] = None,
    starttimes: Optional[Dict[str, float]] = None,
) -> Dict[str, LabelSet]:
    """解析"一个大 CSV，每行一条震相"的格式。

    默认列名：file_id, station, phase, time。可用 columns 覆盖以适配官方列名，
    例如 columns={"file_id": "waveform", "time": "arrival_time"}。

    Args:
        csv_path: CSV 路径。
        time_mode: 传给 normalize_time。
        columns: 列名映射（内部名 -> CSV 实际列名），缺省用内部名。
        starttimes: 当 time_mode="relative" 时，提供 {file_id: starttime_utc}。

    Returns:
        {file_id: LabelSet}。
    """
    col = {"file_id": "file_id", "station": "station", "phase": "phase", "time": "time"}
    if columns:
        col.update(columns)

    out: Dict[str, LabelSet] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row[col["file_id"]]).strip()
            station = str(row.get(col["station"], "")).strip()
            phase = str(row[col["phase"]]).strip().upper()[:1]  # 容忍 "Pg"/"Pn" -> "P"
            st = starttimes.get(fid) if starttimes else None
            t = normalize_time(row[col["time"]], time_mode, st)
            out.setdefault(fid, LabelSet(file_id=fid))
            out[fid].picks.append(LabelPick(phase=phase, time_utc=t, station=station))
    return out


def parse_per_file(
    labels_dir: str,
    time_mode: str = "unix",
    starttimes: Optional[Dict[str, float]] = None,
) -> Dict[str, LabelSet]:
    """解析"每个波形一个同名 .json 标注"的格式。

    约定每个 json 形如：
        {"picks": [{"phase": "P", "time": 12.34, "station": "XX.STA"}, ...]}
    file_id 取自 json 文件名（去扩展名）。

    Args:
        labels_dir: 存放各标注文件的目录。
        time_mode: 传给 normalize_time。
        starttimes: relative 模式下的 {file_id: starttime_utc}。

    Returns:
        {file_id: LabelSet}。
    """
    out: Dict[str, LabelSet] = {}
    for name in os.listdir(labels_dir):
        if not name.endswith(".json"):
            continue
        fid = os.path.splitext(name)[0]
        with open(os.path.join(labels_dir, name), encoding="utf-8") as f:
            payload = json.load(f)
        st = starttimes.get(fid) if starttimes else None
        ls = LabelSet(file_id=fid)
        for item in payload.get("picks", []):
            phase = str(item["phase"]).strip().upper()[:1]
            t = normalize_time(item["time"], time_mode, st)
            ls.picks.append(
                LabelPick(phase=phase, time_utc=t, station=str(item.get("station", "")))
            )
        out[fid] = ls
    return out


# 格式注册表：官方格式确定后，优先在这里挂一个解析器，而不是改训练代码。
_PARSERS: Dict[str, Callable[..., Dict[str, LabelSet]]] = {
    "long_csv": parse_long_csv,
    "per_file": parse_per_file,
}


def load_labels(fmt: str, **kwargs) -> Dict[str, LabelSet]:
    """统一入口：按格式名分发到对应解析器。

    Args:
        fmt: "long_csv" / "per_file" / 或你后续注册的新格式名。
        **kwargs: 透传给对应解析器。

    Returns:
        {file_id: LabelSet}。

    Raises:
        ValueError: 未注册的格式名。
    """
    if fmt not in _PARSERS:
        raise ValueError(
            f"未知标注格式 {fmt!r}。已支持：{list(_PARSERS)}。"
            f"新格式请在 label_adapter._PARSERS 注册解析器。"
        )
    return _PARSERS[fmt](**kwargs)
