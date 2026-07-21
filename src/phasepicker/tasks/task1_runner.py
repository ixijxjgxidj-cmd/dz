"""T1 端到端 runner（Phase arrival end-to-end runner）.

把官方 ``.mseed`` 输入真正接到现有 PhaseNet picker，并把内部 ``Pick``
（绝对 epoch 秒）换算成官方 T1 要求的 **相对波形起点秒**，产出 Task1Result。

===== 两套时间坐标（务必分清）=====
- 内部真理来源：``Pick.time_utc`` 是绝对 Unix epoch 秒（见 types.py 的设计说明）。
- 官方 T1 答案：P/S 到时是"相对波形第一个采样点的秒"（float）。
换算只发生在这一层，且只有一步：

    relative_s = pick.time_utc - waveform.starttime_utc

上游（picker / mseed_reader）与下游（submission_writer / official_eval）都不碰
这次换算，坐标系边界清晰，官方格式变动也不波及推理逻辑。

===== 容错哲学（延续 mseed_reader）=====
"宁可某个文件返回空 P/S，也绝不让整批崩掉。" 读波形失败、picker 抛错、或干脆
没有任何 pick，都以"空 Task1Result + 一条 warning"收场——空结果在官方评分里
只通过数量误差项体现，不会污染其它文件。
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Sequence, Union

from ..types import ExamSample, PhaseType, Pick, Task1Result, Waveform

logger = logging.getLogger(__name__)

# picker.pick 的返回既可能是单个 Pick，也可能是 List[Pick]（SeisBenchPicker 返回列表）。
# 用一个联合类型统一承接，_iter_picks 负责摊平成 List[Pick]。
PickResult = Union[Pick, Iterable[Pick], None]

# load_waveforms_fn 契约：吃一个 ExamSample，吐该文件里的所有 Waveform（可能多台站）。
# 真实实现基于 ObsPy（mseed_reader）；测试可注入返回 mock Waveform 的纯函数。
LoadWaveformsFn = Callable[[ExamSample], Sequence[Waveform]]


def _iter_picks(result: PickResult) -> List[Pick]:
    """把 picker.pick 的返回统一摊平成 List[Pick]，兼容单值 / 可迭代 / None。"""
    if result is None:
        return []
    if isinstance(result, Pick):
        return [result]
    return list(result)


def picks_to_task1_result(
    file_id: str,
    picks: Iterable[Pick],
    waveform_starttime_utc: float,
) -> Task1Result:
    """把一组绝对时间 Pick 换算成相对秒并归入 Task1Result（纯逻辑，无 I/O）。

    这是换算的唯一落点：每个 Pick 相对 ``waveform_starttime_utc`` 求偏移秒，
    按 P/S 分流，丢弃负偏移（pick 落在波形起点之前——理论上不该出现，出现即
    数据/对齐异常，宁可丢弃也不上报一个荒谬到时），最后各自升序排序。

    Args:
        file_id: 文件标识（basename），作为对齐用的唯一键。
        picks: 内部 Pick 列表（time_utc 为绝对 epoch 秒）。
        waveform_starttime_utc: 波形第一个采样点的绝对时间（epoch 秒）。

    Returns:
        Task1Result，p_times_s / s_times_s 为相对秒（float，升序）。
    """
    p_times: List[float] = []
    s_times: List[float] = []
    for pick in picks:
        rel = float(pick.time_utc) - float(waveform_starttime_utc)
        if rel < 0.0:
            logger.warning(
                "丢弃负相对到时 [%s] phase=%s rel=%.4fs（pick 落在波形起点之前）",
                file_id,
                getattr(pick.phase, "value", pick.phase),
                rel,
            )
            continue
        if pick.phase == PhaseType.P:
            p_times.append(rel)
        elif pick.phase == PhaseType.S:
            s_times.append(rel)
    p_times.sort()
    s_times.sort()
    return Task1Result(file_id=file_id, p_times_s=p_times, s_times_s=s_times)


def pick_waveform_to_task1_result(
    file_id: str,
    waveform: Waveform,
    picker,
) -> Task1Result:
    """对单个 Waveform 跑 picker，产出相对秒的 Task1Result。

    picker 抛错不会外泄——捕获后返回空 P/S 并记一条 warning，保证批处理不中断。

    Args:
        file_id: 文件标识（basename）。
        waveform: 预处理后的单台站三分量波形（携带 starttime_utc 锚点）。
        picker: 任意实现 ``pick(waveform) -> Pick | List[Pick]`` 的拾取器。

    Returns:
        Task1Result（相对秒，升序）。
    """
    try:
        picks = _iter_picks(picker.pick(waveform))
    except Exception as exc:  # noqa: BLE001 —— 单文件推理失败不该拖垮整批
        logger.warning("picker 对 [%s] 推理失败：%r（返回空结果）", file_id, exc)
        return Task1Result(file_id=file_id)
    return picks_to_task1_result(file_id, picks, waveform.starttime_utc)


def _merge_task1_results(file_id: str, parts: Iterable[Task1Result]) -> Task1Result:
    """把同一文件内多个波形（多台站）的 Task1Result 合并为一个。

    一个 .mseed 文件可能含多台站；官方以"文件"为最小评测单位，故所有台站的
    P/S 都并入同一个 Task1Result。合并后各自重新升序排序，保证输出稳定。
    """
    p_times: List[float] = []
    s_times: List[float] = []
    for part in parts:
        p_times.extend(part.p_times_s)
        s_times.extend(part.s_times_s)
    p_times.sort()
    s_times.sort()
    return Task1Result(file_id=file_id, p_times_s=p_times, s_times_s=s_times)


def run_task1_samples(
    samples: Iterable[ExamSample],
    load_waveforms_fn: LoadWaveformsFn,
    picker,
) -> Dict[str, Task1Result]:
    """批处理入口：对一批 ExamSample 逐个读波形、推理、换算，汇总成结果字典。

    对每个样本：
      1. 用 load_waveforms_fn 读出该文件的所有 Waveform（可能多台站）；
      2. 逐波形跑 picker，得到各自的相对秒 Task1Result；
      3. 把同文件多台站的结果合并为一个 Task1Result。
    读取失败或空波形/空 pick，都返回空 P/S（不崩溃），并记 warning。

    Args:
        samples: 待处理的 ExamSample 列表（通常来自 official_exam.scan_exam_input，
            并已用 task==T1 过滤；非 T1 样本此处不做额外拦截，交由调用方筛选）。
        load_waveforms_fn: 读波形的可注入函数，吃 ExamSample 吐 Sequence[Waveform]。
        picker: 拾取器（实现 pick(waveform)）。

    Returns:
        {file_id: Task1Result}。每个输入样本都有一条对应结果（哪怕是空 P/S）。
    """
    results: Dict[str, Task1Result] = {}
    for sample in samples:
        file_id = sample.file_id
        try:
            waveforms = load_waveforms_fn(sample)
        except Exception as exc:  # noqa: BLE001 —— 读取失败降级为空结果
            logger.warning("读取波形失败 [%s]（来源 %s）：%r", file_id, sample.source_path, exc)
            results[file_id] = Task1Result(file_id=file_id)
            continue

        if not waveforms:
            logger.warning("波形为空 [%s]（来源 %s）：返回空 P/S", file_id, sample.source_path)
            results[file_id] = Task1Result(file_id=file_id)
            continue

        parts = [
            pick_waveform_to_task1_result(file_id, wf, picker)
            for wf in waveforms
        ]
        results[file_id] = _merge_task1_results(file_id, parts)
    return results
