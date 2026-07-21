"""T1 端到端 runner 的单元测试——纯标准库，不依赖 obspy/torch/seisbench.

用 mock Waveform + mock picker 验证 runner 的换算与容错语义：
- epoch 秒 → 相对秒（relative_s = pick.time_utc - waveform.starttime_utc）
- 多 P/S 支持
- 相对秒升序排序
- 负相对秒被过滤（并告警）
- 同一文件多 waveform（多台站）的 picks 合并进同一个 Task1Result
- 读取失败 / 空波形 / picker 抛错都降级为空 P/S，不崩溃
- 写出后能被 official_answers.parse_task1_answer_lines 原样读回（round-trip）

两种运行方式：
    pytest tests/test_task1_runner.py
    python  tests/test_task1_runner.py
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamSample, ExamTask, PhaseType, Pick, Task1Result, Waveform
from phasepicker.tasks.task1_runner import (
    pick_waveform_to_task1_result,
    picks_to_task1_result,
    run_task1_samples,
)
from phasepicker.io.submission_writer import write_task1_results
from phasepicker.io.official_answers import parse_task1_answer_lines


# ----------------------- mock 工具 -----------------------


class _FakeArray:
    """极简的 fake ndarray，只提供 Waveform.n_samples 需要的 .shape，
    避免为纯逻辑测试引入 numpy 依赖（虽然本环境有 numpy，但 runner 逻辑与 numpy 无关）。"""

    def __init__(self, n_channels: int, n_samples: int):
        self.shape = (n_channels, n_samples)


def _wf(starttime_utc: float, station: str = "NET.STA", n_samples: int = 3000) -> Waveform:
    return Waveform(
        data=_FakeArray(3, n_samples),
        sampling_rate=100.0,
        starttime_utc=starttime_utc,
        station=station,
    )


class _ListPicker:
    """mock picker：对任意波形返回预先设定的一批 Pick（List[Pick]）。

    支持按 station 返回不同的 picks，用于多台站合并测试。默认忽略波形内容。
    """

    def __init__(self, picks_by_station=None, default_picks=None):
        self._by_station = picks_by_station or {}
        self._default = default_picks or []

    def pick(self, wf: Waveform):
        return self._by_station.get(wf.station, self._default)


class _RaisingPicker:
    """mock picker：pick 时抛异常，验证 runner 的容错降级。"""

    def pick(self, wf: Waveform):
        raise RuntimeError("boom")


class _SinglePicker:
    """mock picker：返回单个 Pick（非列表），验证 _iter_picks 兼容单值。"""

    def __init__(self, pick):
        self._pick = pick

    def pick(self, wf: Waveform):
        return self._pick


# ----------------------- epoch → 相对秒 -----------------------


def test_epoch_to_relative_basic():
    start = 1_600_000_000.0
    picks = [
        Pick(phase=PhaseType.P, time_utc=start + 17.28, confidence=0.9, station="NET.STA"),
        Pick(phase=PhaseType.S, time_utc=start + 26.96, confidence=0.8, station="NET.STA"),
    ]
    r = picks_to_task1_result("T1.A.Q0001.mseed", picks, start)
    # float 在 1.6e9 量级相减有 ~1e-6 舍入，用近似断言（远小于 P=0.1s 计分容差）
    assert len(r.p_times_s) == 1 and abs(r.p_times_s[0] - 17.28) < 1e-4
    assert len(r.s_times_s) == 1 and abs(r.s_times_s[0] - 26.96) < 1e-4


def test_pick_waveform_end_to_end():
    start = 1_600_000_000.0
    picker = _ListPicker(default_picks=[
        Pick(phase=PhaseType.P, time_utc=start + 5.0, confidence=0.9, station="NET.STA"),
        Pick(phase=PhaseType.S, time_utc=start + 9.5, confidence=0.7, station="NET.STA"),
    ])
    r = pick_waveform_to_task1_result("T1.A.Q0001.mseed", _wf(start), picker)
    assert r.p_times_s == [5.0]
    assert r.s_times_s == [9.5]


# ----------------------- 多 P/S -----------------------


def test_multi_ps():
    start = 1000.0
    picks = [
        Pick(phase=PhaseType.P, time_utc=start + 35.04, confidence=0.9),
        Pick(phase=PhaseType.P, time_utc=start + 135.76, confidence=0.8),
        Pick(phase=PhaseType.S, time_utc=start + 41.78, confidence=0.7),
        Pick(phase=PhaseType.S, time_utc=start + 142.42, confidence=0.6),
    ]
    r = picks_to_task1_result("f.mseed", picks, start)
    assert len(r.p_times_s) == 2 and len(r.s_times_s) == 2
    assert abs(r.p_times_s[0] - 35.04) < 1e-6 and abs(r.p_times_s[1] - 135.76) < 1e-6
    assert abs(r.s_times_s[0] - 41.78) < 1e-6 and abs(r.s_times_s[1] - 142.42) < 1e-6


# ----------------------- 排序 -----------------------


def test_relative_times_sorted():
    start = 500.0
    # 故意乱序输入
    picks = [
        Pick(phase=PhaseType.P, time_utc=start + 30.0, confidence=0.5),
        Pick(phase=PhaseType.P, time_utc=start + 10.0, confidence=0.5),
        Pick(phase=PhaseType.S, time_utc=start + 40.0, confidence=0.5),
        Pick(phase=PhaseType.S, time_utc=start + 20.0, confidence=0.5),
    ]
    r = picks_to_task1_result("f.mseed", picks, start)
    assert r.p_times_s == [10.0, 30.0]
    assert r.s_times_s == [20.0, 40.0]


# ----------------------- 负相对秒过滤 + 告警 -----------------------


def test_negative_relative_filtered_and_warned(caplog=None):
    start = 1000.0
    picks = [
        Pick(phase=PhaseType.P, time_utc=start - 3.0, confidence=0.9),  # 负相对秒，应丢弃
        Pick(phase=PhaseType.P, time_utc=start + 4.0, confidence=0.9),
        Pick(phase=PhaseType.S, time_utc=start + 8.0, confidence=0.8),
    ]
    logger = logging.getLogger("phasepicker.tasks.task1_runner")
    records = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec)  # type: ignore
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        r = picks_to_task1_result("f.mseed", picks, start)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    # 负值被丢，只保留 +4.0
    assert r.p_times_s == [4.0]
    assert r.s_times_s == [8.0]
    # 至少产生一条告警
    assert any("负相对到时" in rec.getMessage() for rec in records)


# ----------------------- 多 waveform 合并为同一个 Task1Result -----------------------


def test_multi_waveform_merged_into_one_result():
    start = 2000.0
    picker = _ListPicker(picks_by_station={
        "NET.AAA": [Pick(phase=PhaseType.P, time_utc=start + 10.0, confidence=0.9, station="NET.AAA")],
        "NET.BBB": [
            Pick(phase=PhaseType.P, time_utc=start + 5.0, confidence=0.9, station="NET.BBB"),
            Pick(phase=PhaseType.S, time_utc=start + 12.0, confidence=0.8, station="NET.BBB"),
        ],
    })

    def load_fn(sample: ExamSample):
        return [_wf(start, station="NET.AAA"), _wf(start, station="NET.BBB")]

    sample = ExamSample(file_id="T1.A.Q0001.mseed", task=ExamTask.T1, source_path="x")
    out = run_task1_samples([sample], load_fn, picker)
    assert set(out) == {"T1.A.Q0001.mseed"}
    r = out["T1.A.Q0001.mseed"]
    # 两台站的 P 合并并排序：5.0, 10.0
    assert r.p_times_s == [5.0, 10.0]
    assert r.s_times_s == [12.0]


# ----------------------- 容错：读取失败 / 空波形 / picker 抛错 -----------------------


def test_load_failure_returns_empty():
    def load_fn(sample: ExamSample):
        raise IOError("cannot read")

    sample = ExamSample(file_id="bad.mseed", task=ExamTask.T1, source_path="x")
    out = run_task1_samples([sample], load_fn, _ListPicker())
    assert out["bad.mseed"].p_times_s == []
    assert out["bad.mseed"].s_times_s == []


def test_empty_waveforms_returns_empty():
    def load_fn(sample: ExamSample):
        return []

    sample = ExamSample(file_id="empty.mseed", task=ExamTask.T1, source_path="x")
    out = run_task1_samples([sample], load_fn, _ListPicker())
    assert out["empty.mseed"].p_times_s == []
    assert out["empty.mseed"].s_times_s == []


def test_picker_exception_returns_empty():
    start = 100.0

    def load_fn(sample: ExamSample):
        return [_wf(start)]

    sample = ExamSample(file_id="boom.mseed", task=ExamTask.T1, source_path="x")
    out = run_task1_samples([sample], load_fn, _RaisingPicker())
    assert out["boom.mseed"].p_times_s == []
    assert out["boom.mseed"].s_times_s == []


# ----------------------- picker 返回单值兼容 -----------------------


def test_single_pick_return_supported():
    start = 100.0
    picker = _SinglePicker(Pick(phase=PhaseType.P, time_utc=start + 7.0, confidence=0.9))
    r = pick_waveform_to_task1_result("f.mseed", _wf(start), picker)
    assert r.p_times_s == [7.0]
    assert r.s_times_s == []


# ----------------------- round-trip：写出后能被 parser 读回 -----------------------


def test_roundtrip_write_then_parse():
    start = 1000.0
    picks = [
        Pick(phase=PhaseType.P, time_utc=start + 35.04, confidence=0.9),
        Pick(phase=PhaseType.P, time_utc=start + 135.76, confidence=0.8),
        Pick(phase=PhaseType.S, time_utc=start + 41.78, confidence=0.7),
    ]
    r = picks_to_task1_result("T1.A.Q0001.mseed", picks, start)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "T1.an")
        write_task1_results([r], out_path, prefix="exam2025/TASK01/")
        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    parsed = parse_task1_answer_lines(lines)
    assert set(parsed) == {"T1.A.Q0001.mseed"}
    rp = parsed["T1.A.Q0001.mseed"]
    # 写出保留两位小数，读回应与原值在两位小数精度上一致
    assert rp.p_times_s == [35.04, 135.76]
    assert rp.s_times_s == [41.78]


# ----------------------------- standalone runner -----------------------------

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    lines = []
    passed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            lines.append(f"FAIL {fn.__name__}: {exc!r}")
            continue
        passed += 1
        lines.append(f"PASS {fn.__name__}")
    lines.append(f"SUMMARY {passed}/{len(fns)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_run_all())
