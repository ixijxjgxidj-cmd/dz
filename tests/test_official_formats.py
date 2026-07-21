"""官方比赛任务层（T1/T2/T3）的解析 / 写出 / 评估测试——纯标准库.

覆盖：
- 第1轮 T1 单 P/S 解析（``:P:`` 无空格）
- 第2轮 T1 多 P/S 解析（``: P :`` 带空格、分号分隔）
- T2 / T3 解析与 EventClass
- submission writer 写出的行能被 parser 原样读回（round-trip）
- official_eval 的 T2 MAE 与 T3 accuracy
- phasepicker.inference.picker 能 import（dedup 符号补齐后不再失败）

两种运行方式：
    pytest tests/test_official_formats.py
    python  tests/test_official_formats.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import (
    EventClass,
    ExamTask,
    Task1Result,
    Task2Result,
    Task3Result,
)
from phasepicker.io.official_answers import (
    normalize_file_id,
    parse_task1_answer_lines,
    parse_task2_answer_lines,
    parse_task3_answer_lines,
)
from phasepicker.io.submission_writer import (
    format_task1_line,
    format_task2_line,
    format_task3_line,
    write_task1_results,
    write_task2_submission,
    write_task3_submission,
)
from phasepicker.eval.official_eval import (
    evaluate_task1,
    evaluate_task2,
    evaluate_task3,
)
from phasepicker.io.official_exam import classify_task


# ----------------------- T1 解析：第1轮单 P/S -----------------------

def test_parse_t1_round1_single_ps():
    line = "exam2025/TASK01/T1.A.Q0001.mseed :P:   17.28 :S:   26.96"
    res = parse_task1_answer_lines([line])
    assert set(res) == {"T1.A.Q0001.mseed"}
    r = res["T1.A.Q0001.mseed"]
    assert r.p_times_s == [17.28]
    assert r.s_times_s == [26.96]


# ----------------------- T1 解析：第2轮多 P/S -----------------------

def test_parse_t1_round2_multi_ps():
    line = "./T1-Q/T1.A.Q0001.mseed : P : 35.04;135.76 : S : 41.78;142.42"
    res = parse_task1_answer_lines([line])
    r = res["T1.A.Q0001.mseed"]
    assert r.p_times_s == [35.04, 135.76]
    assert r.s_times_s == [41.78, 142.42]


def test_parse_t1_mixed_and_blank_lines():
    lines = [
        "",
        "exam2025/TASK01/T1.A.Q0001.mseed :P: 1.00 :S: 2.00",
        "   ",
        "garbage line without markers",
        "./T1-Q/T1.A.Q0002.mseed : P : 3.0;4.0 : S : 5.0",
    ]
    res = parse_task1_answer_lines(lines)
    assert set(res) == {"T1.A.Q0001.mseed", "T1.A.Q0002.mseed"}
    assert res["T1.A.Q0002.mseed"].p_times_s == [3.0, 4.0]
    assert res["T1.A.Q0002.mseed"].s_times_s == [5.0]


# ----------------------- T2 解析 -----------------------

def test_parse_t2():
    lines = [
        "exam2025/TASK02/T2.A.Q0001.mseed       4.3",
        "exam2025/TASK02/T2.A.Q0002.mseed       2.7",
    ]
    res = parse_task2_answer_lines(lines)
    assert abs(res["T2.A.Q0001.mseed"].magnitude - 4.3) < 1e-9
    assert abs(res["T2.A.Q0002.mseed"].magnitude - 2.7) < 1e-9


# ----------------------- T3 解析 + EventClass -----------------------

def test_parse_t3_and_eventclass():
    lines = [
        "exam2025/TASK03/T3.A.Q0001.mseed\t1",
        "exam2025/TASK03/T3.A.Q0002.mseed\t5",
        "exam2025/TASK03/T3.A.Q0003.mseed\t9",   # 非法类别，应被跳过
    ]
    res = parse_task3_answer_lines(lines)
    assert set(res) == {"T3.A.Q0001.mseed", "T3.A.Q0002.mseed"}
    assert res["T3.A.Q0001.mseed"].label == int(EventClass.EARTHQUAKE) == 1
    assert res["T3.A.Q0002.mseed"].label == int(EventClass.OTHER) == 5


def test_eventclass_values():
    assert int(EventClass.EARTHQUAKE) == 1
    assert int(EventClass.EXPLOSION) == 2
    assert int(EventClass.COLLAPSE) == 3
    assert int(EventClass.LANDSLIDE) == 4
    assert int(EventClass.OTHER) == 5


# ----------------------- normalize_file_id -----------------------

def test_normalize_file_id():
    assert normalize_file_id("exam2025/TASK01/T1.A.Q0001.mseed") == "T1.A.Q0001.mseed"
    assert normalize_file_id(".\\T1-Q\\T1.A.Q0002.mseed") == "T1.A.Q0002.mseed"
    assert normalize_file_id("  T1.A.Q0003.mseed  ") == "T1.A.Q0003.mseed"


# ----------------------- round-trip：写出 → 再解析读回 -----------------------

def test_t1_roundtrip_multi():
    orig = Task1Result(file_id="T1.A.Q0001.mseed", p_times_s=[35.04, 135.76], s_times_s=[41.78, 142.42])
    line = format_task1_line(orig, prefix="./T1-Q/")
    back = parse_task1_answer_lines([line])["T1.A.Q0001.mseed"]
    assert back.p_times_s == [35.04, 135.76]
    assert back.s_times_s == [41.78, 142.42]


def test_t1_roundtrip_single():
    orig = Task1Result(file_id="T1.A.Q0009.mseed", p_times_s=[17.28], s_times_s=[26.96])
    line = format_task1_line(orig, prefix="exam2025/TASK01/")
    back = parse_task1_answer_lines([line])["T1.A.Q0009.mseed"]
    assert back.p_times_s == [17.28]
    assert back.s_times_s == [26.96]


def test_t2_roundtrip():
    orig = Task2Result(file_id="T2.A.Q0001.mseed", magnitude=4.3)
    line = format_task2_line(orig, prefix="exam2025/TASK02/")
    back = parse_task2_answer_lines([line])["T2.A.Q0001.mseed"]
    assert abs(back.magnitude - 4.3) < 1e-9


def test_t3_roundtrip():
    orig = Task3Result(file_id="T3.A.Q0001.mseed", label=int(EventClass.EXPLOSION))
    line = format_task3_line(orig, prefix="exam2025/TASK03/")
    back = parse_task3_answer_lines([line])["T3.A.Q0001.mseed"]
    assert back.label == 2


def test_writers_to_file_and_read_back():
    t1 = [Task1Result("T1.A.Q0001.mseed", [1.0], [2.0]),
          Task1Result("T1.A.Q0002.mseed", [3.0, 4.0], [5.0])]
    t2 = [Task2Result("T2.A.Q0001.mseed", 4.3)]
    t3 = [Task3Result("T3.A.Q0001.mseed", 1)]
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "t1.txt")
        p2 = os.path.join(d, "t2.txt")
        p3 = os.path.join(d, "t3.txt")
        write_task1_results(t1, p1, prefix="./T1-Q/")
        write_task2_submission(t2, p2, prefix="exam2025/TASK02/")
        write_task3_submission(t3, p3, prefix="exam2025/TASK03/")
        with open(p1, encoding="utf-8") as f:
            r1 = parse_task1_answer_lines(f.read().splitlines())
        with open(p2, encoding="utf-8") as f:
            r2 = parse_task2_answer_lines(f.read().splitlines())
        with open(p3, encoding="utf-8") as f:
            r3 = parse_task3_answer_lines(f.read().splitlines())
    assert r1["T1.A.Q0002.mseed"].p_times_s == [3.0, 4.0]
    assert abs(r2["T2.A.Q0001.mseed"].magnitude - 4.3) < 1e-9
    assert r3["T3.A.Q0001.mseed"].label == 1


# ----------------------- official_eval：T1 -----------------------

def test_eval_task1_perfect():
    ans = {"T1.A.Q0001.mseed": Task1Result("T1.A.Q0001.mseed", [10.0], [15.0])}
    pred = {"T1.A.Q0001.mseed": Task1Result("T1.A.Q0001.mseed", [10.0], [15.0])}
    rep = evaluate_task1(pred, ans)
    assert rep.n_files == 1
    assert abs(rep.total_score - 2.0) < 1e-9
    assert rep.missing == [] and rep.extra == []


# ----------------------- official_eval：T2 MAE -----------------------

def test_eval_task2_mae():
    ans = {
        "a.mseed": Task2Result("a.mseed", 4.0),
        "b.mseed": Task2Result("b.mseed", 3.0),
        "c.mseed": Task2Result("c.mseed", 5.0),  # 答案有，预测无 → missing
    }
    pred = {
        "a.mseed": Task2Result("a.mseed", 4.5),   # 误差 0.5
        "b.mseed": Task2Result("b.mseed", 2.5),   # 误差 0.5
        "d.mseed": Task2Result("d.mseed", 1.0),   # 预测有，答案无 → extra
    }
    rep = evaluate_task2(pred, ans)
    assert rep.count == 2
    assert abs(rep.mae - 0.5) < 1e-9
    assert rep.missing == ["c.mseed"]
    assert rep.extra == ["d.mseed"]


# ----------------------- official_eval：T3 accuracy -----------------------

def test_eval_task3_accuracy():
    ans = {
        "a.mseed": Task3Result("a.mseed", 1),
        "b.mseed": Task3Result("b.mseed", 2),
        "c.mseed": Task3Result("c.mseed", 3),
        "d.mseed": Task3Result("d.mseed", 4),
    }
    pred = {
        "a.mseed": Task3Result("a.mseed", 1),   # 对
        "b.mseed": Task3Result("b.mseed", 2),   # 对
        "c.mseed": Task3Result("c.mseed", 5),   # 错
        "d.mseed": Task3Result("d.mseed", 4),   # 对
    }
    rep = evaluate_task3(pred, ans)
    assert rep.count == 4
    assert rep.correct == 3
    assert abs(rep.accuracy - 0.75) < 1e-9
    assert rep.confusion[(3, 5)] == 1
    assert rep.confusion[(1, 1)] == 1


# ----------------------- 任务归类 -----------------------

def test_classify_task():
    assert classify_task("exam2025/TASK01/T1.A.Q0001.mseed") == ExamTask.T1
    assert classify_task("./T2-Q/T2.A.Q0001.mseed") == ExamTask.T2
    assert classify_task("somewhere/TASK03/foo.mseed") == ExamTask.T3
    # 无 TASKxx/Tx-Q 目录标记时，退化为看 basename 前缀
    assert classify_task("random/dir/T1.A.Q0001.mseed") == ExamTask.T1
    assert classify_task("random/dir/whatever.mseed") is None


# ----------------------- 推理入口 import（dedup 符号补齐验证） -----------------------

def test_picker_imports_after_dedup_fix():
    # 关键回归：picker.py 依赖 dedup_picks / DedupConfig。若缺失会在 import 期崩。
    import importlib
    mod = importlib.import_module("phasepicker.inference.picker")
    assert hasattr(mod, "SeisBenchPicker")
    from phasepicker.postprocess.dedup import DedupConfig, dedup_picks
    from phasepicker.types import Pick, PhaseType
    # dedup_picks 无 cfg 时行为等价 deduplicate
    picks = [
        Pick(phase=PhaseType.P, time_utc=100.00, confidence=0.6, station="NET.STA"),
        Pick(phase=PhaseType.P, time_utc=100.02, confidence=0.9, station="NET.STA"),
    ]
    out = dedup_picks(picks)
    assert len(out) == 1 and out[0].confidence == 0.9
    # 给定 cfg：把 P 窗口调到 0（严格小于 → 永不合并），两个应都保留
    out2 = dedup_picks(picks, DedupConfig(p_merge_window_s=0.0))
    assert len(out2) == 2


# ----------------------------- standalone runner -----------------------------

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    lines = []
    for fn in fns:
        fn()
        lines.append(f"PASS {fn.__name__}")
    lines.append(f"SUMMARY {len(fns)}/{len(fns)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import base64
    out = _run_all()
    sys.stderr.write("B64:" + base64.b64encode(out.encode()).decode() + ":B64\n")
