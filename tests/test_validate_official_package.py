"""validate_official_package 的纯逻辑 / smoke 测试——纯标准库.

不碰真实官方 zip（那两个大包不在仓库里），而是在临时目录里合成
"迷你官方包"：单层 zip（仿第1轮）与嵌套 zip（仿第2轮），覆盖：
- import 正常（脚本能被当模块加载）
- 只读扫描 + 嵌套 zip 下钻
- 答案文件识别（.an 与 .an.txt 两种命名）
- T1/T2/T3 归类与 detail 统计
- input↔answer 的 missing/extra 对齐
- 第2轮 T3 尾随类别词（``1 earquake``）能被解析（回归：原 rsplit 会误吞标签）

两种运行方式：
    pytest tests/test_validate_official_package.py
    python  tests/test_validate_official_package.py
"""

import io
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from phasepicker.types import ExamTask  # noqa: E402
import validate_official_package as V  # noqa: E402


# ----------------------- 合成迷你官方包 -----------------------


def _make_round1_zip(path: str) -> None:
    """单层 zip：答案 *.an.txt 与 mseed（TASKxx/）同层，仿第1轮。"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("exam/T1.an.txt",
                    "exam/TASK01/T1.A.Q0001.mseed :P:   1.0 :S:   2.0\n"
                    "exam/TASK01/T1.A.Q0002.mseed :P:   3.0 :S:   4.0\n")
        zf.writestr("exam/T2.an.txt",
                    "exam/TASK02/T2.A.Q0001.mseed  4.3\n"
                    "exam/TASK02/T2.A.Q0002.mseed  5.1\n")
        zf.writestr("exam/T3.an.txt",
                    "exam/TASK03/T3.A.Q0001.mseed\t1\n"
                    "exam/TASK03/T3.A.Q0002.mseed\t2\n")
        for i in (1, 2):
            zf.writestr(f"exam/TASK01/T1.A.Q000{i}.mseed", b"\x00")
            zf.writestr(f"exam/TASK02/T2.A.Q000{i}.mseed", b"\x00")
            zf.writestr(f"exam/TASK03/T3.A.Q000{i}.mseed", b"\x00")


def _make_round2_zip(path: str) -> None:
    """嵌套 zip：外层含 answer.zip + data.zip，仿第2轮。

    故意制造一处 missing（答案有 Q0003、数据只有 Q0001/Q0002）验证对齐，
    并让 T1 出现多 P/S、T3 带尾随类别词。
    """
    answer_buf = io.BytesIO()
    with zipfile.ZipFile(answer_buf, "w") as az:
        az.writestr("answer/T1.an",
                    "./T1-Q/T1.A.Q0001.mseed : P : 35.04;135.76 : S : 41.78;142.42\n"
                    "./T1-Q/T1.A.Q0002.mseed : P : 26.53 : S : 37.97\n")
        az.writestr("answer/T2.an",
                    "./T2-Q/T2.Q0001.mseed 5\n"
                    "./T2-Q/T2.Q0002.mseed 3.5\n")
        az.writestr("answer/T3.an",
                    "./T3-Q/T3.A.Q0001.mseed  1 earquake\n"
                    "./T3-Q/T3.A.Q0002.mseed  2 explosion\n"
                    "./T3-Q/T3.A.Q0003.mseed  1 earquake\n")  # Q0003 无对应 mseed → missing

    data_buf = io.BytesIO()
    with zipfile.ZipFile(data_buf, "w") as dz:
        for i in (1, 2):
            dz.writestr(f"exam-data/T1-Q/T1.A.Q000{i}.mseed", b"\x00")
            dz.writestr(f"exam-data/T2-Q/T2.Q000{i}.mseed", b"\x00")
            dz.writestr(f"exam-data/T3-Q/T3.A.Q000{i}.mseed", b"\x00")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("outer/1_answer.zip", answer_buf.getvalue())
        zf.writestr("outer/exam-data.zip", data_buf.getvalue())


# ----------------------- 测试 -----------------------


def test_round1_layout_and_stats():
    with tempfile.TemporaryDirectory() as d:
        zp = os.path.join(d, "r1.zip")
        _make_round1_zip(zp)
        rep = V.validate_package(zp)

    assert rep.round_kind == "round1"
    assert rep.n_mseed == 6
    t1 = rep.stats[ExamTask.T1]
    assert t1.n_input == 2 and t1.n_answer == 2 and t1.matched == 2
    assert not t1.missing and not t1.extra
    assert t1.detail["p_per_file_max"] == 1
    t2 = rep.stats[ExamTask.T2]
    assert abs(t2.detail["mag_min"] - 4.3) < 1e-9
    assert abs(t2.detail["mag_max"] - 5.1) < 1e-9
    t3 = rep.stats[ExamTask.T3]
    assert t3.detail["class_dist"] == {1: 1, 2: 1}


def test_round2_nested_and_missing():
    with tempfile.TemporaryDirectory() as d:
        zp = os.path.join(d, "r2.zip")
        _make_round2_zip(zp)
        rep = V.validate_package(zp)

    assert rep.round_kind == "round2"
    # 嵌套 zip 里 6 个 mseed 都应被下钻扫到
    assert rep.n_mseed == 6

    t1 = rep.stats[ExamTask.T1]
    assert t1.n_input == 2 and t1.n_answer == 2 and t1.matched == 2
    assert t1.detail["p_per_file_max"] == 2  # 多 P/S
    assert t1.detail["files_with_multi_ps"] == 1

    # T3 尾随类别词必须能解析出 3 条答案，且 Q0003 无 mseed → 1 个 missing
    t3 = rep.stats[ExamTask.T3]
    assert t3.n_answer == 3, "第2轮 T3 尾随类别词应被正确解析"
    assert t3.n_input == 2
    assert t3.matched == 2
    assert t3.missing == ["T3.A.Q0003.mseed"]
    assert t3.detail["class_dist"] == {1: 2, 2: 1}


def test_answers_only_skips_diff():
    with tempfile.TemporaryDirectory() as d:
        zp = os.path.join(d, "r2.zip")
        _make_round2_zip(zp)
        rep = V.validate_package(zp, answers_only=True)
    # answers_only 下不做对齐：missing/extra 保持空、matched=0
    t3 = rep.stats[ExamTask.T3]
    assert t3.n_answer == 3
    assert t3.matched == 0 and not t3.missing and not t3.extra


def test_main_smoke(capsys=None):
    """main() 端到端跑通并返回 0（import + 打印路径都正常）。"""
    with tempfile.TemporaryDirectory() as d:
        zp = os.path.join(d, "r1.zip")
        _make_round1_zip(zp)
        rc = V.main(["--zip", zp])
    assert rc == 0


if __name__ == "__main__":
    test_round1_layout_and_stats()
    test_round2_nested_and_missing()
    test_answers_only_skips_diff()
    test_main_smoke()
    print("ok: all validate_official_package tests passed")
