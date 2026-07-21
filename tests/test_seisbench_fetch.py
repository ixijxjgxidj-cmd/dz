"""seisbench_fetch 纯逻辑单元测试（到时列解析 / 波形规整 / 断点进度）。

这些函数不依赖 seisbench 本体，可直接跑：
    pytest tests/test_seisbench_fetch.py
    python  tests/test_seisbench_fetch.py

覆盖的核心不变量（对应 memory: seisbench-resample-rescales-picks）:
- resolve_picks 直接返回 metadata 里的 *_sample 值，【绝不】再手动 ×N，
  因为 SeisBench 构造时 sampling_rate=100 已把到时换算到目标采样率。
- 规范列缺失时，按 trace_P.*_arrival_sample / trace_S.*_arrival_sample
  回落到震相变体（ETHZ/GEOFON 的 Pg/Pn/Sg/Sn），取第一个有效值。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from seisbench_fetch import (  # noqa: E402
    _first_valid,
    _is_valid_sample,
    load_progress,
    resolve_picks,
    save_progress,
    to_3xn,
    _P_PRIMARY,
    _P_VARIANT,
    _S_PRIMARY,
    _S_VARIANT,
)


# ---------------------------------------------------------------------------
# _is_valid_sample
# ---------------------------------------------------------------------------

def test_is_valid_sample_accepts_nonneg_finite():
    assert _is_valid_sample(0)
    assert _is_valid_sample(123)
    assert _is_valid_sample(4567.0)
    assert _is_valid_sample(np.float32(10.0))


def test_is_valid_sample_rejects_none_nan_neg_empty():
    assert not _is_valid_sample(None)
    assert not _is_valid_sample(float("nan"))
    assert not _is_valid_sample(-1)
    assert not _is_valid_sample(-0.5)
    assert not _is_valid_sample([])
    assert not _is_valid_sample(np.array([]))


def test_is_valid_sample_reads_first_of_array():
    # SeisBench 有时把单值包成 array；取第一个元素判断
    assert _is_valid_sample(np.array([500.0]))
    assert not _is_valid_sample(np.array([-1.0]))


def test_is_valid_sample_rejects_unparseable():
    assert not _is_valid_sample("not-a-number")
    assert not _is_valid_sample(object())


# ---------------------------------------------------------------------------
# resolve_picks: 规范列优先，且不做任何缩放
# ---------------------------------------------------------------------------

def test_resolve_picks_primary_columns_no_rescale():
    # 关键回归：值原样返回，不得 ×2 / ×N
    meta = {_P_PRIMARY: 1500.0, _S_PRIMARY: 3000.0}
    p, s = resolve_picks(meta)
    assert p == 1500.0
    assert s == 3000.0


def test_resolve_picks_missing_returns_minus_one():
    p, s = resolve_picks({})
    assert p == -1.0
    assert s == -1.0


def test_resolve_picks_p_present_s_missing():
    p, s = resolve_picks({_P_PRIMARY: 800.0})
    assert p == 800.0
    assert s == -1.0


def test_resolve_picks_nan_primary_falls_through_to_minus_one():
    # 规范列是 NaN 且无变体 → -1（而不是把 NaN 传下去）
    p, s = resolve_picks({_P_PRIMARY: float("nan"), _S_PRIMARY: float("nan")})
    assert p == -1.0
    assert s == -1.0


# ---------------------------------------------------------------------------
# resolve_picks: 震相变体回落（ETHZ/GEOFON 的 Pg/Pn/Sg/Sn）
# ---------------------------------------------------------------------------

def test_resolve_picks_variant_fallback_pg_sg():
    meta = {"trace_Pg_arrival_sample": 1200.0, "trace_Sg_arrival_sample": 2400.0}
    p, s = resolve_picks(meta)
    assert p == 1200.0
    assert s == 2400.0


def test_resolve_picks_variant_pn_sn():
    meta = {"trace_Pn_arrival_sample": 999.0, "trace_Sn_arrival_sample": 1998.0}
    p, s = resolve_picks(meta)
    assert p == 999.0
    assert s == 1998.0


def test_resolve_picks_primary_wins_over_variant():
    # 规范列有效时优先用它，忽略变体
    meta = {
        _P_PRIMARY: 100.0,
        "trace_Pg_arrival_sample": 777.0,
        _S_PRIMARY: 200.0,
        "trace_Sn_arrival_sample": 888.0,
    }
    p, s = resolve_picks(meta)
    assert p == 100.0
    assert s == 200.0


def test_resolve_picks_variant_skips_invalid_first():
    # 第一个变体是 NaN，应跳到下一个有效变体
    meta = {
        "trace_Pn_arrival_sample": float("nan"),
        "trace_Pg_arrival_sample": 1350.0,
    }
    p, _ = resolve_picks(meta)
    assert p == 1350.0


def test_variant_regex_does_not_match_primary_lowercase():
    # 大小写敏感：小写 p 的规范列不该被大写变体正则误伤（由 _first_valid 逻辑保证）
    assert not _P_VARIANT.match(_P_PRIMARY)
    assert not _S_VARIANT.match(_S_PRIMARY)
    assert _P_VARIANT.match("trace_Pg_arrival_sample")
    assert _S_VARIANT.match("trace_Sn_arrival_sample")


def test_first_valid_direct():
    # 直接测底层：primary 命中
    assert _first_valid({_P_PRIMARY: 42.0}, _P_PRIMARY, _P_VARIANT) == 42.0
    # primary 缺失 → 变体命中
    assert _first_valid(
        {"trace_Pg_arrival_sample": 7.0}, _P_PRIMARY, _P_VARIANT
    ) == 7.0
    # 全无 → -1
    assert _first_valid({}, _P_PRIMARY, _P_VARIANT) == -1.0


# ---------------------------------------------------------------------------
# to_3xn: 波形规整
# ---------------------------------------------------------------------------

def test_to_3xn_passthrough_3byn():
    w = np.arange(3 * 100, dtype=np.float32).reshape(3, 100)
    out = to_3xn(w)
    assert out.shape == (3, 100)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, w)


def test_to_3xn_transposes_nby3():
    w = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    out = to_3xn(w)
    assert out.shape == (3, 100)


def test_to_3xn_trims_extra_channels():
    # 多于 3 分量时只取前 3
    w = np.arange(5 * 200, dtype=np.float32).reshape(5, 200)
    out = to_3xn(w)
    assert out.shape == (3, 200)


def test_to_3xn_rejects_fewer_than_3_channels():
    assert to_3xn(np.zeros((2, 100), dtype=np.float32)) is None


def test_to_3xn_rejects_1d():
    assert to_3xn(np.zeros(100, dtype=np.float32)) is None


def test_to_3xn_result_is_contiguous():
    w = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    out = to_3xn(w)
    assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# 断点进度：load / save 往返 + 原子替换
# ---------------------------------------------------------------------------

def test_load_progress_missing_returns_defaults(tmp_path):
    prog = load_progress(str(tmp_path / "nope.json"))
    assert prog == {"done": 0, "written": 0, "written_bytes": 0}


def test_load_progress_corrupt_returns_defaults(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ this is not json", encoding="utf-8")
    prog = load_progress(str(p))
    assert prog == {"done": 0, "written": 0, "written_bytes": 0}


def test_save_then_load_roundtrip(tmp_path):
    p = str(tmp_path / "progress.json")
    save_progress(p, done=1234, written=1000, written_bytes=5_000_000)
    prog = load_progress(p)
    assert prog == {"done": 1234, "written": 1000, "written_bytes": 5_000_000}


def test_save_progress_is_atomic_no_tmp_left(tmp_path):
    p = str(tmp_path / "progress.json")
    save_progress(p, done=5, written=5, written_bytes=10)
    # 原子替换后不应残留 .tmp
    assert not os.path.exists(p + ".tmp")
    with open(p, encoding="utf-8") as f:
        assert json.load(f)["done"] == 5


def test_save_progress_overwrites(tmp_path):
    p = str(tmp_path / "progress.json")
    save_progress(p, 1, 1, 1)
    save_progress(p, 9, 8, 7)
    assert load_progress(p) == {"done": 9, "written": 8, "written_bytes": 7}


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    import tempfile
    from pathlib import Path
    lines = []
    passed = 0
    for fn in fns:
        try:
            # 手动给需要 tmp_path 的用例注入一个临时目录
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
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
