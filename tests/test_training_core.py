"""训练脚手架纯逻辑核心的单元测试（无 torch/obspy 也能跑）.

覆盖四块最容易出隐蔽 bug、又直接影响"可复现 + 不丢成果 + 不虚高"的逻辑：
  1) config   —— 保存/加载往返、未知字段容错、自检
  2) seed     —— 同种子可复现、不同种子有差异
  3) splits   —— 无泄漏、可复现、增量稳定、退化保护
  4) checkpoint —— best/last 指针、续训、剪枝、外送钩子、manifest 恢复
  5) label_adapter —— 相对/绝对时间归一化、CSV/JSON 解析、Pg/Pn->P

两种运行方式：
    pytest tests/test_training_core.py
    python  tests/test_training_core.py     # 无 pytest 时 standalone
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.training.config import TrainConfig
from phasepicker.training.seed import seed_everything
from phasepicker.training.splits import split_by_group, kfold_by_group
from phasepicker.training.checkpoint import CheckpointManager, LAST_NAME, BEST_NAME
from phasepicker.training.label_adapter import (
    normalize_time,
    parse_long_csv,
    parse_per_file,
    LabelPick,
)

import random


# ----------------------------- config -----------------------------

def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cfg = TrainConfig(experiment_name="exp_x", batch_size=128, learning_rate=5e-4)
        p = os.path.join(d, "cfg.json")
        cfg.save(p)
        back = TrainConfig.load(p)
        assert back.experiment_name == "exp_x"
        assert back.batch_size == 128
        assert abs(back.learning_rate - 5e-4) < 1e-12


def test_config_load_ignores_unknown_fields():
    # 旧 checkpoint 可能缺新字段 / 有已删字段，加载不应崩
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"experiment_name": "old", "REMOVED_FIELD": 123}, f)
        back = TrainConfig.load(p)
        assert back.experiment_name == "old"


def test_config_validate_catches_bad_values():
    bad = TrainConfig(batch_size=0, epochs=-1, val_fraction=1.5, learning_rate=0)
    problems = bad.validate()
    assert len(problems) >= 4
    good = TrainConfig()
    assert good.validate() == []


# ----------------------------- seed -----------------------------

def test_seed_reproducible():
    seed_everything(123, strict=False)
    a = [random.random() for _ in range(5)]
    seed_everything(123, strict=False)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_different_seeds_differ():
    seed_everything(1, strict=False)
    a = [random.random() for _ in range(5)]
    seed_everything(2, strict=False)
    b = [random.random() for _ in range(5)]
    assert a != b


# ----------------------------- splits -----------------------------

def test_split_no_leak_by_group():
    # 10 个事件，每个事件 3 个窗口 → 同事件必须整体归一侧
    group_keys = []
    for ev in range(10):
        group_keys += [f"ev{ev}"] * 3
    res = split_by_group(group_keys, val_fraction=0.3, seed=42)
    res.assert_no_leak()  # 不抛异常即通过
    # 同一 event 的三个下标必须全在同侧
    for ev in range(10):
        idxs = [i for i, k in enumerate(group_keys) if k == f"ev{ev}"]
        in_val = [i in res.val_idx for i in idxs]
        assert all(in_val) or not any(in_val), f"事件 ev{ev} 被拆散，发生泄漏"


def test_split_reproducible():
    keys = [f"ev{i%20}" for i in range(200)]
    r1 = split_by_group(keys, val_fraction=0.25, seed=7)
    r2 = split_by_group(keys, val_fraction=0.25, seed=7)
    assert r1.val_groups == r2.val_groups
    assert r1.train_idx == r2.train_idx


def test_split_incremental_stability():
    # 增量加数据：老组的归属不应改变（哈希切分的关键优势）
    keys_small = [f"ev{i}" for i in range(20)]
    keys_big = keys_small + [f"ev{i}" for i in range(20, 40)]
    r_small = split_by_group(keys_small, val_fraction=0.3, seed=9)
    r_big = split_by_group(keys_big, val_fraction=0.3, seed=9)
    # ev0..ev19 中，进验证集的那些，在扩充后仍应在验证集
    small_val = set(r_small.val_groups)
    big_val = set(r_big.val_groups)
    assert small_val.issubset(big_val), "增量后老组归属变了，验证集不稳定"


def test_split_degenerate_protection():
    # 只有 2 个组也不能出现某侧为空
    res = split_by_group(["a", "a", "b", "b"], val_fraction=0.1, seed=1)
    assert len(res.train_idx) > 0 and len(res.val_idx) > 0


def test_kfold_no_leak():
    keys = [f"ev{i%15}" for i in range(150)]
    folds = kfold_by_group(keys, n_folds=5, seed=3)
    assert len(folds) >= 2
    for fold in folds:
        fold.assert_no_leak()


# ----------------------------- checkpoint -----------------------------

def _fake_save(path, payload):
    # 假的 torch.save：把 payload 写成 JSON，便于验证续训能读回
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _fake_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_checkpoint_best_and_last_pointers():
    with tempfile.TemporaryDirectory() as d:
        cm = CheckpointManager(d, save_fn=_fake_save, higher_is_better=True)
        cm.save(step=100, epoch=0, payload={"model_state": "w1"}, val_score=0.5)
        cm.save(step=200, epoch=1, payload={"model_state": "w2"}, val_score=0.8)
        cm.save(step=300, epoch=2, payload={"model_state": "w3"}, val_score=0.6)
        # best 应指向 val_score 最高（0.8, w2），last 指向最新（w3）
        best = _fake_load(cm.best_path())
        last = _fake_load(cm.latest_path())
        assert best["model_state"] == "w2"
        assert last["model_state"] == "w3"
        assert abs(cm.best_score - 0.8) < 1e-9


def test_checkpoint_resume_across_sessions():
    # 模拟"关机后重开"：新建一个 manager 指向同目录，应能恢复簿记并续训
    with tempfile.TemporaryDirectory() as d:
        cm1 = CheckpointManager(d, save_fn=_fake_save)
        cm1.save(step=100, epoch=0, payload={"model_state": "w1", "step": 100}, val_score=0.7)
        # 新 session
        cm2 = CheckpointManager(d, save_fn=_fake_save)
        payload = cm2.load_latest(load_fn=_fake_load)
        assert payload is not None
        assert payload["step"] == 100
        assert abs(cm2.best_score - 0.7) < 1e-9  # manifest 恢复了 best_score


def test_checkpoint_prune_keeps_best():
    with tempfile.TemporaryDirectory() as d:
        cm = CheckpointManager(d, save_fn=_fake_save, keep_last_n=2)
        # 第 1 个给最高分，之后一路走低；剪枝时 best 必须留住
        cm.save(step=1, epoch=0, payload={"model_state": "best_w"}, val_score=0.99)
        for s in range(2, 8):
            cm.save(step=s, epoch=0, payload={"model_state": f"w{s}"}, val_score=0.1)
        best = _fake_load(cm.best_path())
        assert best["model_state"] == "best_w", "剪枝误删了最佳权重"
        # 历史按步档不应无限增长
        ckpts = [f for f in os.listdir(d) if f.startswith("step_")]
        assert len(ckpts) <= 3  # keep_last_n=2 + 可能的 best 档


def test_checkpoint_upload_hook_called_and_safe():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        cm = CheckpointManager(d, save_fn=_fake_save, upload_hook=lambda p: calls.append(p))
        cm.save(step=1, epoch=0, payload={"model_state": "w"}, val_score=0.5)
        assert len(calls) >= 1  # 至少外送了一次

    # 钩子抛异常也绝不能中断训练
    def boom(_):
        raise RuntimeError("网络炸了")
    with tempfile.TemporaryDirectory() as d:
        cm = CheckpointManager(d, save_fn=_fake_save, upload_hook=boom)
        cm.save(step=1, epoch=0, payload={"model_state": "w"}, val_score=0.5)  # 不应抛


# ----------------------------- label_adapter -----------------------------

def test_normalize_time_unix():
    assert normalize_time("1600000000.5", "unix") == 1600000000.5


def test_normalize_time_relative():
    # 相对 3.5s + 起点 1000 → 1003.5（这正是防止到时系统性偏移的锚定步骤）
    assert normalize_time(3.5, "relative", starttime_utc=1000.0) == 1003.5


def test_normalize_time_relative_requires_starttime():
    try:
        normalize_time(3.5, "relative")
        assert False, "缺 starttime 应报错"
    except ValueError:
        pass


def test_parse_long_csv_with_phase_variants():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "labels.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("file_id,station,phase,time\n")
            f.write("eq1,NET.A,Pg,100.0\n")   # Pg 应归一成 P
            f.write("eq1,NET.A,Sn,105.0\n")   # Sn 应归一成 S
            f.write("eq2,NET.B,P,200.0\n")
        result = parse_long_csv(p, time_mode="unix")
        assert set(result.keys()) == {"eq1", "eq2"}
        phases_eq1 = sorted(pk.phase for pk in result["eq1"].picks)
        assert phases_eq1 == ["P", "S"]


def test_parse_long_csv_custom_columns():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "labels.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("waveform,sta,ph,arrival\n")
            f.write("eqX,S1,P,50.0\n")
        result = parse_long_csv(
            p, time_mode="unix",
            columns={"file_id": "waveform", "station": "sta", "phase": "ph", "time": "arrival"},
        )
        assert "eqX" in result
        assert result["eqX"].picks[0].time_utc == 50.0


def test_parse_per_file_json():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "eq1.json"), "w", encoding="utf-8") as f:
            json.dump({"picks": [
                {"phase": "P", "time": 10.0, "station": "A"},
                {"phase": "S", "time": 15.0, "station": "A"},
            ]}, f)
        result = parse_per_file(d, time_mode="unix")
        assert "eq1" in result
        assert len(result["eq1"].picks) == 2


def test_labelpick_rejects_bad_phase():
    try:
        LabelPick(phase="X", time_utc=1.0)
        assert False, "非法震相应报错"
    except ValueError:
        pass


# ----------------------------- runner -----------------------------

def _run_all():
    import io, contextlib
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    results = []
    for fn in fns:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            results.append(f"PASS {fn.__name__}")
        except Exception as exc:
            results.append(f"FAIL {fn.__name__}: {exc!r}")
    passed = sum(1 for r in results if r.startswith("PASS"))
    report = "\n".join(results) + f"\nSUMMARY {passed}/{len(fns)}"
    # 通过 stderr 输出，规避挂载盘吞 stdout 的问题
    sys.stderr.write(report + "\n")
    return passed == len(fns)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
