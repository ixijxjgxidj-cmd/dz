"""嵌套官方包读取、波形特征、T2/T3 可训练模型测试。"""

import io
import os
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_package_answers, read_source_bytes
from phasepicker.tasks.baseline_models import (
    load_bundle,
    save_bundle,
    train_event_baseline,
    train_magnitude_baseline,
)
from phasepicker.tasks.waveform_features import FEATURE_NAMES, extract_waveform_features
from phasepicker.types import ExamTask


def _sine(freq, amp=1.0, n=1000, sr=100.0):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_feature_shape_and_finite():
    stream = [
        ("BHZ", 100.0, _sine(2.0, 2.0)),
        ("BHN", 100.0, _sine(3.0, 1.0)),
        ("BHE", 100.0, _sine(5.0, 0.5)),
    ]
    feat = extract_waveform_features(stream)
    assert feat.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feat).all()


def test_feature_sees_amplitude_and_frequency():
    low = extract_waveform_features([("BHZ", 100.0, _sine(2.0, 1.0))])
    loud = extract_waveform_features([("BHZ", 100.0, _sine(2.0, 10.0))])
    high = extract_waveform_features([("BHZ", 100.0, _sine(12.0, 1.0))])
    assert loud[FEATURE_NAMES.index("Z_log_std")] > low[FEATURE_NAMES.index("Z_log_std")]
    assert high[FEATURE_NAMES.index("Z_dominant_freq_hz")] > low[FEATURE_NAMES.index("Z_dominant_freq_hz")]


def test_train_save_load_models():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y_mag = 4.0 + 0.3 * X[:, 0]
    y_cls = np.where(X[:, 1] > 0, 1, 2)
    mag = train_magnitude_baseline(X, y_mag, random_state=1)
    cls = train_event_baseline(X, y_cls, random_state=1)
    with tempfile.TemporaryDirectory() as d:
        p2 = os.path.join(d, "t2.joblib")
        p3 = os.path.join(d, "t3.joblib")
        save_bundle(mag, p2)
        save_bundle(cls, p3)
        mag2 = load_bundle(p2, "T2")
        cls2 = load_bundle(p3, "T3")
    assert isinstance(float(mag2.predict_one(X[0])), float)
    assert int(cls2.predict_one(X[0])) in {1, 2}
    assert mag2.is_placeholder is False and cls2.is_placeholder is False


def _make_nested_zip(path):
    answer = io.BytesIO()
    with zipfile.ZipFile(answer, "w") as az:
        az.writestr("answer/T2.an", "./T2-Q/T2.Q0001.mseed 5.2\n")
        az.writestr("answer/T3.an", "./T3-Q/T3.A.Q0001.mseed 2 explosion\n")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as dz:
        dz.writestr("exam/T2-Q/T2.Q0001.mseed", b"fake-t2")
        dz.writestr("exam/T3-Q/T3.A.Q0001.mseed", b"fake-t3")
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("answer.zip", answer.getvalue())
        outer.writestr("data.zip", data.getvalue())


def test_nested_scan_bytes_and_answers():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "round2.zip")
        _make_nested_zip(path)
        samples = scan_exam_input(path)
        assert len(samples) == 2
        by_task = {s.task: s for s in samples}
        assert read_source_bytes(by_task[ExamTask.T2].source_path) == b"fake-t2"
        t2 = read_package_answers(path, ExamTask.T2)
        t3 = read_package_answers(path, ExamTask.T3)
    assert abs(t2["T2.Q0001.mseed"].magnitude - 5.2) < 1e-9
    assert t3["T3.A.Q0001.mseed"].label == 2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    lines = []
    for fn in fns:
        fn()
        lines.append(f"PASS {fn.__name__}")
    lines.append(f"SUMMARY {len(fns)}/{len(fns)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_run_all())
