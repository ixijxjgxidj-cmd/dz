"""T2/T3 可训练 CPU 基线模型与模型包持久化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .waveform_features import FEATURE_NAMES


MODEL_FORMAT_VERSION = 1


@dataclass
class BaselineModelBundle:
    """把估计器、特征契约、训练来源和评估指标放在一个可审计文件中。"""

    task: str
    estimator: Any
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    trained_on: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = MODEL_FORMAT_VERSION

    @property
    def is_placeholder(self) -> bool:
        return False

    def validate_features(self, features: Sequence[float]) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if self.feature_names != list(FEATURE_NAMES):
            raise ValueError("模型特征版本与当前代码不一致，请重新训练模型")
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"模型需要 {len(self.feature_names)} 个特征，实际得到 {x.shape[1]}")
        return x

    def predict_one(self, features: Sequence[float]):
        x = self.validate_features(features)
        return self.estimator.predict(x)[0]

    def predict_proba_one(self, features: Sequence[float]):
        x = self.validate_features(features)
        if not hasattr(self.estimator, "predict_proba"):
            return None
        return self.estimator.predict_proba(x)[0]


def train_magnitude_baseline(X, y, random_state: int = 42) -> BaselineModelBundle:
    """训练 ExtraTrees 震级回归基线。"""
    from sklearn.ensemble import ExtraTreesRegressor

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return BaselineModelBundle(task="T2", estimator=model)


def train_event_baseline(X, y, random_state: int = 42) -> BaselineModelBundle:
    """训练带类别均衡权重的 ExtraTrees 五分类基线。"""
    from sklearn.ensemble import ExtraTreesClassifier

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    model = ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return BaselineModelBundle(task="T3", estimator=model)


def save_bundle(bundle: BaselineModelBundle, path: str) -> None:
    import joblib

    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str, expected_task: str | None = None) -> BaselineModelBundle:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, BaselineModelBundle):
        raise TypeError("文件不是 phasepicker BaselineModelBundle")
    if bundle.format_version != MODEL_FORMAT_VERSION:
        raise ValueError(f"不支持的模型格式版本：{bundle.format_version}")
    if expected_task and bundle.task != expected_task:
        raise ValueError(f"需要 {expected_task} 模型，实际是 {bundle.task}")
    return bundle
