"""T2 震级任务接口 + 占位 predictor（Magnitude task）.

===== 诚实边界（务必保留）=====
eew/magnitude.py 里的 estimate_magnitude 产出的是**未标定的相对量级**
（赛题波形无位置信息、无仪器响应标定）。**绝不能**把那个未标定值直接冒充
T2 的最终提交震级——那会在有标注留出集上系统性偏差、误导评估。

因此这里的占位 predictor 明确标注 ``is_placeholder=True``，并默认返回一个
中性常数（数据集震级中位数量级），只为打通 T1→T2→T3 的端到端流程与格式测试。
真正的 T2 模型（回归网络 / 标定后的 Pd 公式）实现后替换 predict 即可，
接口 MagnitudePredictor 不变。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..types import ExamSample, Task2Result


class MagnitudePredictor(ABC):
    """T2 震级预测器抽象契约：吃一个 ExamSample，吐一个 Task2Result。"""

    @abstractmethod
    def predict(self, sample: ExamSample) -> Task2Result:
        raise NotImplementedError


class ConstantMagnitudePredictor(MagnitudePredictor):
    """占位 predictor：对所有样本返回同一个常数震级。

    仅用于流程打通与格式测试，**不是**可提交的模型。默认常数取一个中性值，
    可通过构造参数调整。is_placeholder 恒为 True，提醒下游这不是标定结果。
    """

    is_placeholder = True

    def __init__(self, default_magnitude: float = 3.0):
        self._default = float(default_magnitude)

    def predict(self, sample: ExamSample) -> Task2Result:
        return Task2Result(file_id=sample.file_id, magnitude=self._default)


class TrainedMagnitudePredictor(MagnitudePredictor):
    """从已训练模型包预测震级；这是真实波形驱动模型，不是占位常数。"""

    is_placeholder = False

    def __init__(self, model_path: str, stream_loader: Optional[Callable] = None):
        from .baseline_models import load_bundle

        self._bundle = load_bundle(model_path, expected_task="T2")
        if stream_loader is None:
            from ..io.official_waveforms import read_mseed_stream

            stream_loader = read_mseed_stream
        self._stream_loader = stream_loader

    def predict(self, sample: ExamSample) -> Task2Result:
        from .waveform_features import extract_waveform_features

        stream = self._stream_loader(sample.source_path)
        features = extract_waveform_features(stream)
        magnitude = float(self._bundle.predict_one(features))
        # 去掉树模型极少数外推异常，并限制在官方历史合理范围外留少量余量。
        magnitude = min(9.9, max(0.0, magnitude))
        return Task2Result(file_id=sample.file_id, magnitude=magnitude)
