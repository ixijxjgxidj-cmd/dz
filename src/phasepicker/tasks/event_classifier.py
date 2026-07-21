"""T3 事件分类任务接口 + 占位 predictor（Event classification task）.

类别（EventClass）：1 地震 / 2 爆破 / 3 塌陷 / 4 滑坡 / 5 其它。

与 magnitude_task 同理，这里只给接口与占位实现，把端到端流程和格式测试跑通。
占位 predictor 默认全部判为"地震"（数据集中占绝大多数的类别），明确标注
``is_placeholder=True``。真正的 T3 分类模型实现后替换 predict 即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..types import EventClass, ExamSample, Task3Result


class EventClassifier(ABC):
    """T3 分类器抽象契约：吃一个 ExamSample，吐一个 Task3Result。"""

    @abstractmethod
    def predict(self, sample: ExamSample) -> Task3Result:
        raise NotImplementedError


class ConstantEventClassifier(EventClassifier):
    """占位 predictor：对所有样本返回同一个类别（默认"地震"）。

    仅用于流程打通与格式测试，**不是**可提交的模型。is_placeholder 恒为 True。
    """

    is_placeholder = True

    def __init__(self, default_label: int = int(EventClass.EARTHQUAKE)):
        self._default = int(default_label)

    def predict(self, sample: ExamSample) -> Task3Result:
        return Task3Result(file_id=sample.file_id, label=self._default, confidence=None)


class TrainedEventClassifier(EventClassifier):
    """从已训练模型包预测五类事件，并返回最大类别概率作为置信度。"""

    is_placeholder = False

    def __init__(self, model_path: str, stream_loader: Optional[Callable] = None):
        from .baseline_models import load_bundle

        self._bundle = load_bundle(model_path, expected_task="T3")
        if stream_loader is None:
            from ..io.official_waveforms import read_mseed_stream

            stream_loader = read_mseed_stream
        self._stream_loader = stream_loader

    def predict(self, sample: ExamSample) -> Task3Result:
        from .waveform_features import extract_waveform_features

        stream = self._stream_loader(sample.source_path)
        features = extract_waveform_features(stream)
        label = int(self._bundle.predict_one(features))
        if label not in {int(v) for v in EventClass}:
            raise ValueError(f"模型输出非法 T3 类别：{label}")
        proba = self._bundle.predict_proba_one(features)
        confidence = float(max(proba)) if proba is not None and len(proba) else None
        return Task3Result(file_id=sample.file_id, label=label, confidence=confidence)
