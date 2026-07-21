"""官方比赛任务层：T1 runner、T2 震级、T3 事件分类。"""

from .event_classifier import ConstantEventClassifier, EventClassifier, TrainedEventClassifier
from .magnitude_task import ConstantMagnitudePredictor, MagnitudePredictor, TrainedMagnitudePredictor

__all__ = [
    "ConstantEventClassifier",
    "ConstantMagnitudePredictor",
    "EventClassifier",
    "MagnitudePredictor",
    "TrainedEventClassifier",
    "TrainedMagnitudePredictor",
]
