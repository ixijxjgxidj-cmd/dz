"""地震早期预警展示层（EEW display layer）.

这一层不参与初赛/复赛官方自动评分（官方只评 P/S 到时），作用是把"P/S 震相
拾取"接续成完整的应急管理故事，用于决赛专家评审那 20%（技术性、创新性、可靠
性、应用前景）。

科学口径：做的是"震后秒级快速监测预警"，不是"震前预测"。地震发生后用最先到达
的 P 波，在破坏性更强的 S 波/面波到达前，快速估计发震时刻、震中位置/方向、震
级，为远处地区争取数秒~数十秒预警。

前提：地点/震级估计需要台站经纬度 + 多台站 P 到时。若官方数据"不含位置信息"，
本层用自备台站表或演示数据运行，并在材料中如实说明，绝不硬吹单条无位置波形能
算震中。零重依赖（仅 numpy），可被单元测试覆盖。
"""

from .locate import (
    Station,
    StationArrival,
    LocationResult,
    estimate_origin_and_location,
    estimate_back_azimuth,
    warning_time_at,
)
from .magnitude import estimate_magnitude, MagnitudeResult

__all__ = ["Station", "StationArrival", "LocationResult", "estimate_origin_and_location", "estimate_back_azimuth", "warning_time_at", "estimate_magnitude", "MagnitudeResult"]
