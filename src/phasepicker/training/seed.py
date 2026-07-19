"""可复现性（Reproducibility）—— 固定所有随机源。

===== 为什么这个模块单独存在（写给非 AI 背景的队友）=====
深度学习训练里有很多随机性来源：权重初始化、数据打乱顺序、数据增强、
dropout、GPU 上的非确定性算子……如果不固定，同样的代码、同样的数据，
每次训练出来的模型都不一样，分数也会飘。

这会带来两个致命问题：
  1) 无法复现：你今天调出一个高分权重，明天重训却复现不出来，
     决赛技术文档里的实验对比表就失去可信度。
  2) 无法归因：换了个超参分数变了，到底是超参起作用，还是纯粹随机波动？
     不固定种子就永远说不清。

所以每次训练开始前，必须调用 seed_everything(seed)。同一个 seed + 同一份
数据 + 同一套代码，应当得到（近乎）一致的结果。

===== 关于"完全确定" vs "性能" =====
强制 CUDA 完全确定（deterministic）会关掉一些快速但非确定的算子，训练会变慢。
本模块用 strict 参数暴露这个取舍：
  - strict=True ：完全可复现，训练略慢。做正式实验对比时用。
  - strict=False：允许 cuDNN 自动寻找最快算法（benchmark），只固定种子，
                  结果几乎一致但不保证逐位相同。快速试验时用。
"""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: int = 42, strict: bool = True) -> int:
    """固定 Python / NumPy / PyTorch（CPU+CUDA）的所有随机源。

    Args:
        seed: 随机种子。
        strict: True 时开启 PyTorch 确定性算法与固定 cuDNN，完全可复现但略慢；
                False 时只播种并开启 cuDNN benchmark，追求速度。

    Returns:
        实际使用的 seed（便于写进日志/checkpoint 元数据）。

    Note:
        torch 未安装时（如纯数据环境）会自动跳过 torch 部分，不报错，
        以保证本模块在无 GPU 依赖的环境也能被 import 和单元测试。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if strict:
            # 完全确定：关闭 benchmark、开启确定性算法
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # 某些算子需要设这个环境变量才能确定，且要在 import torch 前后都稳妥
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001 —— 旧版本 torch 无此 API，忽略
                pass
        else:
            # 追求速度：让 cuDNN 自动挑最快算法
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except ImportError:
        pass

    return seed


def make_worker_init_fn(base_seed: int):
    """为 DataLoader 的多 worker 生成确定性的 worker_init_fn。

    多进程数据加载时，每个 worker 有独立的随机状态，若不显式播种，
    数据增强的随机性在不同 worker 间不可控，破坏可复现性。

    用法（在 trainer 里）：
        DataLoader(..., worker_init_fn=make_worker_init_fn(cfg.seed))
    """

    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed % (2**32))
        except ImportError:
            pass

    return _init
