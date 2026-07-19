#!/usr/bin/env bash
# =============================================================================
# GPU box (FunHPC / Cloud Studio) 开机一键脚本
#   适配: Tesla P4 + torch 2.0.1+cu118 + python 3.10 镜像
#   这台机器关机不保存 -> 每次开机跑一遍这条脚本即可就位
#
# 用法:  bash setup.sh
#
# 前提: 你已把权重备份 phasenet_stead_weights.tar.gz 传回 /data/coding/
#       (下载到本地的那个 1.1MB 文件; 没有它脚本会自动跨境重下, 慢但仍可用)
# =============================================================================
set -e

# 自动定位脚本所在目录 -> 仓库根 -> 权重备份，Gitee clone 到任何路径都能用
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CACHE=/data/seisbench_cache
# 权重优先找仓库里的 weights/，找不到再退回 /data/coding/
if [ -f "$REPO_ROOT/weights/phasenet_stead_weights.tar.gz" ]; then
  WEIGHT_BACKUP="$REPO_ROOT/weights/phasenet_stead_weights.tar.gz"
else
  WEIGHT_BACKUP=/data/coding/phasenet_stead_weights.tar.gz
fi

echo "==================== [1/5] 基本信息 ===================="
echo "目录: $(pwd)  |  python: $(which python)"
python --version

echo "==================== [2/5] 设置 seisbench 缓存目录 ===================="
export SEISBENCH_CACHE_ROOT=$CACHE
mkdir -p "$CACHE"
# 写进 bashrc, 新开终端也生效 (关机会丢, 所以每次都写)
grep -q SEISBENCH_CACHE_ROOT ~/.bashrc 2>/dev/null || \
  echo "export SEISBENCH_CACHE_ROOT=$CACHE" >> ~/.bashrc
echo "SEISBENCH_CACHE_ROOT=$CACHE"

echo "==================== [3/5] 安装 seisbench + obspy ===================="
# torch 已由镜像内置, 不要动; 只装缺的. 用清华镜像加速.
python -m pip install --root-user-action=ignore \
  -i https://pypi.tuna.tsinghua.edu.cn/simple seisbench obspy

echo "==================== [4/5] 恢复预训练权重 (免跨境重下) ===================="
if [ -f "$WEIGHT_BACKUP" ]; then
  echo "发现权重备份, 解压恢复中..."
  tar xzf "$WEIGHT_BACKUP" -C "$CACHE"
  echo "权重已恢复到 $CACHE"
else
  echo "!! 未找到 $WEIGHT_BACKUP"
  echo "!! 首次加载模型会跨境下载(约1MB, 慢但能成). 建议下次把备份传回来."
fi

echo "==================== [5/5] GPU 真实计算自检 + 模型加载 ===================="
python - << 'PYEOF'
import torch
x = torch.randn(1000, 1000).cuda()
print("OK GPU compute:", float((x @ x).sum()))
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
import seisbench.models as sbm
m = sbm.PhaseNet.from_pretrained("stead")
print("PhaseNet 