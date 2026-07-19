#!/usr/bin/env bash
# =============================================================================
# Cloud Studio 免费 GPU 一键训练脚本（针对其"关页面自动关机 + 硬盘15天清理"特性设计）
# =============================================================================
#
# 为什么需要这个脚本（写给非 AI 背景的队友）：
# Cloud Studio 免费空间有两个"会吃掉成果"的机制：
#   1) 关闭编辑器浏览器标签页，约 10 分钟后自动关机；
#   2) 赠送的 80GB 云硬盘只免费保留 15 天，超期清理。
# 所以训练必须做到：①后台跑，不怕关页面；②checkpoint 实时推到阿里云 OSS，
# 不把权重留在会被清理的本地盘。这个脚本就是把这套纪律固化下来。
#
# 使用步骤：
#   1) 首次进入空间，先跑一次环境准备：  bash scripts/cloudstudio_launch.sh setup
#   2) 数据准备好后，启动后台训练：      bash scripts/cloudstudio_launch.sh train
#   3) 随时查看进度：                    bash scripts/cloudstudio_launch.sh status
#   4) 机器被关机后重开，续训：          bash scripts/cloudstudio_launch.sh resume
#
# 关键环境变量（在空间里 export 或写进 ~/.bashrc）：
#   OSS_URI       —— 权重同步目标，如 oss://your-bucket/runs/   （强烈建议必填）
#   DATA_ROOT     —— 训练数据目录（可先用 OSS 拉到本地，见 setup）
#   DATA_OSS_URI  —— 训练数据在 OSS 的地址，setup 时自动拉到 DATA_ROOT
#   EXP           —— 实验名，默认 phasenet_ft_v1
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

EXP="${EXP:-phasenet_ft_v1}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/runs}"
LOG_FILE="$OUTPUT_ROOT/$EXP/train.log"
PID_FILE="$OUTPUT_ROOT/$EXP/train.pid"

cmd_setup() {
  echo "[setup] 安装依赖（锁版本，保证可复现）..."
  pip install -r requirements.txt

  # ossutil：阿里云 OSS 命令行工具，用于跨云同步权重/数据
  if ! command -v ossutil >/dev/null 2>&1; then
    echo "[setup] 未检测到 ossutil。请先安装并 ossutil config 配置好阿里云 AK/SK。"
    echo "        参考：https://help.aliyun.com/zh/oss/developer-reference/ossutil"
  fi

  # 若指定了数据 OSS 地址，开机即拉到本地（避免每次手动上传，省 session 时间）
  if [[ -n "${DATA_OSS_URI:-}" ]]; then
    echo "[setup] 从 OSS 拉取训练数据：$DATA_OSS_URI -> $DATA_ROOT"
    mkdir -p "$DATA_ROOT"
    ossutil cp -r "$DATA_OSS_URI" "$DATA_ROOT" --update
  fi

  echo "[setup] 完成。GPU 信息："
  python -c "import torch; print('CUDA可用:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || true
}

# 内部：真正拉起训练进程（后台、nohup、日志重定向）
_launch() {
  local resume_flag="$1"
  mkdir -p "$(dirname "$LOG_FILE")"

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[train] 已有训练在跑（PID=$(cat "$PID_FILE")）。先 stop 再启动，或用 status 查看。"
    exit 1
  fi

  echo "[train] 后台启动训练：exp=$EXP data=$DATA_ROOT oss=${OSS_URI:-<未设置>}"
  # nohup + & ：即使浏览器标签关闭、SSH 断开，训练进程仍继续；关键就在这里
  nohup python scripts/train.py \
      --data-root "$DATA_ROOT" \
      --exp "$EXP" \
      --output-root "$OUTPUT_ROOT" \
      --oss-uri "${OSS_URI:-}" \
      $resume_flag \
      > "$LOG_FILE" 2>&1 &

  echo $! > "$PID_FILE"
  echo "[train] 已在后台运行，PID=$(cat "$PID_FILE")"
  echo "[train] 查看进度： bash scripts/cloudstudio_launch.sh status"
  echo "[train] 重要提醒：checkpoint 会按 sync_every_n_steps 自动推到 OSS，"
  echo "         机器即使被自动关机，最近一次同步后的权重也已安全存在 OSS。"
}

cmd_train()  { _launch ""; }
cmd_resume() { _launch "--resume"; }

cmd_status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[status] 训练进行中，PID=$(cat "$PID_FILE")"
  else
    echo "[status] 无运行中的训练进程（可能已完成或被关机）。"
  fi
  echo "[status] 最近日志（末 20 行）："
  [[ -f "$LOG_FILE" ]] && tail -n 20 "$LOG_FILE" || echo "  （暂无日志）"
}

cmd_stop() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" && echo "[stop] 已停止 PID=$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
  else
    echo "[stop] 没有运行中的训练进程。"
  fi
}

case "${1:-}" in
  setup)  cmd_setup ;;
  train)  cmd_train ;;
  resume) cmd_resume ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  *)
    echo "用法: bash scripts/cloudstudio_launch.sh {setup|train|resume|status|stop}"
    echo "  setup  首次准备环境+拉数据；train 后台开训；resume 断点续训；"
    echo "  status 看进度；stop 停止训练。"
    exit 1
    ;;
esac
