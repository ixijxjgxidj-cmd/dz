# 地震震相拾取系统 — 参赛工程仓库

面向"中国（广西）—东盟人工智能+应急管理科技创新大赛 / 防震减灾专项赛道 · 自动震相拾取"的工程代码。

这个仓库存在的首要目的：**GPU 免费主机关机即清空，用 Gitee 仓库做持久化。每次开机 `git clone` 一条命令拉回全部环境。**

---

## 一、每次开机三步复现（最重要）

免费 GPU 机器（FunHPC / Cloud Studio）关机会清空一切。开新机器后：

```bash
# 1) 拉回整个仓库（含代码 + 权重备份，总共几 MB，秒级）
cd /data/coding
git clone https://gitee.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2) 一键装环境 + 恢复权重 + GPU 自检（几分钟）
bash scripts/setup.sh

# 3) 开不断线会话，跑闭环验证
tmux new -s seis
python scripts/closed_loop.py
```

看到 `CLOSED LOOP OK` 就说明环境、GPU、模型、评分全部就位。

> **注意镜像选择**：进机器时务必选 `PyTorch 2.0.1 / Python 3.10 / CUDA 11.8` 的镜像。
> 这是 Tesla P4（老架构 sm_61）唯一稳定可用的组合，新版 torch 会报 `no kernel image`。

---

## 二、仓库里有什么

| 路径 | 作用 |
|---|---|
| `scripts/setup.sh` | 开机一键：设缓存目录、装 seisbench+obspy、从 `weights/` 恢复权重、GPU 自检 |
| `scripts/baseline_synth.py` | 合成三分量波形 → PhaseNet 推理，验证推理链路 |
| `scripts/closed_loop.py` | **读取→推理→评分**完整闭环（自包含，内嵌已测评分逻辑） |
| `scripts/run_local_scoring.py` | 本地评分 CLI，官方评分规则复刻 |
| `scripts/train.py` | 微调训练入口（官方数据到位后用） |
| `weights/phasenet_stead_weights.tar.gz` | PhaseNet 预训练权重备份（1.1MB，免跨境重下） |
| `src/phasepicker/` | 工程内核：mseed 读取、预处理、推理封装、去重、评分、训练脚手架、EEW 展示层 |
| `tests/` | 纯逻辑核心单元测试（评分/时间对齐/去重/训练/EEW，共 58 项） |
| `MASTER_PLAN.md` | 融合版方案总纲（策略 + 工程 + 决赛叙事） |
| `ROADMAP.md` | 优先级路线图，三个"胜负手" |

---

## 三、关键约束与设计原则

- **网络**：国内免费机跨境下载数据集极慢且关机不留。因此不下大数据集，
  用合成数据验证链路；预训练权重（仅 1.1MB）备份进仓库。
- **持久化**：GPU 机器上一切关机即毁。唯一可信副本 = 本地 + Gitee 仓库。
  训练产物（checkpoint）需实时外送（见 `src/phasepicker/training/checkpoint.py`）。
- **时间对齐**：模型输出采样点下标，换算绝对到时的逻辑集中在
  `src/phasepicker/utils/timing.py`，有专门测试守护（最易"整盘皆输"处）。
- **评分驱动**：调参唯一客观依据是本地评分脚本（复刻官方规则），
  偏向高精确率（数量误差每超 1 个扣 0.5 分，宁漏勿多报）。

---

## 四、官方数据到位后第一件事

**先摸清数据格式，别急着训练**。确认三点后告诉维护者：
1. 波形格式（mseed？）、采样率是否 100Hz；
2. 真值标注：P/S 到时是"相对采样点"还是"绝对时间"，字段名；
3. 一个波形样例 + 一条标注样例。

然后把 `closed_loop.py` 里的合成数据加载换成真实加载器，**评分部分一行不动**，
即可看到 PhaseNet 在真实广西数据上的**真实基线分**——一切提分（阈值、后处理、微调）从这里开始。
