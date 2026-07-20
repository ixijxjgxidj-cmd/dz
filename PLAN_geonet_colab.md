# 计划：GeoNet(新西兰) 数据 → Colab 微调 PhaseNet

## 目标
在 Google Colab 上，用 GeoNet FDSN 直连拉取新西兰真实地震数据做 PhaseNet 微调，
断点续训、微调前后各评一次分（合成评分 + 真实留出集）。数据规模：尽量多，受 Colab 时限约束。

## 关键既有事实（基于 GitHub 最新版 4aa3bd8，已核实，不改）
- `finetune_phasenet.py` 的 `--data <hdf5>` 走 `load_hdf5_dataset`，只认这个格式：
  - group `"data"`，每条 dataset = 一条波形 `(3, N)`
  - attrs：`p_sample_100hz`、`s_sample_100hz`、`sampling_rate=100.0`
  - 到时缺失用 `-1`
- 这与 `diting_subset.py` / `diting_seisbench.py` 写盘格式一致。
  → **GeoNet 取数脚本只要产出同一格式，微调脚本一行不用改。**
- 上次修好的 BN 冻结 + best 回滚护栏仍在，训不崩的保障已就位。
- 最新版 checkpoint 加载已修为 `torch.load(..., weights_only=False)`（PyTorch 2.6+ 必需）。
- 现有 `eval_score` 用合成 case（证明"没训崩"），但证明不了真实数据泛化。
- **先例**：仓库已有 `weights/phasenet_ethz_ft1_best.pt`，说明已在 ETHZ（SeisBench 内置
  欧洲数据集，`sbd.ETHZ()` 一行下载）上微调过一次，但那次的取数没沉淀成脚本
  （曾用临时的 `/content/ethz_100hz_subset.hdf5`）。
  → **本次核心工程差异**：GeoNet 不是 SeisBench 内置数据集，必须走 obspy FDSN
  直连（catalog + waveform），比 ETHZ 的一行下载复杂，正是要固化成可复现脚本的地方。

## 改动一览（3 个文件）

### 1. 新增 `scripts/geonet_fetch.py`（数据取数，仿 diting_seisbench 的断点续传纪律）
用 `obspy.clients.fdsn.Client("GEONET")` 从新西兰 FDSN 拉：
- **catalog 阶段**：`get_events(starttime, endtime, minmagnitude, maxmagnitude, region)`
  拿事件列表；每个事件的 QuakeML 含 picks（phase_hint=P/S + station）。
- **波形阶段**：对每个 (事件, 台站)，聚合该台的 P/S pick，用 `get_waveforms` 取
  覆盖 P（且尽量含 S）的一段三分量波形。
- **重采样**：GeoNet 多为 100Hz/50Hz 混合 → 统一强制重采样到 100Hz，
  P/S 到时采样点下标同步换算（复用 diting 里的重采样逻辑）。
- **写盘**：group `"data"` + attrs（p_sample_100hz/s_sample_100hz/sampling_rate），
  gzip 压缩，增量 append。
- **断点续传**：progress.json 记已完成事件数；Ctrl+C 优雅退出；重跑跳过已完成。
- **边界参数**：`--start --end --minmag --maxmag --lat --lon --maxradius
  --max_traces --max_gb --win`（窗口默认 3001）。默认给一组"新西兰主震区、
  近两年、M≥3"的合理值，Colab 一段时间内能持续拉。
- **无鉴权**：GeoNet FDSN 开放，不需账号（脚本头部注明）。

### 2. 修改 `scripts/finetune_phasenet.py`（加真实留出集评分）
- 新增 `--holdout <hdf5>` 参数：指向一份 GeoNet 留出集（与训练集不重叠）。
- 新增 `eval_score_real(model, holdout_path, ...)`：用 `model.classify()` 对留出集
  逐条推理，复用已有的 `score_file` 打分规则（P≤0.1s 满分等），输出平均分/残差/满分率。
- 微调【前】【后】各跑一次：**合成评分 + 真实留出评分**，对比打印。
  → 合成分守"没训崩"，真实分看"有没有学到新西兰数据的东西"。
- best 护栏的选优指标：默认仍用合成分（保命），但**真实留出分一起打印**，
  必要时可加 `--select-by real` 让 best 以真实分选优（默认 synth，改动最小、最稳）。

### 3. 新增 `notebooks/colab_finetune_geonet.ipynb`（Colab 一键流程）
分 cell：
1. 挂载 Google Drive（`drive.mount`）——权重/数据/checkpoint 全落 Drive，断连不丢。
2. clone 仓库 + `pip install seisbench obspy h5py`（torch Colab 自带）。
3. 恢复 stead 预训练权重（优先用仓库里 `weights/phasenet_stead_weights.tar.gz`，
   免跨境重下）。
4. 跑 `geonet_fetch.py` → 数据写到 Drive（可分次跑，断点续传）。
   自动切出训练集 / 真实留出集（如 90/10）。
5. 跑 `finetune_phasenet.py --data <drive训练集> --holdout <drive留出集>
   --out <drive/runs> --resume`。
6. 打印微调前后对比（合成 + 真实），best.pt 已在 Drive。
- 每个重活 cell 顶部标注"断连后从这里重跑即可续"，对齐 Colab 免费版会话时限。

## Colab 时限对策（对应你选的"尽量多"）
- 数据与 checkpoint **全部指向 Drive 路径**，断连重连后 `--resume` 续训、
  `geonet_fetch` 续拉，不从头再来。
- `geonet_fetch` 支持 `--max_minutes`（软时限）：接近 Colab 单次上限前优雅落盘退出。

## 验证
- 本地无 torch/seisbench，只能 `py_compile` 语法校验 + 纯函数单测
  （重采样换算、窗口/到时下标换算、HDF5 读写往返）。
- 真实拉取 + GPU 微调在 Colab 上验证（我给出可直接跑的 notebook）。
- 不动已修好的 BN 冻结 / best 回滚逻辑。

## 不做（避免范围膨胀）
- 不接日本 NIED（需注册鉴权，等新西兰跑通再说）。
- 不重写 src/phasepicker 那套包管线（本次复用 scripts 版，改动最小、风险最低）。
- 不追求刷分，先把"真实数据微调不崩、真实留出分可测"这条链路打通。
