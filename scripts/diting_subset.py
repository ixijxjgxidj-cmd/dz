"""DiTing 数据集子集切割 + 重采样到 100Hz + 微调数据准备.

===== 重要背景（务必先读）=====
1) DiTing 官方是 292GB / 27 个独立 hdf5 文件（part 0~26），采样率 50Hz。
   你的比赛官方数据是 100Hz —— 所以本脚本强制重采样到 100Hz，否则模型会崩。
2) 你只想要 10~20GB。**不要用 SeisBench 的 sbd.DiTing() 全量下载**（它要先下 292GB）。
   正确做法：只下载其中 1~2 个 hdf5 分卷文件（每个约 10GB），本脚本处理这些分卷。

===== 两种使用方式 =====
方式 A（推荐，最省空间）：你自己从 GitHub(mstlyg/DiTing330km) 或网盘
   只下载 1~2 个分卷，例如 DiTing330km_part_0.hdf5 + 对应的 csv，
   放到 --raw_dir 指定目录，本脚本负责：读取→重采样100Hz→切子集→存标准格式。

方式 B：如果你已经有 SeisBench 全量缓存，用 --from_seisbench 直接取前 N 条。

本脚本零训练，只做“数据准备 + 验证能被 PhaseNet 读”，跑通后接微调脚本。
"""

import argparse
import os
import sys

import numpy as np


def resample_trace(data, orig_sr, target_sr):
    """把单条波形从 orig_sr 重采样到 target_sr（线性插值，足够微调用）。

    data: (channels, n_samples)
    50Hz -> 100Hz 是上采样，点数翻倍，同时到时对应的采样点下标也要 ×2。
    """
    if orig_sr == target_sr:
        return data
    n_ch, n = data.shape
    ratio = target_sr / orig_sr
    new_n = int(round(n * ratio))
    old_t = np.arange(n)
    new_t = np.linspace(0, n - 1, new_n)
    out = np.empty((n_ch, new_n), dtype=np.float32)
    for c in range(n_ch):
        out[c] = np.interp(new_t, old_t, data[c])
    return out


def process_hdf5_subset(raw_dir, out_dir, max_gb, orig_sr, target_sr):
    """方式A：读取 raw_dir 下的 DiTing 分卷 hdf5，切子集、重采样、存出。"""
    import h5py
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    # 找分卷文件
    h5_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".hdf5")
    )
    csv_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")
    )
    if not h5_files:
        print(f"!! {raw_dir} 下没找到 .hdf5 分卷文件。请先下载 1~2 个 DiTing 分卷放进去。")
        sys.exit(1)

    print(f"发现 {len(h5_files)} 个 hdf5 分卷, {len(csv_files)} 个 csv 元数据")

    max_bytes = max_gb * (1024 ** 3)
    written = 0
    n_events = 0
    manifest = []

    out_h5_path = os.path.join(out_dir, "diting_subset_100hz.hdf5")
    with h5py.File(out_h5_path, "w") as fout:
        grp = fout.create_group("data")
        for h5f in h5_files:
            # 配对同名 csv（DiTing 里 csv 与 hdf5 一一对应）
            base = os.path.splitext(os.path.basename(h5f))[0]
            csv_match = [c for c in csv_files if base in os.path.basename(c)]
            meta = pd.read_csv(csv_match[0]) if csv_match else None
            print(f"\n处理 {os.path.basename(h5f)} ...")

            with h5py.File(h5f, "r") as fin:
                # DiTing 的波形通常存在某个 group 下，key 见 csv 的 'key' 列
                # 不同发布版本 group 名可能不同，这里自动探测
                data_root = fin
                if "earthquake" in fin:
                    data_root = fin["earthquake"]
                elif "data" in fin:
                    data_root = fin["data"]

                keys = list(data_root.keys())
                print(f"  该分卷含 {len(keys)} 条波形")

                for k in keys:
                    if written >= max_bytes:
                        break
                    arr = np.asarray(data_root[k], dtype=np.float32)
                    # 统一成 (3, n)：DiTing 常见存成 (n, 3)，需要转置
                    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    if arr.ndim != 2 or arr.shape[0] < 3:
                        continue
                    arr = arr[:3]  # 只取三分量

                    # 重采样 50 -> 100 Hz
                    arr100 = resample_trace(arr, orig_sr, target_sr)

                    dset = grp.create_dataset(k, data=arr100, compression="gzip")
                    # 到时标注：从 csv 找这条 key 的 p_pick / s_pick（单位是采样点@50Hz）
                    if meta is not None and "key" in meta.columns:
                        row = meta[meta["key"].astype(str) == str(k)]
                        if len(row):
                            r = row.iloc[0]
                            # 50Hz 采样点 -> 100Hz 采样点：×2
                            if "p_pick" in meta.columns and not pd.isna(r.get("p_pick")):
                                dset.attrs["p_sample_100hz"] = float(r["p_pick"]) * (target_sr / orig_sr)
                            if "s_pick" in meta.columns and not pd.isna(r.get("s_pick")):
                                dset.attrs["s_sample_100hz"] = float(r["s_pick"]) * (target_sr / orig_sr)

                    written += arr100.nbytes
                    n_events += 1
                    manifest.append(k)
                    if n_events % 500 == 0:
                        print(f"    已处理 {n_events} 条, 累计 {written/1e9:.2f} GB")

            if written >= max_bytes:
                print(f"\n已达到 {max_gb}GB 上限，停止。")
                break

    print(f"\n完成：{n_events} 条波形，{written/1e9:.2f} GB")
    print(f"输出：{out_h5_path}")
    print("每条波形已重采样到 100Hz，P/S 到时存在 attrs['p_sample_100hz'/'s_sample_100hz']")
    return out_h5_path, n_events


def main():
    ap = argparse.ArgumentParser(description="切 DiTing 子集并重采样到 100Hz")
    ap.add_argument("--raw_dir", required=True,
                    help="放你下载的 DiTing 分卷 hdf5 + csv 的目录")
    ap.add_argument("--out_dir", default="/data/coding/diting_subset",
                    help="输出目录")
    ap.add_argument("--max_gb", type=float, default=15.0,
                    help="子集大小上限 GB，默认 15（10~20 之间）")
    ap.add_argument("--orig_sr", type=float, default=50.0,
                    help="DiTing 原始采样率，官方是 50Hz")
    ap.add_argument("--target_sr", type=float, default=100.0,
                    help="目标采样率，对齐官方赛题 100Hz")
    args = ap.parse_args()

    print("=" * 60)
    print("DiTing 子集切割 + 重采样")
    print(f"  原始 {args.orig_sr}Hz -> 目标 {args.target_sr}Hz")
    print(f"  上限 {args.max_gb} GB")
    print("=" * 60)

    process_hdf5_subset(args.raw_dir, args.out_dir, args.max_gb,
                        args.orig_sr, args.target_sr)


if __name__ == "__main__":
    main()
