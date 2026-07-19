"""SeisBench 下载 DiTing 子集 + 重采样到 100Hz + 断点续传.

===== 为什么要断点续传（写给非 AI 背景的队友）=====
DiTing 在 SeisBench 里体量巨大，网络可能中途断、GPU 机器还会关机清空。
如果每次中断都从头再来，宝贵的下载时间全浪费了。所以本脚本做成"可续"：
  - 每处理完一条波形就写盘、并把"已完成的条数"记进 progress.json；
  - 中断后重跑同一条命令，它读 progress.json，自动跳过已完成的，从断点接着下；
  - 支持 Ctrl+C 优雅退出：退出前保存进度，下次接着来。

===== 关键处理 =====
1. DiTing 是 50Hz，官方赛题是 100Hz —— 强制重采样到 100Hz，
   且 P/S 到时的采样点下标同步 ×2，否则到时会系统性偏移。
2. 输出增量写入 HDF5（append），中断不丢已下的部分。
3. 用 --max_traces 或 --max_gb 控制子集大小，别把盘写爆。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def resample_50_to_100(wave: np.ndarray) -> np.ndarray:
    """把 (3, n) 的 50Hz 波形线性插值上采样到 100Hz -> (3, 2n-1 近似 2n)。"""
    c, n = wave.shape
    old_t = np.arange(n)
    new_t = np.linspace(0, n - 1, num=2 * n)  # 2 倍点数
    out = np.empty((c, len(new_t)), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(new_t, old_t, wave[i]).astype(np.float32)
    return out


def load_progress(progress_path: str) -> dict:
    """读断点进度；没有则从头开始。"""
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"done": 0, "written_bytes": 0}


def save_progress(progress_path: str, done: int, written_bytes: int) -> None:
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": done, "written_bytes": written_bytes}, f)
    os.replace(tmp, progress_path)  # 原子替换，防写一半被关机损坏


def main():
    parser = argparse.ArgumentParser(description="SeisBench DiTing 子集下载(断点续传)")
    parser.add_argument("--cache", default="/data/seisbench_cache",
                        help="SeisBench 缓存目录(数据下载到这)")
    parser.add_argument("--out", default="/data/coding/diting_subset_100hz.hdf5",
                        help="切好的子集输出文件")
    parser.add_argument("--max_traces", type=int, default=20000,
                        help="最多处理多少条波形(控制子集大小)")
    parser.add_argument("--max_gb", type=float, default=15.0,
                        help="子集大小上限(GB),到了就停")
    parser.add_argument("--save_every", type=int, default=200,
                        help="每处理多少条存一次进度")
    args = parser.parse_args()

    # 缓存目录必须在装 seisbench 前设好
    os.environ["SEISBENCH_CACHE_ROOT"] = args.cache
    os.makedirs(args.cache, exist_ok=True)

    import h5py
    import seisbench.data as sbd

    progress_path = args.out + ".progress.json"
    prog = load_progress(progress_path)
    start_idx = prog["done"]
    written_bytes = prog["written_bytes"]

    print("==== 加载 DiTing 元数据(首次会下元数据,稍等) ====")
    print(f"缓存目录: {args.cache}")
    print(f"断点续传: 从第 {start_idx} 条开始(已完成 {start_idx} 条, "
          f"已写 {written_bytes/1e9:.2f}GB)")

    # 惰性下载:只在真正读某条波形时才下那条数据块
    data = sbd.DiTing(sampling_rate=None, cache="trace")  # sampling_rate=None: 拿原始50Hz,自己重采样
    meta = data.metadata
    total = len(meta)
    print(f"DiTing 总波形数: {total}")

    limit = min(start_idx + args.max_traces, total)
    max_bytes = args.max_gb * 1e9

    # 输出文件:续传用 append,首次用 write
    mode = "a" if os.path.exists(args.out) and start_idx > 0 else "w"
    h5 = h5py.File(args.out, mode)
    grp = h5.require_group("data")

    done = start_idx
    try:
        for idx in range(start_idx, limit):
            if written_bytes >= max_bytes:
                print(f"\n已达 {args.max_gb}GB 上限,停止。")
                break

            # 读第 idx 条(此处触发该条的惰性下载)
            wave, mrow = data.get_sample(idx)
            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim != 2:
                continue
            if wave.shape[0] > wave.shape[1]:  # (n,3) -> (3,n)
                wave = wave.T

            # 50Hz -> 100Hz,到时下标同步 ×2
            wave100 = resample_50_to_100(wave)
            p_pick = mrow.get("trace_p_arrival_sample", mrow.get("p_pick"))
            s_pick = mrow.get("trace_s_arrival_sample", mrow.get("s_pick"))
            p100 = float(p_pick) * 2 if p_pick is not None and not _isnan(p_pick) else -1.0
            s100 = float(s_pick) * 2 if s_pick is not None and not _isnan(s_pick) else -1.0

            key = f"trace_{idx:08d}"
            if key in grp:  # 续传时可能已存在,跳过
                done = idx + 1
                continue
            dset = grp.create_dataset(key, data=wave100, compression="gzip", compression_opts=4)
            dset.attrs["p_sample_100hz"] = p100
            dset.attrs["s_sample_100hz"] = s100
            dset.attrs["sampling_rate"] = 100.0

            written_bytes += wave100.nbytes
            done = idx + 1

            if done % args.save_every == 0:
                h5.flush()
                save_progress(progress_path, done, written_bytes)
                print(f"  进度 {done}/{limit}  已写 {written_bytes/1e9:.2f}GB", flush=True)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 中断,保存断点...")
    finally:
        h5.flush()
        h5.close()
        save_progress(progress_path, done, written_bytes)
        print(f"\n断点已保存: 完成 {done} 条, 共 {written_bytes/1e9:.2f}GB")
        print(f"重跑同一条命令即可从第 {done} 条继续。")


def _isnan(x) -> bool:
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr)
        sys.exit(1)
