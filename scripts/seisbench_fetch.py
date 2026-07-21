"""任意 SeisBench 内置数据集 -> 100Hz -> 标准HDF5 (断点续传 + 软时限).

===== 这个脚本做什么（写给非 AI 背景的队友）=====
DiTing 先不用了。SeisBench 里还有一大堆现成、开放、一行就能拉的地震数据集
（美国 STEAD、意大利 INSTANCE、瑞士 ETHZ、全球 GEOFON、南加州 SCEDC、智利 Iquique
、全球小震 LenDB、美国地质调查局 NEIC、德州 TXED、美西北 PNW…）。本脚本把
diting_seisbench.py 泛化成"选哪个数据集都行"：用 --dataset 指定名字即可。

产出格式与 geonet_fetch.py / diting_seisbench.py **严格一致**，所以
finetune_phasenet.py 一行不用改就能读：
  HDF5: group "data" 下每条 dataset = 一条波形 (3, win 或原长)，attrs：
    - p_sample_100hz / s_sample_100hz : P/S 到时在该波形内的采样点下标（缺失 = -1）
    - sampling_rate : 100.0

===== 和 DiTing 脚本最关键的一处不同（务必看懂，否则到时会系统性错位）=====
diting_seisbench.py 用 sampling_rate=None 拿原始 50Hz，再手动把到时下标 ×2。
那个 ×2 是 DiTing 专属的补丁。**本脚本改为让 SeisBench 直接重采样到 100Hz**
（构造时传 sampling_rate=100）。SeisBench 会自动把所有以 `_sample` 结尾的元数据列
（含 trace_p_arrival_sample / trace_s_arrival_sample）按重采样因子换算好
（见 seisbench/data/base.py 的 _get_sample_unify_sampling_rate）。
所以这里【绝不能再手动 ×N】——STEAD/INSTANCE/SCEDC/TXED/PNW 原生就是 100Hz，
再 ×2 会把到时下标翻倍，模型学到的 P/S 位置全错。

===== 抗中断设计（对齐 Colab 免费版会话时限 + 断连清空）=====
1. progress.json 记"已处理条数"；重跑同一条命令自动跳过已完成，从断点续拉。
2. Ctrl+C 优雅退出：退出前落盘 + 存进度。
3. --max_minutes 软时限：接近 Colab 单次上限前主动落盘退出，不丢已下的部分。
4. 输出用 append 模式，已存在的 key 跳过，续传不重复处理。

===== 惰性下载 =====
构造 sbd.<Dataset>(cache="trace") 时只下载【元数据 CSV】（几十~几百 MB）。
波形按条惰性下载，只有真正读到某条时才下那条。用 --max_traces / --max_gb 封顶，
不会把整个数据集（可能几十上百 GB）拖下来。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 到时列名解析（纯逻辑，不依赖 seisbench，便于单测）
# ---------------------------------------------------------------------------

# 大多数数据集用这两个规范列名；ETHZ/GEOFON 等会用带震相细分的变体
# （trace_Pg_arrival_sample / trace_Pn_arrival_sample / trace_Sg_... 等）。
_P_PRIMARY = "trace_p_arrival_sample"
_S_PRIMARY = "trace_s_arrival_sample"
_P_VARIANT = re.compile(r"^trace_P.*_arrival_sample$")
_S_VARIANT = re.compile(r"^trace_S.*_arrival_sample$")


def _first_valid(meta: dict, primary: str, variant: re.Pattern):
    """取一条到时下标：先认规范列名，缺了再按震相变体正则找第一个非空。

    返回浮点采样点下标（已是目标采样率下的值），找不到返回 -1.0。
    """
    val = meta.get(primary, None)
    if _is_valid_sample(val):
        return float(np.asarray(val).reshape(-1)[0])
    # 回落：扫所有匹配变体的列，取第一个有效值（ETHZ/GEOFON 的 Pg/Pn/Sg/Sn）
    for key in meta:
        if variant.match(str(key)):
            v = meta.get(key)
            if _is_valid_sample(v):
                return float(np.asarray(v).reshape(-1)[0])
    return -1.0


def _is_valid_sample(v) -> bool:
    if v is None:
        return False
    try:
        arr = np.asarray(v, dtype="float64").reshape(-1)
    except (TypeError, ValueError):
        return False
    if arr.size == 0:
        return False
    x = arr[0]
    return np.isfinite(x) and x >= 0


def resolve_picks(meta: dict):
    """从 get_sample 返回的 metadata dict 解析 (p_sample, s_sample)@目标采样率。

    这里的下标已由 SeisBench 换算到目标采样率，本函数不做任何 ×N 缩放。
    """
    return _first_valid(meta, _P_PRIMARY, _P_VARIANT), _first_valid(meta, _S_PRIMARY, _S_VARIANT)


def to_3xn(wave: np.ndarray) -> np.ndarray | None:
    """把 get_sample 的波形规整成 (3, n)：不足三分量返回 None，(n,3) 自动转置。"""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2:
        return None
    if wave.shape[0] > wave.shape[1]:  # (n, C) -> (C, n)
        wave = wave.T
    if wave.shape[0] < 3:
        return None
    return np.ascontiguousarray(wave[:3])


# ---------------------------------------------------------------------------
# 断点进度
# ---------------------------------------------------------------------------

def load_progress(progress_path: str) -> dict:
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"done": 0, "written": 0, "written_bytes": 0}


def save_progress(progress_path: str, done: int, written: int, written_bytes: int) -> None:
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": done, "written": written, "written_bytes": written_bytes}, f)
    os.replace(tmp, progress_path)  # 原子替换，防写一半被关机损坏


# ---------------------------------------------------------------------------
# 数据集解析
# ---------------------------------------------------------------------------

# 队友常用的、含 P/S 到时的震相拾取类数据集。名字大小写不敏感。
# 括号里是原生采样率，仅供参考——SeisBench 会统一重采样到 100Hz，无需手动处理。
_KNOWN = {
    "stead": "STEAD",                      # 美国 STEAD (100Hz)
    "instance": "InstanceCountsCombined",  # 意大利 INSTANCE (100Hz)
    "ethz": "ETHZ",                        # 瑞士 (混合采样率)
    "geofon": "GEOFON",                    # 全球 GEOFON (混合)
    "scedc": "SCEDC",                      # 南加州 (100Hz, 体量大)
    "iquique": "Iquique",                  # 智利 Iquique (100Hz, 小而快，适合先跑通)
    "lendb": "LenDB",                      # 全球小震 (原生 20Hz)
    "neic": "NEIC",                        # USGS 全球 (原生 40Hz)
    "txed": "TXED",                        # 德州 (100Hz)
    "pnw": "PNW",                          # 美国西北 (100Hz)
}


def resolve_dataset_class(name: str):
    """把用户给的名字解析成 seisbench.data 里的类。支持别名和原类名。"""
    import seisbench.data as sbd

    key = name.strip().lower()
    cls_name = _KNOWN.get(key, name.strip())
    cls = getattr(sbd, cls_name, None)
    if cls is None:
        avail = ", ".join(sorted(_KNOWN))
        raise SystemExit(
            f"未知数据集 {name!r}。可用别名: {avail}\n"
            f"（也可直接给 seisbench.data 里的类名，如 InstanceCountsCombined、CEED 等）"
        )
    return cls, cls_name


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="任意 SeisBench 数据集 -> 100Hz 标准HDF5 (断点续传, 软时限)")
    parser.add_argument("--dataset", default="iquique",
                        help="数据集名/别名: " + ", ".join(sorted(_KNOWN))
                             + "（默认 iquique，小而快，先用它跑通全链路）")
    parser.add_argument("--cache", default="/content/drive/MyDrive/dizheng/seisbench_cache",
                        help="SeisBench 缓存目录（Colab 建议指到 Drive，断连不丢元数据）")
    parser.add_argument("--out", default="/content/drive/MyDrive/dizheng/data/seisbench_100hz.hdf5",
                        help="输出 HDF5（Colab 建议指到 Drive 目录）")
    parser.add_argument("--split", default="",
                        help="只取某个 split(train/dev/test)；留空=全部。部分数据集无 split 列，留空即可")
    parser.add_argument("--require-s", action="store_true",
                        help="只保留同时有 P 和 S 到时的波形（默认只要有 P 就收）")
    parser.add_argument("--target-sr", type=float, default=100.0,
                        help="目标采样率，对齐官方赛题 100Hz（一般不用改）")
    parser.add_argument("--max_traces", type=int, default=20000,
                        help="最多写多少条波形（控制子集大小）")
    parser.add_argument("--max_gb", type=float, default=10.0,
                        help="子集大小上限(GB)，到了就停")
    parser.add_argument("--max_minutes", type=float, default=0.0,
                        help="软时限(分钟)，>0 时接近 Colab 会话上限前主动落盘退出；0=不限")
    parser.add_argument("--save_every", type=int, default=200,
                        help="每处理多少条存一次进度")
    parser.add_argument("--limit-scan", type=int, default=0,
                        help="最多扫描元数据前 N 条(0=全部)；数据集很大时用它缩短扫描范围")
    args = parser.parse_args()

    # 缓存目录必须在 import seisbench 前设好（seisbench 读环境变量）
    os.environ.setdefault("SEISBENCH_CACHE_ROOT", args.cache)
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    import h5py

    cls, cls_name = resolve_dataset_class(args.dataset)

    progress_path = args.out + ".progress.json"
    prog = load_progress(progress_path)
    start_idx = prog["done"]
    written = prog["written"]
    written_bytes = prog["written_bytes"]

    print("=" * 64)
    print(f"数据集: {args.dataset} -> seisbench.data.{cls_name}")
    print(f"缓存目录: {args.cache}")
    print(f"输出: {args.out}")
    print(f"断点续传: 从第 {start_idx} 条开始（已写 {written} 条, {written_bytes/1e9:.2f}GB）")
    print("=" * 64)

    print("==== 加载元数据（首次会下该数据集的 CSV 元数据，稍等）====")
    # sampling_rate=target_sr：让 SeisBench 统一重采样并【自动换算到时下标】。
    # cache="trace"：波形按条惰性下载，不会一次性拖下整个数据集。
    data = cls(sampling_rate=args.target_sr, cache="trace")

    # 可选 split 过滤（有的数据集没有 split 列，get_split 会给出提示）
    if args.split:
        try:
            data = data.get_split(args.split)
            print(f"已筛 split={args.split}")
        except Exception as exc:  # noqa: BLE001
            print(f"[提示] 该数据集不支持 split={args.split}（{exc!r}），改用全部数据。")

    total = len(data)
    scan_limit = total if args.limit_scan <= 0 else min(total, start_idx + args.limit_scan)
    print(f"{cls_name} 波形总数: {total}（本次扫描到第 {scan_limit} 条为止）")

    mode = "a" if os.path.exists(args.out) and start_idx > 0 else "w"
    h5 = h5py.File(args.out, mode)
    grp = h5.require_group("data")
    max_bytes = args.max_gb * 1e9
    t_begin = time.time()
    done = start_idx
    skipped_no_p = 0

    try:
        for idx in range(start_idx, scan_limit):
            if written_bytes >= max_bytes:
                print(f"\n已达 {args.max_gb}GB 上限，停止。")
                break
            if written >= args.max_traces:
                print(f"\n已达 {args.max_traces} 条上限，停止。")
                break
            if args.max_minutes > 0 and (time.time() - t_begin) / 60.0 >= args.max_minutes:
                print(f"\n达软时限 {args.max_minutes} 分钟，主动落盘退出（重跑续拉）。")
                break

            key = f"trace_{idx:08d}"
            if key in grp:  # 续传去重
                done = idx + 1
                continue

            # 读第 idx 条（触发该条惰性下载 + 重采样 + 到时自动换算）
            try:
                wave, meta = data.get_sample(idx)
            except Exception as exc:  # noqa: BLE001 — 单条坏数据不该拖垮整批
                print(f"  [跳过] 第 {idx} 条读取失败: {exc!r}")
                done = idx + 1
                continue

            wave = to_3xn(wave)
            if wave is None:
                done = idx + 1
                continue

            p100, s100 = resolve_picks(meta)  # 已是 100Hz 下的下标，勿再 ×N
            if p100 < 0:                       # 至少要有 P 才有训练价值
                skipped_no_p += 1
                done = idx + 1
                continue
            if args.require_s and s100 < 0:
                done = idx + 1
                continue

            dset = grp.create_dataset(key, data=wave, compression="gzip", compression_opts=4)
            dset.attrs["p_sample_100hz"] = float(p100)
            dset.attrs["s_sample_100hz"] = float(s100)
            dset.attrs["sampling_rate"] = float(args.target_sr)
            dset.attrs["source_dataset"] = cls_name
            written_bytes += wave.nbytes
            written += 1
            done = idx + 1

            if written % args.save_every == 0:
                h5.flush()
                save_progress(progress_path, done, written, written_bytes)
                print(f"  进度 {done}/{scan_limit} | 已写 {written} 条 "
                      f"{written_bytes/1e9:.2f}GB", flush=True)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 中断，保存断点...")
    finally:
        h5.flush()
        h5.close()
        save_progress(progress_path, done, written, written_bytes)
        print(f"\n断点已保存: 扫到第 {done} 条, 写 {written} 条, {written_bytes/1e9:.2f}GB")
        if skipped_no_p:
            print(f"（其中 {skipped_no_p} 条因无 P 到时被跳过——噪声/无标注样本，正常现象）")
        print(f"重跑同一条命令即可从第 {done} 条继续。")
        print(f"输出: {args.out}  ->  可直接喂给 finetune_phasenet.py 的 --data / --holdout")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr)
        sys.exit(1)
