"""GeoNet(新西兰) FDSN 直连取数 + 重采样到 100Hz + 断点续传.

===== 为什么单独写这个脚本（写给非 AI 背景的队友）=====
DiTing/ETHZ 是 SeisBench 内置数据集，一行 `sbd.DiTing()` 就能下。
但**新西兰的 GeoNet 不是 SeisBench 内置数据集**，必须走 obspy 的 FDSN 协议
自己拉：先查"事件目录"（哪年哪月发生了哪些地震、每个地震在各台站的 P/S 到时），
再按 (事件, 台站) 逐条下三分量波形。所以本脚本比 DiTing 那个复杂一层。

好消息：GeoNet 的 FDSN 服务是**完全开放的**，不需要注册账号或密钥。

===== 产出格式（与 diting_seisbench.py 严格一致，微调脚本零改动即可读）=====
HDF5：group "data" 下每条 dataset = 一条波形 (3, win)，attrs：
  - p_sample_100hz / s_sample_100hz : P/S 到时在窗口内的采样点下标（缺失 = -1）
  - sampling_rate : 100.0
这正是 finetune_phasenet.py 里 load_hdf5_dataset 认的格式。

===== 抗中断设计（对齐 Colab 免费版会话时限 + 断连清空）=====
1. progress.json 记"已完成事件数"；重跑同一条命令自动跳过已完成事件，从断点续拉。
2. Ctrl+C 优雅退出：退出前落盘 + 存进度。
3. --max_minutes 软时限：接近 Colab 单次上限前主动落盘退出，不丢已下的部分。
4. 输出用 append 模式，已存在的 key 跳过，续传不重复下载。

===== 关键处理 =====
- GeoNet 波形采样率不统一（常见 50Hz/100Hz/200Hz）——一律强制重采样到 100Hz，
  且 P/S 到时的采样点下标按比例同步换算，否则到时会系统性偏移（和 DiTing 同款坑）。
- 每条窗口以 P 到时为锚点裁固定长度 win（默认 3001，PhaseNet 默认输入长度），
  尽量把 S 也框进来；到时落窗口外则记 -1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 纯逻辑（不依赖 obspy/torch）—— 可单测，避免"到时错位"这类隐蔽 bug
# ---------------------------------------------------------------------------

def resample_to_100(wave: np.ndarray, orig_sr: float) -> np.ndarray:
    """把 (3, n) 波形从 orig_sr 线性插值重采样到 100Hz。

    上采样(50->100)点数翻倍，下采样(200->100)点数减半。线性插值对微调足够。
    """
    if abs(orig_sr - 100.0) < 1e-6:
        return wave.astype(np.float32)
    c, n = wave.shape
    ratio = 100.0 / orig_sr
    new_n = max(1, int(round(n * ratio)))
    old_t = np.arange(n)
    new_t = np.linspace(0, n - 1, num=new_n)
    out = np.empty((c, new_n), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(new_t, old_t, wave[i]).astype(np.float32)
    return out


def rescale_sample_index(sample_idx: float, orig_sr: float) -> float:
    """把某到时在 orig_sr 下的采样点下标，换算到 100Hz 下的下标。

    下标随采样率等比缩放：new = old * (100 / orig_sr)。
    """
    if sample_idx is None or sample_idx < 0:
        return -1.0
    return float(sample_idx) * (100.0 / orig_sr)


def cut_window(wave100: np.ndarray, p_idx100: float, s_idx100: float, win: int):
    """以 P 到时为锚点，从 100Hz 波形裁一段固定长度 win 的窗口。

    优先让 P 落在窗口靠前 1/3 处，尽量把后面的 S 也框进来。
    返回 (window(3,win), p_in_win, s_in_win)；到时落窗外记 -1。
    """
    c, n = wave100.shape
    anchor = int(round(p_idx100)) if p_idx100 >= 0 else n // 3
    start = int(np.clip(anchor - win // 3, 0, max(0, n - win)))

    if n >= win:
        w = wave100[:, start:start + win]
    else:
        w = np.zeros((c, win), dtype=np.float32)
        w[:, :n] = wave100

    def _shift(idx):
        if idx < 0:
            return -1.0
        j = idx - start
        return float(j) if 0 <= j < win else -1.0

    return w.astype(np.float32), _shift(p_idx100), _shift(s_idx100)


def load_progress(progress_path: str) -> dict:
    """读断点进度；没有则从头开始。"""
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"done_events": 0, "written_traces": 0, "written_bytes": 0}


def save_progress(progress_path: str, done_events: int,
                  written_traces: int, written_bytes: int) -> None:
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "done_events": done_events,
            "written_traces": written_traces,
            "written_bytes": written_bytes,
        }, f)
    os.replace(tmp, progress_path)  # 原子替换，防写一半被关机损坏


# ---------------------------------------------------------------------------
# FDSN 取数（依赖 obspy）—— 在 Colab/GPU 上跑
# ---------------------------------------------------------------------------

def _extract_picks(event):
    """从一个 obspy Event 提取 {station_code: {"P": UTCDateTime, "S": UTCDateTime}}。

    GeoNet 的 QuakeML 里，pick 通过 arrival.phase 或 pick.phase_hint 标注 P/S，
    waveform_id 里含台站码。这里两种来源都兜住。
    """
    picks_by_id = {}
    for pk in event.picks:
        wid = pk.waveform_id
        sta = getattr(wid, "station_code", None) if wid is not None else None
        phase = (pk.phase_hint or "").upper().strip()
        picks_by_id[pk.resource_id.id] = (sta, phase, pk.time)

    result = {}
    origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
    # 优先用 origin.arrivals 的 phase（更权威），回落到 pick.phase_hint
    if origin is not None and origin.arrivals:
        for arr in origin.arrivals:
            pick_id = getattr(arr, "pick_id", None)
            pid = pick_id.id if pick_id is not None else None
            if pid not in picks_by_id:
                continue
            sta, hint, t = picks_by_id[pid]
            phase = (arr.phase or hint or "").upper().strip()
            if sta is None or not phase:
                continue
            _stash_pick(result, sta, phase, t)
    else:
        for sta, phase, t in picks_by_id.values():
            if sta and phase:
                _stash_pick(result, sta, phase, t)
    return result


def _stash_pick(result: dict, sta: str, phase: str, t) -> None:
    """把 pick 归入 station 的 P/S 槽位（只认 P/S，取每站最早的一个）。"""
    kind = "P" if phase.startswith("P") else "S" if phase.startswith("S") else None
    if kind is None:
        return
    slot = result.setdefault(sta, {})
    if kind not in slot or t < slot[kind]:
        slot[kind] = t


def main():
    parser = argparse.ArgumentParser(
        description="GeoNet(新西兰) FDSN 取数 -> 100Hz -> 标准HDF5 (断点续传)")
    parser.add_argument("--out", default="/content/geonet_100hz.hdf5",
                        help="输出 HDF5 (Colab 建议指到 Drive 目录，断连不丢)")
    parser.add_argument("--start", default="2023-01-01",
                        help="事件起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01",
                        help="事件结束日期 YYYY-MM-DD")
    parser.add_argument("--minmag", type=float, default=3.0, help="最小震级")
    parser.add_argument("--maxmag", type=float, default=6.0, help="最大震级(避开超大震削波)")
    # 新西兰主震区（大致），默认圈住南北岛
    parser.add_argument("--lat", type=float, default=-41.0, help="中心纬度")
    parser.add_argument("--lon", type=float, default=174.0, help="中心经度")
    parser.add_argument("--maxradius", type=float, default=6.0,
                        help="事件搜索半径(度)")
    parser.add_argument("--maxstations", type=int, default=8,
                        help="每个事件最多取多少台(控制单事件条数)")
    parser.add_argument("--win", type=int, default=3001,
                        help="每条波形窗口长度(PhaseNet默认3001)")
    parser.add_argument("--pre", type=float, default=10.0,
                        help="P到时前取多少秒(用于定位窗口起点)")
    parser.add_argument("--max_traces", type=int, default=20000,
                        help="最多写多少条波形(控制子集大小)")
    parser.add_argument("--max_gb", type=float, default=10.0,
                        help="子集大小上限(GB)")
    parser.add_argument("--max_minutes", type=float, default=0.0,
                        help="软时限(分钟),>0 时接近 Colab 会话上限前主动落盘退出;0=不限")
    parser.add_argument("--channels", default="HH?,BH?,EH?",
                        help="优先通道(逗号分隔,按序尝试)")
    parser.add_argument("--save_every", type=int, default=20,
                        help="每处理多少事件存一次进度")
    args = parser.parse_args()

    import h5py
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNException

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    progress_path = args.out + ".progress.json"
    prog = load_progress(progress_path)
    start_event = prog["done_events"]
    written_traces = prog["written_traces"]
    written_bytes = prog["written_bytes"]

    print("==== GeoNet FDSN 取数 ====")
    print(f"时间窗 {args.start} ~ {args.end} | 震级 {args.minmag}~{args.maxmag} | "
          f"区域 中心({args.lat},{args.lon}) 半径{args.maxradius}°")
    print(f"断点续传: 从第 {start_event} 个事件开始 "
          f"(已写 {written_traces} 条, {written_bytes/1e9:.2f}GB)")

    client = Client("GEONET")

    print("==== 查询事件目录 ====")
    try:
        cat = client.get_events(
            starttime=UTCDateTime(args.start),
            endtime=UTCDateTime(args.end),
            minmagnitude=args.minmag,
            maxmagnitude=args.maxmag,
            latitude=args.lat, longitude=args.lon, maxradius=args.maxradius,
            orderby="time-asc",
        )
    except FDSNException as exc:
        print(f"[事件查询失败] {exc!r}", file=sys.stderr)
        sys.exit(2)
    total_events = len(cat)
    print(f"命中事件数: {total_events}")

    mode = "a" if os.path.exists(args.out) and start_event > 0 else "w"
    h5 = h5py.File(args.out, mode)
    grp = h5.require_group("data")
    chan_prefs = [c.strip() for c in args.channels.split(",") if c.strip()]
    max_bytes = args.max_gb * 1e9
    t_begin = time.time()
    done_events = start_event

    try:
        for ei in range(start_event, total_events):
            if written_bytes >= max_bytes:
                print(f"\n已达 {args.max_gb}GB 上限,停止。")
                break
            if written_traces >= args.max_traces:
                print(f"\n已达 {args.max_traces} 条上限,停止。")
                break
            if args.max_minutes > 0 and (time.time() - t_begin) / 60.0 >= args.max_minutes:
                print(f"\n达软时限 {args.max_minutes} 分钟,主动落盘退出(重跑续拉)。")
                break

            event = cat[ei]
            picks = _extract_picks(event)
            if not picks:
                done_events = ei + 1
                continue

            n_this = 0
            for sta, ps in picks.items():
                if n_this >= args.maxstations:
                    break
                p_t = ps.get("P")
                s_t = ps.get("S")
                if p_t is None:  # 至少要有 P 才裁窗
                    continue

                key = f"ev{ei:06d}_{sta}"
                if key in grp:  # 续传去重
                    continue

                # 取窗:P 前 pre 秒起,长度覆盖 win/100 秒 + 余量
                t0 = p_t - args.pre
                dur = args.win / 100.0 + args.pre + 5.0
                st = _fetch_stream(client, sta, t0, dur, chan_prefs)
                if st is None or len(st) < 3:
                    continue

                arr, orig_sr = _stream_to_array(st)
                if arr is None:
                    continue

                # 到时 -> 原采样率下的样本下标(相对 t0)
                p_idx = (p_t - t0) * orig_sr
                s_idx = (s_t - t0) * orig_sr if s_t is not None else -1.0

                wave100 = resample_to_100(arr, orig_sr)
                p100 = rescale_sample_index(p_idx, orig_sr)
                s100 = rescale_sample_index(s_idx, orig_sr)
                w, p_in, s_in = cut_window(wave100, p100, s100, args.win)

                dset = grp.create_dataset(
                    key, data=w, compression="gzip", compression_opts=4)
                dset.attrs["p_sample_100hz"] = float(p_in)
                dset.attrs["s_sample_100hz"] = float(s_in)
                dset.attrs["sampling_rate"] = 100.0
                dset.attrs["station"] = sta
                written_bytes += w.nbytes
                written_traces += 1
                n_this += 1

            done_events = ei + 1
            if done_events % args.save_every == 0:
                h5.flush()
                save_progress(progress_path, done_events, written_traces, written_bytes)
                print(f"  事件 {done_events}/{total_events} | 已写 {written_traces} 条 "
                      f"{written_bytes/1e9:.2f}GB", flush=True)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 中断,保存断点...")
    finally:
        h5.flush()
        h5.close()
        save_progress(progress_path, done_events, written_traces, written_bytes)
        print(f"\n断点已保存: 完成 {done_events} 个事件, 写 {written_traces} 条, "
              f"{written_bytes/1e9:.2f}GB")
        print(f"重跑同一条命令即可从第 {done_events} 个事件继续。")


def _fetch_stream(client, station, t0, dur, chan_prefs):
    """按通道优先级尝试取某台三分量波形。任一优先级取到即返回。"""
    from obspy.clients.fdsn.header import FDSNException
    for chan in chan_prefs:
        try:
            st = client.get_waveforms(
                network="*", station=station, location="*",
                channel=chan, starttime=t0, endtime=t0 + dur)
            if st is not None and len(st) >= 3:
                return st
        except (FDSNException, Exception):  # noqa: BLE001 — 缺数据是常态,跳过
            continue
    return None


def _stream_to_array(st):
    """把 obspy Stream 转成 (3, n) 数组 + 采样率。

    取 Z/N/E(或 1/2/Z)三分量,按 Z,N,E 顺序对齐;长度不齐则裁到最短。
    返回 (array, sampling_rate);无法组成三分量返回 (None, None)。
    """
    st = st.copy()
    st.merge(method=1, fill_value=0)  # 合并同 id 的分段
    # 按分量分组:取每个分量第一条
    comp = {}
    for tr in st:
        c = tr.stats.channel[-1].upper()
        comp.setdefault(c, tr)

    def pick(*names):
        for nm in names:
            if nm in comp:
                return comp[nm]
        return None

    z = pick("Z")
    n = pick("N", "1")
    e = pick("E", "2")
    if z is None or n is None or e is None:
        return None, None

    sr = float(z.stats.sampling_rate)
    m = min(len(z.data), len(n.data), len(e.data))
    if m <= 0:
        return None, None
    arr = np.vstack([z.data[:m], n.data[:m], e.data[:m]]).astype(np.float32)
    return arr, sr


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr)
        sys.exit(1)
