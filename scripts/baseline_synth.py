"""基线推理验证（合成波形，零下载）.

背景：这台国内机器跨境下载 SeisBench 数据集龟速，但我们其实不需要真实数据集来
验证链路。本脚本自己合成几条"带 P/S 震相的三分量波形"，喂给已下载的预训练
PhaseNet，看它能否拾取出 P/S 到时。跑通即证明：环境 + GPU + 预训练模型 + 推理
全链路 OK。等官方数据到位，把合成波形换成官方波形即可。

合成方法（简化但物理合理）：
  - 100Hz 采样，60 秒 = 6000 点，三分量 (Z,N,E)
  - 背景加低幅高斯噪声
  - 在某个时刻注入 P 波：高频衰减振荡，主要出现在 Z（垂直）分量
  - 稍后注入 S 波：较低频、幅值更大，主要出现在水平分量 (N,E)
  - 我们知道注入的真实 P/S 位置，可与模型拾取结果对照
"""

import sys
import numpy as np


def synth_trace(n=6000, sr=100.0, p_sample=1500, s_sample=2800, seed=0):
    """合成一条三分量波形，返回 (3, n) 数组与真实 P/S 采样点。"""
    rng = np.random.RandomState(seed)
    z = rng.normal(0, 0.02, n)
    ns = rng.normal(0, 0.02, n)
    e = rng.normal(0, 0.02, n)

    # P 波：高频衰减振荡，主要在 Z 分量
    tp = np.arange(0, 400)
    p_wave = np.exp(-tp / 120.0) * np.sin(2 * np.pi * 8.0 * tp / sr)
    z[p_sample:p_sample + len(tp)] += 1.0 * p_wave
    ns[p_sample:p_sample + len(tp)] += 0.3 * p_wave
    e[p_sample:p_sample + len(tp)] += 0.3 * p_wave

    # S 波：较低频、幅值更大，主要在水平分量
    ts = np.arange(0, 600)
    s_wave = np.exp(-ts / 200.0) * np.sin(2 * np.pi * 3.5 * ts / sr)
    ns[s_sample:s_sample + len(ts)] += 1.6 * s_wave
    e[s_sample:s_sample + len(ts)] += 1.6 * s_wave
    z[s_sample:s_sample + len(ts)] += 0.4 * s_wave

    return np.vstack([z, ns, e]).astype("float32"), p_sample, s_sample


def main():
    import torch
    import seisbench.models as sbm
    from obspy import Stream, Trace, UTCDateTime

    print("==================== 加载预训练 PhaseNet ====================")
    model = sbm.PhaseNet.from_pretrained("stead")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"模型就位，device={device}")

    print("\n==================== 合成 3 条波形并推理 ====================")
    sr = 100.0
    cases = [
        dict(p_sample=1500, s_sample=2800, seed=1),
        dict(p_sample=2000, s_sample=3500, seed=2),
        dict(p_sample=1000, s_sample=2200, seed=3),
    ]
    for i, c in enumerate(cases):
        arr, p_true, s_true = synth_trace(sr=sr, **c)
        st = Stream()
        for ch, name in zip(arr, ["Z", "N", "E"]):
            tr = Trace(data=ch)
            tr.stats.sampling_rate = sr
            tr.stats.starttime = UTCDateTime(0)
            tr.stats.channel = "HH" + name
            tr.stats.station = "SYN"
            st.append(tr)

        out = model.classify(st)
        picks = getattr(out, "picks", out)
        print(f"\n  样本#{i}  真值: P@{p_true}(={p_true/sr:.1f}s)  S@{s_true}(={s_true/sr:.1f}s)")
        print(f"          模型拾取到 {len(picks)} 个震相:")
        for p in list(picks):
            ptype = getattr(p, "phase", "?")
            pk = getattr(p, "peak_time", None)
            # peak_time 是 UTCDateTime，减去 starttime(0) 得秒
            sec = float(pk - UTCDateTime(0)) if pk is not None else float("nan")
            samp = int(round(sec * sr))
            conf = getattr(p, "peak_value", None)
            truth = p_true if str(ptype).upper().startswith("P") else s_true
            err_samp = samp - truth
            print(f"      {ptype}: 采样点≈{samp} ({sec:.2f}s) "
                  f"| 与真值差 {err_samp:+d}点({err_samp/sr:+.2f}s) | conf={conf}")

    print("\n=======================================================")
    print("基线链路跑通：环境 + GPU + 预训练模型 + 推理 全部 OK")
    print("注：合成波形只为验证链路可用；到时精度需用真实数据评估。")
    print("官方数据到位后，把合成波形换成官方波形，用本地评分脚本打分。")
    print("=======================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr)
        sys.exit(1)
