"""基线推理验证脚本（P4 / SeisBench PhaseNet）.

目的：在还没有官方数据、也不训练的情况下，先证明"加载预训练模型 → 喂波形 →
拾取 P/S 到时"这整条链路在这台机器上跑得通。这是零训练成本的第一个可用基线。

做的四件事：
  1) 加载在 STEAD 上预训练好的 PhaseNet（无需训练）；
  2) 列出还有哪些现成权重可选；
  3) 下载一个小的公开数据集样例（ETHZ，体量适中，便于快速验证）；
  4) 对若干条真实波形跑 annotate + classify，打印拾取到的 P/S 到时。

跑通它 = 环境 + 模型 + 数据 + 推理 全链路通。之后官方数据一到，
只要把数据换成官方波形即可。
"""

import sys


def main() -> None:
    import seisbench.models as sbm
    import seisbench.data as sbd

    print("==================== [1] 加载预训练 PhaseNet ====================")
    model = sbm.PhaseNet.from_pretrained("stead")
    model.eval()
    print("已加载 PhaseNet(stead) 权重")

    print("\n==================== [2] 可选的其他预训练权重 ====================")
    try:
        print("PhaseNet 可用权重:", sbm.PhaseNet.list_pretrained())
    except Exception as exc:  # noqa: BLE001
        print("列出权重失败（不影响主流程）:", repr(exc))

    print("\n==================== [3] 下载公开数据集样例(ETHZ) ====================")
    # ETHZ 体量适中，适合快速跑通链路；首次运行会自动下载到 SEISBENCH_CACHE_ROOT
    data = sbd.ETHZ(sampling_rate=100)  # 统一到 100Hz，与官方数据一致
    print(data)
    print("样本总数:", len(data))

    print("\n==================== [4] 对前几条波形跑基线推理 ====================")
    n_demo = min(3, len(data))
    for i in range(n_demo):
        wf = data.get_waveforms(i)  # (channels, samples)
        # SeisBench 需要 obspy Stream 或 numpy；这里用模型的 annotate 接口
        import numpy as np
        arr = np.asarray(wf, dtype="float32")
        # 用 classify 直接拿离散 pick（内部会做窗口化与概率峰值提取）
        from obspy import Stream, Trace
        st = Stream()
        for c in range(min(3, arr.shape[0])):
            tr = Trace(data=arr[c])
            tr.stats.sampling_rate = 100.0
            tr.stats.channel = ["ENZ"[c]] if False else ["Z", "N", "E"][c]
            st.append(tr)
        out = model.classify(st)
        picks = getattr(out, "picks", out)
        print(f"  样本#{i}: 拾取到 {len(picks)} 个震相")
        for p in list(picks)[:6]:
            ptype = getattr(p, "phase", "?")
            peak = getattr(p, "peak_time", getattr(p, "start_time", "?"))
            conf = getattr(p, "peak_value", None)
            print(f"      {ptype} @ {peak}  conf={conf}")

    print("\n=======================================================")
    print("基线链路跑通：环境 + 预训练模型 + 公开数据 + 推理 全部 OK")
    print("下一步：官方数据到位后，把波形换成官方文件，用本地评分脚本打分。")
    print("=======================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print("\n[出错]", repr(exc), file=sys.stderr)
        print("常见原因：数据集下载中断/磁盘不足/网络问题。可重跑，已下载部分会复用缓存。", file=sys.stderr)
        sys.exit(1)
