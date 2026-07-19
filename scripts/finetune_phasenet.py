#!/usr/bin/env python3
"""PhaseNet 微调训练脚本（自包含，可断点续训，微调前后自动对比打分）.

===== 这个脚本做什么（写给非 AI 背景的队友）=====
1. 加载 SeisBench 的 PhaseNet 预训练权重（stead）作为起点；
2. 在训练数据上继续训练（微调），让它适应我们的数据；
3. 微调【前】先打一次分，微调【后】再打一次分，直接看 P/S 精度有没有提升；
4. 每个 epoch 存 checkpoint，机器关机/断网后重跑同一条命令即可【断点续训】。

===== 数据来源可切换（关键设计）=====
- 默认用【合成数据】：现在就能跑通整条训练逻辑，不必等 DiTing/官方数据；
- DiTing / 官方数据到位后：把 --data 换成 hdf5 路径，脚本自动读真实数据，
  训练与评分逻辑【一行不用改】。DiTing 子集格式 = 我们 diting_seisbench.py 的输出
  （group "data" 下每条 dataset，attrs 含 p_sample_100hz / s_sample_100hz）。

===== 为什么评分逻辑内嵌 =====
GPU 机器上没有 phasepicker 包，所以把已测过 20/20 的评分规则原样抄进来，
保证本地分和官方口径一致（P<=0.1s 满分,1.0s 零分; S<=0.2s 满分,2.0s 零分;
数量误差>5% 每个扣 0.5）。
"""
from __future__ import annotations
import argparse, json, os, sys, math, time
import numpy as np

# ============ 内嵌评分（与已测 scorer 一字不差）============
_PHASE = {"P": (0.1, 1.0), "S": (0.2, 2.0)}

def phase_time_score(residual_s, phase_type):
    full, zero = _PHASE[phase_type]
    r = abs(residual_s)
    if r <= full: return 1.0
    if r >= zero: return 0.0
    return (zero - r) / (zero - full)

def match_phases(pred_times, true_times, phase_type):
    _, zero = _PHASE[phase_type]
    cand = []
    for i, pt in enumerate(pred_times):
        for j, tt in enumerate(true_times):
            r = abs(pt - tt)
            if r < zero: cand.append((r, i, j))
    cand.sort(key=lambda x: x[0])
    up, ut, matched = set(), set(), []
    for r, i, j in cand:
        if i in up or j in ut: continue
        up.add(i); ut.add(j); matched.append((i, j, r))
    return matched

def count_error_penalty(n_pred, n_true):
    if n_true == 0: return 0.5 * n_pred
    allowed = 0.05 * n_true
    diff = abs(n_pred - n_true)
    if diff <= allowed: return 0.0
    return 0.5 * int(math.ceil(diff - allowed))

def score_file(pred, truth):
    def sp(items, t): return [tt for pt, tt in items if pt == t]
    pp, ps = sp(pred, "P"), sp(pred, "S")
    tp, ts = sp(truth, "P"), sp(truth, "S")
    mp = match_phases(pp, tp, "P"); ms = match_phases(ps, ts, "S")
    p_sc = sum(phase_time_score(r, "P") for _, _, r in mp)
    s_sc = sum(phase_time_score(r, "S") for _, _, r in ms)
    pen = count_error_penalty(len(pp)+len(ps), len(tp)+len(ts))
    total = max(0.0, p_sc + s_sc - pen)
    return dict(total=total, p_sc=p_sc, s_sc=s_sc, pen=pen,
                pres=[r for _,_,r in mp], sres=[r for _,_,r in ms])

# ============ 合成数据（训练/评分用，与 closed_loop 同套路）============
def synth_window(n, sr, p_sample, s_sample, seed):
    rng = np.random.RandomState(seed)
    z = rng.normal(0, 0.02, n); ns = rng.normal(0, 0.02, n); e = rng.normal(0, 0.02, n)
    tp = np.arange(0, 400)
    pw = np.exp(-tp/120.0)*np.sin(2*np.pi*8.0*tp/sr)
    if p_sample >= 0:
        z[p_sample:p_sample+len(tp)] += 1.0*pw
        ns[p_sample:p_sample+len(tp)] += 0.3*pw
        e[p_sample:p_sample+len(tp)] += 0.3*pw
    ts = np.arange(0, 600)
    sw = np.exp(-ts/200.0)*np.sin(2*np.pi*3.5*ts/sr)
    if s_sample >= 0:
        ns[s_sample:s_sample+len(ts)] += 1.6*sw
        e[s_sample:s_sample+len(ts)] += 1.6*sw
        z[s_sample:s_sample+len(ts)] += 0.4*sw
    return np.vstack([z, ns, e]).astype("float32")

def normalize(x):
    x = x - x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-6
    return (x / s).astype("float32")

def gaussian(n, center, sigma):
    t = np.arange(n)
    return np.exp(-0.5*((t-center)/sigma)**2).astype("float32")

def make_soft_label(n, p_sample, s_sample, label_order, sigma_p=20, sigma_s=30):
    """按模型 label 顺序生成软标签 (C, n)。

    PhaseNet 的 forward 在当前 SeisBench 版本里已经输出 softmax 概率，因此训练目标
    也必须是每个采样点三通道和为 1 的概率分布。旧写法在 P/S 有重叠尾巴时可能让
    P+S+N > 1；这里统一做一次逐点归一化，保证 loss 口径稳定。
    """
    P = gaussian(n, p_sample, sigma_p) if p_sample >= 0 else np.zeros(n, "float32")
    S = gaussian(n, s_sample, sigma_s) if s_sample >= 0 else np.zeros(n, "float32")
    N = np.clip(1.0 - P - S, 0, 1).astype("float32")
    chans = []
    for lab in label_order:
        u = str(lab).upper()
        chans.append(P if u.startswith("P") else S if u.startswith("S") else N)
    y = np.vstack(chans).astype("float32")
    y /= np.maximum(y.sum(axis=0, keepdims=True), 1e-6)
    return y

def phasenet_log_probs(out):
    """把 PhaseNet forward 输出统一转成 log-probabilities。

    SeisBench PhaseNet(stead) 的 forward 通常已经是 softmax 概率；少数版本或未来模型
    可能返回 logits。这里用通道和是否接近 1 来判断，避免对概率再 softmax 一次。
    """
    import torch

    with torch.no_grad():
        o = out.detach()
        channel_sum = o.sum(dim=1)
        is_prob = (
            torch.isfinite(o).all()
            and float(o.min()) >= -1e-5
            and float(o.max()) <= 1.0 + 1e-5
            and torch.allclose(
                channel_sum,
                torch.ones_like(channel_sum),
                rtol=1e-3,
                atol=1e-3,
            )
        )
    if is_prob:
        return torch.log(out.clamp_min(1e-7))
    return torch.log_softmax(out, dim=1)

def set_safe_finetune_mode(model, update_bn=False):
    """小样本微调时冻结 BatchNorm/Dropout 的训练态。

    这次崩溃最像 BN running stats 被 40 条高度相似的合成数据冲坏：训练 loss 看似正常，
    但 eval/classify 使用被污染的 running stats 后 P/S 峰消失。卷积参数仍然会训练；
    只是 BN 用预训练统计量、Dropout 关闭，让训练前向和推理前向保持一致。
    """
    import torch

    model.train()
    if update_bn:
        return

    bn_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()
            for p in module.parameters(recurse=False):
                p.requires_grad = False
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()

def save_checkpoint(path, model, opt, epoch, loss, best_score, args, extra=None):
    import torch

    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "epoch": epoch,
        "loss": loss,
        "best_score": best_score,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

# ============ 数据集构造 ============
def build_synth_dataset(n_samples, win, sr, seed0=0):
    """随机造一批训练窗口。返回 [(wave(3,win), p, s), ...]。"""
    rng = np.random.RandomState(seed0)
    items = []
    for k in range(n_samples):
        p = int(rng.randint(int(win*0.15), int(win*0.45)))
        s = int(p + rng.randint(int(win*0.15), int(win*0.35)))
        s = min(s, win-650)
        wave = synth_window(win, sr, p, s, seed=seed0+k+1)
        items.append((normalize(wave), p, s))
    return items

def load_hdf5_dataset(path, win):
    """读 diting_seisbench.py 产出的子集。数据到位后走这条,逻辑不变。"""
    import h5py
    items = []
    with h5py.File(path, "r") as f:
        grp = f["data"]
        for key in grp:
            d = grp[key]
            wave = np.asarray(d, dtype="float32")
            if wave.shape[0] > wave.shape[1]:
                wave = wave.T
            p = int(d.attrs.get("p_sample_100hz", -1))
            s = int(d.attrs.get("s_sample_100hz", -1))
            # 裁/补到统一窗口 win,并同步平移到时
            wave, p, s = _fit_window(wave, p, s, win)
            items.append((normalize(wave), p, s))
    return items

def _fit_window(wave, p, s, win):
    c, n = wave.shape
    if n == win:
        return wave, p, s
    if n > win:
        # 以 P 为中心裁一段,尽量把 P/S 都框进来
        center = p if p >= 0 else n//2
        start = int(np.clip(center - win//3, 0, max(0, n-win)))
        wave = wave[:, start:start+win]
        p = p-start if p >= 0 else -1
        s = s-start if s >= 0 else -1
        if p < 0 or p >= win: p = -1
        if s < 0 or s >= win: s = -1
        return wave, p, s
    out = np.zeros((c, win), dtype="float32"); out[:, :n] = wave
    return out, p, s

# ============ 评分(用 model.classify,前后对比同一套测试集) ============
def eval_score(model, sr, device):
    import torch
    from obspy import Stream, Trace, UTCDateTime
    cases = [(1500,2800,101),(2000,3500,102),(1000,2200,103),(2500,4200,104),(1800,3100,105)]
    t0 = UTCDateTime(0)
    reports = []
    model.eval()
    for (ps_, ss_, sd) in cases:
        arr = synth_window(6000, sr, ps_, ss_, seed=sd)
        st = Stream()
        for ch, name in zip(arr, ["Z","N","E"]):
            tr = Trace(data=ch); tr.stats.sampling_rate = sr
            tr.stats.starttime = t0; tr.stats.channel = "HH"+name; tr.stats.station = "SYN"
            st.append(tr)
        out = model.classify(st)
        picks = getattr(out, "picks", out)
        pred = []
        for p in list(picks):
            pk = getattr(p, "peak_time", None)
            sec = float(pk - t0) if pk is not None else float("nan")
            ptype = str(getattr(p, "phase", "?")).upper()
            if ptype in ("P","S") and not math.isnan(sec):
                pred.append((ptype, sec))
        truth = [("P", ps_/sr), ("S", ss_/sr)]
        reports.append(score_file(pred, truth))
    n = len(reports)
    tot = sum(r["total"] for r in reports)
    allp = [x for r in reports for x in r["pres"]]
    alls = [x for r in reports for x in r["sres"]]
    p_hit = np.mean([1.0 if x <= 0.1 else 0.0 for x in allp]) if allp else 0.0
    s_hit = np.mean([1.0 if x <= 0.2 else 0.0 for x in alls]) if alls else 0.0
    return dict(mean_score=tot/n,
                p_res=float(np.mean(allp)) if allp else float("nan"),
                s_res=float(np.mean(alls)) if alls else float("nan"),
                p_hit=p_hit, s_hit=s_hit, n=n)

def print_score(tag, sc):
    print("[%s] 平均分=%.4f/2.0 | P残差=%.3fs(满分率%.0f%%) | S残差=%.3fs(满分率%.0f%%)" % (
        tag, sc["mean_score"], sc["p_res"], sc["p_hit"]*100, sc["s_res"], sc["s_hit"]*100))

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(description="PhaseNet 微调(可断点续训,前后对比)")
    ap.add_argument("--data", default="synth", help="'synth' 或 DiTing子集 hdf5 路径")
    ap.add_argument("--out", default="/data/coding/dizheng/runs/ft1", help="产物目录(checkpoint等)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16, help="8G显存建议8~16,炸显存就调小")
    ap.add_argument("--lr", type=float, default=3e-5, help="微调用小学习率,别破坏预训练特征")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="小样本 sanity check 默认不做权重衰减")
    ap.add_argument("--phase-weight", type=float, default=5.0, help="P/S loss 权重；原 30 对小样本过猛")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值；<=0 表示关闭")
    ap.add_argument("--update-bn", action="store_true", help="允许更新 BatchNorm running stats（默认冻结，防小样本冲坏）")
    ap.add_argument("--score-every", type=int, default=1, help="每多少 epoch 跑一次合成评分并更新 best；<=0 关闭")
    ap.add_argument("--n_synth", type=int, default=400, help="合成模式:造多少训练窗口")
    ap.add_argument("--win", type=int, default=3001, help="训练窗口长度(PhaseNet默认3001)")
    ap.add_argument("--sr", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="从上次 checkpoint 断点续训")
    args = ap.parse_args()

    import torch
    import seisbench.models as sbm
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==== 加载预训练 PhaseNet(stead) ====")
    model = sbm.PhaseNet.from_pretrained("stead").to(device)
    label_order = list(getattr(model, "labels", ["P","S","N"]))
    print("设备=%s | 模型输出通道顺序=%s" % (device, label_order))

    # ---- 微调前基线分 ----
    print("\n==== 微调【前】基线评分 ====")
    before = eval_score(model, args.sr, device)
    print_score("微调前", before)

    # ---- 训练数据 ----
    print("\n==== 构造训练数据 (%s) ====" % args.data)
    if args.data == "synth":
        raw = build_synth_dataset(args.n_synth, args.win, args.sr, seed0=args.seed)
    else:
        raw = load_hdf5_dataset(args.data, args.win)
    print("训练样本数: %d" % len(raw))
    X = np.stack([w for w, _, _ in raw])
    Y = np.stack([make_soft_label(args.win, p, s, label_order) for _, p, s in raw])
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # 小样本微调的第一原则：别让 BN/Dropout 的训练态和 classify/eval 推理态错位。
    # requires_grad 会在这里前设置好，因此 optimizer 不会更新被冻结的 BN affine 参数。
    set_safe_finetune_mode(model, update_bn=args.update_bn)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- 断点续训 ----
    ckpt_last = os.path.join(args.out, "last.pt")
    ckpt_best = os.path.join(args.out, "best.pt")
    start_epoch = 0
    best_score = float("-inf")
    if args.resume and os.path.exists(ckpt_last):
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        if state.get("opt") is not None:
            try:
                opt.load_state_dict(state["opt"])
            except ValueError as exc:
                print("[断点续训] optimizer 状态与当前冻结策略不兼容，已只加载模型权重：%r" % exc)
        start_epoch = state["epoch"]
        best_score = float(state.get("best_score", best_score))
        print("[断点续训] 从 epoch %d 继续，历史 best=%.4f" % (start_epoch, best_score))

    # baseline 同时作为 best 守门员：微调一旦把模型训坏，最终会自动回滚到 best。
    if before["mean_score"] >= best_score:
        best_score = before["mean_score"]
        save_checkpoint(
            ckpt_best, model, opt, start_epoch, loss=None, best_score=best_score,
            args=args, extra={"score": before, "tag": "baseline_or_resume"},
        )

    # ---- 训练循环 ----
    print("\n==== 开始微调: epochs=%d batch=%d lr=%g ====" % (args.epochs, args.batch, args.lr))
    print("     BN/Dropout: %s | phase_weight=%.2f | weight_decay=%g | grad_clip=%g" % (
        "允许更新" if args.update_bn else "冻结为推理态",
        args.phase_weight,
        args.weight_decay,
        args.grad_clip,
    ))
    # 类别权重: P/S 稀疏，但原先 30 倍在小样本上容易过猛。默认 5 倍更保守，
    # 合成 sanity check 的目标是“不破坏预训练峰”，不是强行学出一个新分布。
    w = []
    for lab in label_order:
        u = str(lab).upper()
        w.append(1.0 if u.startswith("N") else args.phase_weight)
    class_w = torch.tensor(w, dtype=torch.float32, device=device).view(1, -1, 1)  # (1,C,1)
    nB = int(math.ceil(len(raw) / args.batch))
    for ep in range(start_epoch, args.epochs):
        set_safe_finetune_mode(model, update_bn=args.update_bn)
        perm = torch.randperm(len(raw))
        ep_loss = 0.0
        for b in range(nB):
            idx = perm[b*args.batch:(b+1)*args.batch]
            xb = X[idx].to(device); yb = Y[idx].to(device)
            out = model(xb)
            logp = phasenet_log_probs(out)
            loss = -(class_w * yb * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ep_loss += float(loss)
        ep_loss /= nB

        score = None
        if args.score_every > 0 and ((ep + 1) % args.score_every == 0 or ep + 1 == args.epochs):
            score = eval_score(model, args.sr, device)
            msg = " | score=%.4f/2.0" % score["mean_score"]
            if score["mean_score"] >= best_score:
                best_score = score["mean_score"]
                save_checkpoint(
                    ckpt_best, model, opt, ep + 1, ep_loss, best_score, args,
                    extra={"score": score, "tag": "best"},
                )
                msg += " (刷新 best)"
        else:
            msg = ""

        save_checkpoint(ckpt_last, model, opt, ep + 1, ep_loss, best_score, args, extra={"score": score})
        with open(os.path.join(args.out, "progress.json"), "w") as f:
            json.dump({"epoch": ep+1, "loss": ep_loss, "best_score": best_score, "score": score}, f)
        print("  epoch %2d/%d  loss=%.5f%s  (已存 last.pt/best.pt)" % (
            ep+1, args.epochs, ep_loss, msg
        ), flush=True)

    # ---- 微调后评分 + 对比 ----
    if os.path.exists(ckpt_best):
        state = torch.load(ckpt_best, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print("\n==== 微调【后】评分（使用 best checkpoint，best=%.4f）====" % float(state.get("best_score", best_score)))
    else:
        print("\n==== 微调【后】评分 ====")
    after = eval_score(model, args.sr, device)
    print_score("微调后", after)

    print("\n==== 对比 ====")
    print_score("微调前", before)
    print_score("微调后", after)
    d = after["mean_score"] - before["mean_score"]
    tag = "(提升↑)" if d > 0 else "(未提升)"
    print("平均分变化: %+.4f  %s" % (d, tag))
    print("\n权重已存: %s/best.pt  (小文件, 记得 push 回 Gitee 保住成果)" % args.out)
    print("提示: 合成数据上预训练模型本就近满分, 关键看'微调后不崩';")
    print("      真正的提升要在 DiTing/官方【真实数据】上才看得出来。")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback; traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr); sys.exit(1)
