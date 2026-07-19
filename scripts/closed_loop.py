"""Self-contained closed loop: synth "fake official data" -> PhaseNet infer -> score.
No dependency on the phasepicker package. Scoring logic is embedded verbatim
from the tested scorer (P: <=0.1s full, 1.0s zero; S: <=0.2s full, 2.0s zero;
count error >5% -> 0.5/each). Official data arrives -> swap the data-loading part.
"""
import sys, math
import numpy as np

# ============ embedded scoring (verbatim from tested scorer) ============
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
    um_p = [i for i in range(len(pred_times)) if i not in up]
    um_t = [j for j in range(len(true_times)) if j not in ut]
    return matched, um_p, um_t

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
    mp, _, _ = match_phases(pp, tp, "P")
    ms, _, _ = match_phases(ps, ts, "S")
    p_sc = sum(phase_time_score(r, "P") for _, _, r in mp)
    s_sc = sum(phase_time_score(r, "S") for _, _, r in ms)
    pen = count_error_penalty(len(pp)+len(ps), len(tp)+len(ts))
    total = max(0.0, p_sc + s_sc - pen)
    pres = [r for _, _, r in mp]; sres = [r for _, _, r in ms]
    return dict(total=total, p_sc=p_sc, s_sc=s_sc, pen=pen,
                np=len(pp), ntp=len(tp), ns=len(ps), nts=len(ts), pres=pres, sres=sres)

# ============ synth "fake official data" ============
def synth_trace(n=6000, sr=100.0, p_sample=1500, s_sample=2800, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.normal(0, 0.02, n); ns = rng.normal(0, 0.02, n); e = rng.normal(0, 0.02, n)
    tp = np.arange(0, 400)
    pw = np.exp(-tp/120.0)*np.sin(2*np.pi*8.0*tp/sr)
    z[p_sample:p_sample+len(tp)] += 1.0*pw; ns[p_sample:p_sample+len(tp)] += 0.3*pw; e[p_sample:p_sample+len(tp)] += 0.3*pw
    ts = np.arange(0, 600)
    sw = np.exp(-ts/200.0)*np.sin(2*np.pi*3.5*ts/sr)
    ns[s_sample:s_sample+len(ts)] += 1.6*sw; e[s_sample:s_sample+len(ts)] += 1.6*sw; z[s_sample:s_sample+len(ts)] += 0.4*sw
    return np.vstack([z, ns, e]).astype("float32"), p_sample, s_sample

def main():
    import torch
    import seisbench.models as sbm
    from obspy import Stream, Trace, UTCDateTime

    print("==== load pretrained PhaseNet ====")
    model = sbm.PhaseNet.from_pretrained("stead")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    print("model ready, device=", device)

    sr = 100.0
    # fake official test set: (p_sample, s_sample, seed)
    cases = [(1500,2800,1),(2000,3500,2),(1000,2200,3),(2500,4200,4),(1800,3100,5)]
    t0 = UTCDateTime(0)
    file_reports = []
    for i, (ps_, ss_, sd) in enumerate(cases):
        arr, p_true, s_true = synth_trace(sr=sr, p_sample=ps_, s_sample=ss_, seed=sd)
        st = Stream()
        for ch, name in zip(arr, ["Z","N","E"]):
            tr = Trace(data=ch); tr.stats.sampling_rate = sr
            tr.stats.starttime = t0; tr.stats.channel = "HH"+name; tr.stats.station = "SYN"
            st.append(tr)
        out = model.classify(st)
        picks = getattr(out, "picks", out)
        # predictions as (type, seconds); truth as (type, seconds)
        pred = []
        for p in list(picks):
            pk = getattr(p, "peak_time", None)
            sec = float(pk - t0) if pk is not None else float("nan")
            ptype = str(getattr(p, "phase", "?")).upper()
            if ptype in ("P","S") and not math.isnan(sec):
                pred.append((ptype, sec))
        truth = [("P", p_true/sr), ("S", s_true/sr)]
        rep = score_file(pred, truth)
        file_reports.append(rep)
        pres = ("%.3f"%rep["pres"][0]) if rep["pres"] else "miss"
        sres = ("%.3f"%rep["sres"][0]) if rep["sres"] else "miss"
        print("file#%d total=%.3f | P=%.2f(res %ss) S=%.2f(res %ss) | pen=%.1f | picks P%d/S%d" % (
            i, rep["total"], rep["p_sc"], pres, rep["s_sc"], sres, rep["pen"], rep["np"], rep["ns"]))

    # aggregate
    n = len(file_reports)
    tot = sum(r["total"] for r in file_reports)
    allp = [x for r in file_reports for x in r["pres"]]
    alls = [x for r in file_reports for x in r["sres"]]
    print("\n==== AGGREGATE over %d files ====" % n)
    print("mean total score = %.4f (max=2.0/file)" % (tot/n))
    if allp: print("P residual: mean=%.3fs median=%.3fs max=%.3fs (n=%d)" % (np.mean(allp), np.median(allp), np.max(allp), len(allp)))
    if alls: print("S residual: mean=%.3fs median=%.3fs max=%.3fs (n=%d)" % (np.mean(alls), np.median(alls), np.max(alls), len(alls)))
    print("\n==== CLOSED LOOP OK: read -> infer -> score ====")
    print("Official data arrives -> replace synth_trace() with real loader; scoring stays identical.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback; traceback.print_exc(); sys.exit(1)
