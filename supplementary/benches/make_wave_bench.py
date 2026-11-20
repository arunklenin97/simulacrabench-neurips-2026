"""Wave x country interaction bench for the wave-aware structure (v45).

Takes an existing unhcr-shaped sandbox (default unhcr_like) and, for ~20% of the PREDICT items, makes the answer law of
wave-2 respondents depend on country: a wave-2 row in country c redraws its answer with probability q from a country-
specific distribution m_c (wave-1 rows untouched), so the effect is a pure wave x country interaction that a logit
additive in (wave, country) cannot express.  q is calibrated so that the oracle gain (full wave x country cell law vs the
best additive-in-logit fit, in-sample on all visible rows) averages --gain nats per PREDICT cell over ALL cells.
--control applies the same redraws to rows of BOTH waves (country effect only, zero interaction).

usage: python benches/make_wave_bench.py --base unhcr_like --out wave_ix [--control] [--gain 0.005] [--seed 0]
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))   # where benches are read and written
CTRY = "s1_2_country_asylum_lbl"


def redraw(col, GV, wave, ctry, q, ms, both, rng):
    out = col.copy()
    vals = np.array(sorted(set(col) - {GV}, key=str), dtype=object)
    ok = col != GV
    for c, m in ms.items():
        sel = ok & (ctry == c) & (True if both else (wave == "2"))
        sel = np.flatnonzero(sel)
        hit = sel[rng.random(len(sel)) < q]
        out[hit] = rng.choice(vals, size=len(hit), p=m)
    return out


def oracle_gain(col, GV, wave, ctry, ncells):
    ok = col != GV
    y = pd.factorize(col[ok])[0]
    if len(np.unique(y)) < 2:
        return 0.0
    w, c = wave[ok], ctry[ok]
    ctries = sorted(set(c))
    A = np.column_stack([(w == "2")] + [(c == k) for k in ctries[1:]]).astype(float)
    B = np.column_stack([A] + [((w == "2") & (c == k)) for k in ctries[1:]]).astype(float)
    ll = []
    for X in (A, B):
        lr = LogisticRegression(C=1e4, max_iter=300).fit(X, y)
        p = lr.predict_proba(X)
        ll.append(np.log(np.clip(p[np.arange(len(y)), np.searchsorted(lr.classes_, y)], 1e-9, 1)).sum())
    return (ll[1] - ll[0]) / ncells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unhcr_like")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--gain", type=float, default=0.005)
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sb = SANDBOX
    df = pd.read_parquet(os.path.join(sb, a.base, "respondents.parquet"))
    sch = json.load(open(os.path.join(sb, a.base, "schema.json")))
    GV = sch.get("gated_value", "NA_GATED")
    rec = sch["items"]
    pred = [n for n, r in rec.items() if r["class"] == "PREDICT" and len(r["values"]) >= 2]
    rng = np.random.default_rng(a.seed)
    chosen = list(rng.choice(pred, size=int(a.share * len(pred)), replace=False))
    npred = sum(1 for r in rec.values() if r["class"] == "PREDICT")
    vis = df.role != "TEST"
    wave, ctry = df.wave.to_numpy().astype(str), df[CTRY].to_numpy().astype(str)
    ctries = sorted(set(ctry))
    ms = {}
    for j in chosen:
        k = len(set(df[j]) - {GV})
        ms[j] = {c: rng.dirichlet(np.ones(k) * 0.5) for c in ctries}
    seeds = {j: int(rng.integers(1 << 31)) for j in chosen}

    def build(q, both):
        new = {}
        for j in chosen:
            new[j] = redraw(df[j].to_numpy(object), GV, wave, ctry, q, ms[j], both, np.random.default_rng(seeds[j]))
        return new

    # calibrate q on the interaction design (the control reuses the same q)
    best = None
    for q in np.linspace(0.8, 1.0, 9):
        new = build(q, False)
        g = sum(oracle_gain(new[j][vis.to_numpy()], GV, wave[vis.to_numpy()], ctry[vis.to_numpy()], vis.sum() * npred) * 1.0
                for j in chosen)
        print(f"q={q:.2f} oracle gain {g:.5f} nats/cell")
        if best is None or abs(g - a.gain) < abs(best[1] - a.gain):
            best = (q, g)
    q = best[0]
    new = build(q, a.control)
    out = df.copy()
    for j in chosen:
        out[j] = new[j]
    gi = sum(oracle_gain(out[j].to_numpy(object)[vis.to_numpy()], GV, wave[vis.to_numpy()], ctry[vis.to_numpy()], vis.sum() * npred) for j in chosen)
    print(f"chosen q={q:.2f}; {len(chosen)} items; interaction oracle gain {gi:.5f} nats/cell ({'control' if a.control else 'interaction'})")
    d = os.path.join(sb, a.out)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "respondents.parquet"), index=False)
    json.dump(sch, open(os.path.join(d, "schema.json"), "w"))


if __name__ == "__main__":
    main()
