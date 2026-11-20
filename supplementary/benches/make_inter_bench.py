"""Pairwise-interaction bench for the explicit interaction columns (v60; UNICEF and World Bank).

Takes a sandbox (unicef / world_bank) and, for ~--share of the PREDICT items, makes the answer law depend on the PAIR of
GIVEN columns (a, b): a row in cell (a=i, b=j) redraws its answer with probability q from a cell-specific Dirichlet(0.5)
law, so the effect is a pure a x b interaction that a model additive in (a, b) cannot express.  q is calibrated so the
oracle gain (cell law vs best additive-in-logit fit, in-sample on all visible rows) averages --gain nats per PREDICT cell
over ALL cells.  --control draws from a law that depends on a alone (same q, same items): zero interaction.

usage: python benches/make_inter_bench.py --base unicef --a Country --b Q05 --out inter_unicef [--control] [--gain 0.005]
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))   # where benches are read and written


def onehot(a, lv):
    return np.stack([(a == k) for k in lv], 1).astype(float)


def oracle_gain(col, GV, A, B, ncells):
    ok = col != GV
    y = pd.factorize(col[ok])[0]
    if len(np.unique(y)) < 2:
        return 0.0
    a, b = A[ok], B[ok]
    la, lb = sorted(set(a))[1:], sorted(set(b))[1:]
    add = np.hstack([onehot(a, la), onehot(b, lb)])
    cell = np.array([x + "|" + z for x, z in zip(a, b)])
    cl = [k for k in sorted(set(cell)) if (cell == k).sum() >= 5]
    full = np.hstack([add, onehot(cell, cl)])
    ll = []
    for X in (add, full):
        lr = LogisticRegression(C=1e4, max_iter=200).fit(X, y)
        p = lr.predict_proba(X)
        ll.append(np.log(np.clip(p[np.arange(len(y)), np.searchsorted(lr.classes_, y)], 1e-9, 1)).sum())
    return (ll[1] - ll[0]) / ncells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--gain", type=float, default=0.005)
    ap.add_argument("--share", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sb = SANDBOX
    df = pd.read_parquet(os.path.join(sb, a.base, "respondents.parquet"))
    sch = json.load(open(os.path.join(sb, a.base, "schema.json")))
    GV = sch.get("gated_value", "NA_GATED")
    rec = sch["items"]
    pred = [n for n, r in rec.items() if r["class"] == "PREDICT" and len(r["values"]) >= 2]
    rng = np.random.default_rng(a.seed)
    chosen = list(rng.choice(pred, size=max(1, int(a.share * len(pred))), replace=False))
    npred = sum(1 for r in rec.values() if r["class"] == "PREDICT")
    vis = (df.role != "TEST").to_numpy()
    A, B = df[a.a].to_numpy().astype(str), df[a.b].to_numpy().astype(str)
    cell = np.array([x + "|" + z for x, z in zip(A, B)])
    cells, lva = sorted(set(cell)), sorted(set(A))
    laws = {}
    for j in chosen:
        k = len(set(df[j]) - {GV})
        laws[j] = ({c: rng.dirichlet(np.ones(k) * 0.5) for c in cells}, {c: rng.dirichlet(np.ones(k) * 0.5) for c in lva})
    seeds = {j: int(rng.integers(1 << 31)) for j in chosen}

    def build(q, control):
        new = {}
        for j in chosen:
            col = df[j].to_numpy(object)
            out = col.copy()
            vals = np.array(sorted(set(col) - {GV}, key=str), dtype=object)
            r = np.random.default_rng(seeds[j])
            ok = np.flatnonzero(col != GV)
            hit = ok[r.random(len(ok)) < q]
            for i in hit:
                m = laws[j][1][A[i]] if control else laws[j][0][cell[i]]
                out[i] = r.choice(vals, p=m)
            new[j] = out
        return new

    best = None
    for q in (0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9):
        new = build(q, False)
        g = sum(oracle_gain(new[j][vis], GV, A[vis], B[vis], vis.sum() * npred) for j in chosen)
        print(f"q={q:.2f} oracle gain {g:.5f} nats/cell", flush=True)
        if best is None or abs(g - a.gain) < abs(best[1] - a.gain):
            best = (q, g)
    q = best[0]
    new = build(q, a.control)
    out = df.copy()
    for j in chosen:
        out[j] = new[j]
    gi = sum(oracle_gain(out[j].to_numpy(object)[vis], GV, A[vis], B[vis], vis.sum() * npred) for j in chosen)
    print(f"chosen q={q:.2f}; {len(chosen)} items; interaction oracle gain {gi:.5f} nats/cell ({'control' if a.control else 'interaction'})")
    d = os.path.join(sb, a.out)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "respondents.parquet"), index=False)
    json.dump(sch, open(os.path.join(d, "schema.json"), "w"))


if __name__ == "__main__":
    main()
