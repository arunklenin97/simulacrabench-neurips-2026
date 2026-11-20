"""Messy-routing benchmark: ACS proxy (benches/make_proxy.py, unchanged) + documented real-data routing messiness.

Switchable, applied on top of a freshly generated clean proxy (same rows, split and roles):
  --given   GIVEN-parent gates stop being exact. For a random share of GIVEN-parent gated items the gate agrees with who
            answered only a ~U(0.85,0.95) share of rows. Mechanism: the declared parent is a proxy for the true routing
            variable, so most violations are skipped-side rows in a hidden "alt" region of another GIVEN column that DO
            answer (the school item asked of girls-only households); the rest are asked rows carrying the sentinel.
  --pred    PREDICT-parent gates are degraded: per item a target agreement is drawn (median ~0.83, range 0.3-0.98) and
            random rows are flipped (answered<->sentinel) until it is reached.
  --branch  a branch block: N items declared gated on one PREDICT parent P (3 branches, each with the P-values where that
            branch is mostly asked), but the true routing follows a different PREDICT item Q (3 groups); each branch also
            leaks into the others (overlap), so declared agreement ends up well below Q's. (UNHCR s3_11 vs s3_12.)
usage: python benches/make_messy.py --seed 0 --out messy_all --given --pred --branch
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_proxy as mp
ROOT = mp.ROOT
SANDBOX = mp.SANDBOX
GV = mp.GV


def agreement(col, schema_gate, df):
    asked = (df[col] != GV).to_numpy()
    pred_asked = df[schema_gate["parent"]].isin(schema_gate["observed_if"]).to_numpy()
    return float((asked == pred_asked).mean())


def violate_given(df, items, rng, share, log, lo=0.85, hi=0.95):
    given = [k for k, r in items.items() if r["class"] == "GIVEN"]
    gated = [k for k, r in items.items() if r["class"] == "PREDICT" and r["gate"] and r["gate"]["parent"] in given]
    chosen = [k for k in gated if rng.random() < share]
    n = len(df)
    for k in chosen:
        g = items[k]["gate"]
        a = rng.uniform(lo, hi)
        par = df[g["parent"]].isin(g["observed_if"]).to_numpy()
        asked = (df[k] != GV).to_numpy()
        skipped_idx = np.where(~asked)[0]
        asked_idx = np.where(asked)[0]
        if len(skipped_idx) < 20 or len(asked_idx) < 20:
            continue
        V = int(round((1 - a) * n))
        Vs = min(int(0.7 * V), len(skipped_idx) // 2)
        Va = min(V - Vs, len(asked_idx) // 4)
        # hidden alt region: a random value subset of another GIVEN column (not the parent)
        other = [c for c in given if c != g["parent"] and df[c].nunique() >= 2]
        alt_c = other[rng.integers(len(other))]
        vals = df[alt_c].unique()
        alt_vals = rng.choice(vals, size=max(1, len(vals) // 2), replace=False)
        alt = df[alt_c].isin(alt_vals).to_numpy()
        w = np.where(alt[skipped_idx], 1.0, 0.15)
        pick = rng.choice(skipped_idx, size=Vs, replace=False, p=w / w.sum())
        base_vals = df.loc[asked_idx, k].to_numpy()
        for r in pick:   # answer drawn from asked rows with the same alt-region membership (so it is learnable)
            pool = base_vals[alt[asked_idx] == alt[r]]
            if len(pool) < 20:
                pool = base_vals
            df.at[r, k] = pool[rng.integers(len(pool))]
        if Va:
            pick_a = rng.choice(asked_idx, size=Va, replace=False)
            df.loc[pick_a, k] = GV
        ag = agreement(k, g, df)
        log.append(("given", k, g["parent"], round(a, 3), round(ag, 3)))


def degrade_pred(df, items, rng, log):
    given = [k for k, r in items.items() if r["class"] == "GIVEN"]
    gated = [k for k, r in items.items() if r["class"] == "PREDICT" and r["gate"] and r["gate"]["parent"] not in given]
    n = len(df)
    for k in gated:
        g = items[k]["gate"]
        a0 = agreement(k, g, df)
        # median 0.83, range [0.3, 0.98]
        a = float(np.clip(rng.normal(0.83, 0.12), 0.3, 0.98))
        if rng.random() < 0.1:
            a = float(rng.uniform(0.3, 0.6))
        if a >= a0:
            log.append(("pred", k, g["parent"], round(a, 3), round(a0, 3)))
            continue
        par = df[g["parent"]].isin(g["observed_if"]).to_numpy()
        asked = (df[k] != GV).to_numpy()
        agree_idx = np.where(par == asked)[0]
        nflip = int(round((a0 - a) * n))
        flip = rng.choice(agree_idx, size=min(nflip, len(agree_idx)), replace=False)
        ask_pool = df.loc[asked, k].to_numpy()
        for r in flip:
            if asked[r]:
                df.at[r, k] = GV
            else:
                df.at[r, k] = ask_pool[rng.integers(len(ask_pool))]
        log.append(("pred", k, g["parent"], round(a, 3), round(agreement(k, g, df), 3)))


def mutual_info(x, y):
    t = pd.crosstab(x, y).to_numpy().astype(float)
    p = t / t.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log(p[m] / (px @ py)[m])).sum())


def branch_block(df, items, rng, n_items, overlap, log):
    pred_ungated = [k for k, r in items.items() if r["class"] == "PREDICT" and not r["gate"]]
    cand = [k for k in pred_ungated if 3 <= df[k].nunique() <= 12]
    P = None
    # P: PREDICT parent with >= 3 levels; Q: its best associated other PREDICT item (the "true" routing variable)
    best = None
    for p in cand:
        for q in cand:
            if p == q:
                continue
            mi = mutual_info(df[p], df[q])
            if best is None or mi > best[0]:
                best = (mi, p, q)
    _, P, Q = best
    # Q -> 3 groups by value frequency rank; the true branch variable follows Q (small noise)
    qv = df[Q].value_counts().index.tolist()
    grp = {v: i % 3 for i, v in enumerate(qv)}
    n = len(df)
    B = df[Q].map(grp).to_numpy().copy()
    noise = rng.random(n) < 0.1
    B[noise] = rng.integers(0, 3, noise.sum())
    # declared branch sets on P: P-values where each true branch dominates (as make_proxy would infer)
    ct = pd.crosstab(df[P], B)
    owner = ct.to_numpy().argmax(1)
    sets = {b: [v for v, o in zip(ct.index, owner) if o == b] for b in range(3)}
    for b in range(3):
        if not sets[b]:   # keep every branch declarable
            sets[b] = [ct.index[b % len(ct.index)]]
    gate_parents = {r["gate"]["parent"] for r in items.values() if r.get("gate")}
    rest = [k for k in pred_ungated if k not in (P, Q) and k not in gate_parents]
    rng.shuffle(rest)
    pick = rest[:n_items]
    for i, k in enumerate(pick):
        b = i % 3
        asked = (B == b)
        leak = rng.random(n) < overlap          # overlapping branches: also asked when another branch is the true one
        asked = asked | leak
        df[k] = df[k].where(asked, GV)
        items[k]["gate"] = {"parent": P, "observed_if": [str(v) for v in sets[b]]}
        log.append(("branch", k, P, f"true_via={Q}", round(agreement(k, items[k]["gate"], df), 3),
                    f"Q-route agree={round(float((((df[Q].map(grp) == b).to_numpy()) == (df[k] != GV).to_numpy()).mean()), 3)}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--given", action="store_true")
    ap.add_argument("--pred", action="store_true")
    ap.add_argument("--branch", action="store_true")
    ap.add_argument("--given-share", type=float, default=0.7)
    ap.add_argument("--given-agree-lo", type=float, default=0.85)
    ap.add_argument("--given-agree-hi", type=float, default=0.95)
    ap.add_argument("--branch-items", type=int, default=9)
    ap.add_argument("--overlap", type=float, default=0.12)
    a = ap.parse_args()
    rng = np.random.default_rng(1000 + a.seed)
    base = f"_messybase_{a.out}"
    sys.argv = ["make_proxy", "--seed", str(a.seed), "--out", base]
    mp.main()
    bdir = os.path.join(SANDBOX, base)
    df = pd.read_parquet(os.path.join(bdir, "respondents.parquet"))
    schema = json.load(open(os.path.join(bdir, "schema.json")))
    items = schema["items"]
    log = []
    if a.given:
        violate_given(df, items, rng, a.given_share, log, a.given_agree_lo, a.given_agree_hi)
    if a.pred:
        degrade_pred(df, items, rng, log)
    if a.branch:
        branch_block(df, items, rng, a.branch_items, a.overlap, log)
    out = os.path.join(SANDBOX, a.out)
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    schema["dataset"]["description"] = "ACS proxy with routing messiness: " + ",".join(f for f in ("given", "pred", "branch") if getattr(a, f))
    json.dump(schema, open(os.path.join(out, "schema.json"), "w"), ensure_ascii=False)
    import shutil
    shutil.rmtree(bdir)
    json.dump([list(map(str, l)) for l in log], open(os.path.join(out, "messiness.json"), "w"))
    for l in log:
        print(*l)
    print("wrote", out)


if __name__ == "__main__":
    main()
