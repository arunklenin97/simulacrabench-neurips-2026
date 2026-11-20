"""Synthetic stand-in that reuses the real instrument schema and plants the structures the model has to find:
latent respondent types driven by the GIVEN block, household-composition (sum-type) routing, round-specific gated slots,
and noisy PREDICT-parent gates. Signal strength is arbitrary; use it to check mechanisms and no-harm, never as a score.

usage: python benches/make_unhcr_like.py [--seed 0] [--out unhcr_like] [--types 10]
"""
import argparse, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))   # where benches are read and written
DATA = os.environ.get("SB_DATA_DIR", os.path.join(ROOT, "data"))   # the organisers' public schemas (data/*.json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="unhcr_like")
    ap.add_argument("--types", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sch = json.load(open(os.path.join(DATA, "unhcr.json")))
    items, GV = sch["items"], sch["gated_value"]
    n = sch["dataset"]["n_rows"]
    given = [k for k, r in items.items() if r["class"] == "GIVEN"]
    pred = [k for k, r in items.items() if r["class"] == "PREDICT"]
    cols, code = {}, {}
    for g in given:
        K = len(items[g]["values"])
        p = rng.dirichlet(np.ones(K) * 2.0)
        code[g] = rng.choice(K, size=n, p=p)
        cols[g] = np.array(items[g]["values"], dtype=object)[code[g]]
    wave = code["wave"]
    cnt = [g for g in given if g.startswith("s2_3_")]
    sch_kids = (code["s2_3_boys_6_17"] + code["s2_3_girls_6_17"]) > 0
    size = sum(code[g] for g in cnt)
    # latent type depends on GIVEN through a random linear map
    F = np.hstack([np.eye(len(items[g]["values"]))[code[g]] for g in given] + [np.minimum(size, 12)[:, None] / 6.0])
    W = rng.normal(0, 0.9, (F.shape[1], a.types))
    z_logit = F @ W
    zp = np.exp(z_logit - z_logit.max(1, keepdims=True)); zp /= zp.sum(1, keepdims=True)
    z = np.array([rng.choice(a.types, p=p_) for p_ in zp])
    out, done = {}, {}

    def gen(j):
        if j in done:
            return done[j]
        r = items[j]
        K = len(r["values"])
        th = rng.dirichlet(np.ones(K) * 0.4, size=a.types)
        v = np.array([rng.choice(K, p=th[t]) for t in z]) if K > 0 else np.zeros(n, int)
        vals = np.array(r["values"], dtype=object)[v]
        g = r.get("gate")
        if g:
            par = g["parent"]
            if par in cols:
                pv = cols[par]
            else:
                gen(par); pv = out[par]
            asked = np.isin(pv, g["observed_if"])
            asked = np.where(rng.random(n) < 0.92, asked, ~asked)          # noisy routing
            rule = rng.random()
            if rule < 0.15:
                asked = asked & sch_kids                                       # sum-type routing off the household composition
            elif rule < 0.3:
                asked = asked & (wave == 1)                                    # round-specific slot
            vals = np.where(asked, vals, GV)
        out[j] = vals
        done[j] = True

    for j in pred:
        gen(j)
    resp = pd.DataFrame({"respondent_id": [f"r{i}" for i in range(n)]})
    sp = sch["split"]
    resp["role"] = ["TRAIN"] * sp["n_train"] + ["DEV"] * sp["n_dev"] + ["TEST"] * sp["n_test"]
    for k in items:
        if k in cols: resp[k] = cols[k]
        elif k in out: resp[k] = out[k]
    d = os.path.join(SANDBOX, a.out)
    os.makedirs(d, exist_ok=True)
    resp.to_parquet(os.path.join(d, "respondents.parquet"), index=False)
    json.dump(sch, open(os.path.join(d, "schema.json"), "w"))
    print("wrote", d, resp.shape)


if __name__ == "__main__":
    main()
