"""Real-signal proxy benchmark from public ACS PUMS (Rhode Island persons, 10,401 rows, no registration needed).

Same shape as the competition: a few demographic GIVEN items, ~100 categorical PREDICT items, real skip patterns
(NaN = not asked) turned into gates with a parent found from the data, roles assigned by row order (TRAIN / DEV / TEST).
Writes $SB_SANDBOX_ROOT/<out>/{schema.json,respondents.parquet} so benches/harness.py can run it.

usage: python benches/make_proxy.py [--seed 0] [--out proxy_ri] [--shuffle 1]
"""
import argparse, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("SB_ACS_CSV", os.path.join(ROOT, "proxy", "raw", "psam_p44.csv"))   # ACS 2022 1-year PUMS, Rhode Island persons
SANDBOX = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))   # where benches are read and written
GIVEN = ["AGEP", "SEX", "RAC1P", "HISP", "NATIVITY", "CIT", "MAR", "PUMA", "SCHL", "RELSHIPP"]
DROP = {"SERIALNO", "SPORDER", "ST", "PWGTP", "ADJINC", "RT", "DIVISION", "REGION", "STATE"}
N_PRED = 100
GV = "NA_GATED"


def band(s, edges, labels=None):
    out = pd.cut(s, edges, right=False, labels=labels)
    return out.astype(object).where(out.notna(), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="proxy_ri")
    ap.add_argument("--shuffle", type=int, default=1)
    ap.add_argument("--wave", type=int, default=0, help="add a 2-level GIVEN 'wave' item; this many PREDICT items carry the gated sentinel on wave-1 rows only (round-coding test)")
    ap.add_argument("--sortby", default="", help="keep file order sorted by this raw column instead of shuffling (order-structure test)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = pd.read_csv(RAW, low_memory=False)
    if a.sortby:
        d = d.sort_values(a.sortby, kind="stable").reset_index(drop=True)
    elif a.shuffle:
        d = d.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    cols = {}
    # binned numerics
    cols["AGEP"] = band(d.AGEP, [0, 5, 15, 18, 25, 35, 45, 55, 65, 75, 200], ["0-4", "5-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"])
    for c, edges in [("WKHP", [0, 10, 20, 35, 40, 45, 60, 200]), ("JWMNP", [0, 10, 20, 30, 45, 60, 200]),
                     ("WAGP", [0, 1, 10000, 25000, 40000, 60000, 90000, 150000, 1e9]),
                     ("PINCP", [-1e9, 1, 10000, 25000, 40000, 60000, 90000, 150000, 1e9]),
                     ("INTP", [-1e9, 1, 500, 5000, 1e9]), ("RETP", [0, 1, 15000, 30000, 1e9]),
                     ("SSP", [0, 1, 15000, 1e9]), ("SEMP", [-1e9, 1, 20000, 1e9]), ("OIP", [0, 1, 1e9]), ("PAP", [0, 1, 1e9])]:
        if c in d:
            cols[c] = band(d[c], edges).map(lambda v: f"b{v.left:g}" if isinstance(v, pd.Interval) else np.nan)
    for c in d.columns:
        if c in cols or c in DROP or c.startswith(("F", "PWGTP", "WGTP", "RAC")) and c != "RAC1P":
            continue
        if c.startswith("F") and c != "FER":
            continue
        s = d[c]
        k = s.nunique(dropna=True)
        if 2 <= k <= 30:
            cols[c] = s.map(lambda v: np.nan if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))
    cols = {k: v.astype(object) for k, v in cols.items()}
    # drop near-constant items
    keep = {}
    for k, v in cols.items():
        vc = v.value_counts(normalize=True, dropna=False)
        if vc.iloc[0] < 0.995 and v.notna().sum() > 200:
            keep[k] = v
    cols = keep
    given = [g for g in GIVEN if g in cols and cols[g].isna().sum() == 0]
    full = [c for c in cols if c not in given and cols[c].isna().sum() == 0]
    gated = [c for c in cols if c not in given and 0.01 < cols[c].isna().mean() < 0.99]
    print(f"given {len(given)}, ungated candidates {len(full)}, gated candidates {len(gated)}")
    # choose PREDICT: all gated candidates that find a parent + fill with ungated ones
    parents_pool = given + full
    gates = {}
    for c in gated:
        asked = cols[c].notna().to_numpy()
        best = None
        for p in parents_pool:
            pv = cols[p]
            tab = pd.crosstab(pv, asked)
            if True not in tab.columns:
                continue
            frac = tab[True] / tab.sum(axis=1)
            obs = list(frac.index[frac > 0.5])
            if not obs or len(obs) == len(frac):
                continue
            agree = (pv.isin(obs).to_numpy() == asked).mean()
            if p in given and agree < 0.999:        # GIVEN-parent gates are deterministic in the real data: keep them exact
                continue
            if best is None or agree > best[0]:
                best = (agree, p, obs)
        if best and best[0] >= 0.9:
            gates[c] = (best[1], best[2], best[0])
    print(f"gated items with a parent (agreement >= 0.9): {len(gates)} of {len(gated)}")
    pred_gated = list(gates)
    rest = [c for c in full if c not in given]
    rng.shuffle(rest)
    pred = pred_gated + rest[:max(N_PRED - len(pred_gated), 0)]
    pred = pred[:max(N_PRED, len(pred_gated))]
    # parents that are fully observed but not PREDICT/GIVEN must be present as items: make them PREDICT
    for c in list(gates):
        p = gates[c][0]
        if p not in given and p not in pred:
            pred.append(p)
    items = {}
    n = len(d)
    wave = rng.integers(0, 2, n) if a.wave else None
    wave_items = [c for c in pred if c not in gates and c not in given][:a.wave] if a.wave else []
    for g in given:
        vals = sorted(cols[g].dropna().unique().tolist())
        items[g] = {"question": g, "class": "GIVEN", "values": vals, "gate": None}
    if a.wave:
        items["wave"] = {"question": "wave", "class": "GIVEN", "values": ["1", "2"], "gate": None}
    for c in pred:
        vals = sorted(cols[c].dropna().unique().tolist())
        gate = None
        if c in wave_items:
            par = next(x for x in pred if x not in wave_items and x not in gates)
            gate = {"parent": par, "observed_if": [str(sorted(cols[par].dropna().unique().tolist())[0])]}
        if c in gates:
            gate = {"parent": gates[c][0], "observed_if": [str(v) for v in gates[c][1]]}
        items[c] = {"question": c, "class": "PREDICT", "values": vals, "gate": gate}
    split = {"n_train": int(n * 0.80), "n_dev": int(n * 0.06)}
    split["n_test"] = n - split["n_train"] - split["n_dev"]
    schema = {"dataset": {"n_rows": n, "version": "proxy", "description": "ACS 2022 PUMS Rhode Island persons (proxy)"},
              "items": items, "split": split, "gated_value": GV}
    out = os.path.join(SANDBOX, a.out)
    os.makedirs(out, exist_ok=True)
    resp = pd.DataFrame({"respondent_id": [f"r{i}" for i in range(n)]})
    resp["role"] = ["TRAIN"] * split["n_train"] + ["DEV"] * split["n_dev"] + ["TEST"] * split["n_test"]
    for k, rec in items.items():
        if k == "wave":
            resp[k] = np.where(wave == 1, "1", "2")
            continue
        v = cols[k].copy()
        if k in wave_items:
            v = v.where(wave == 1, GV)
            resp[k] = v
            continue
        if rec["gate"]:
            v = v.where(v.notna(), GV)
        resp[k] = v
    # a PREDICT cell can be NaN only if ungated and missing; fill with the modal level to keep the file complete
    for k, rec in items.items():
        if k in wave_items or k == "wave":
            continue
        if resp[k].isna().any():
            resp[k] = resp[k].fillna(resp[k].mode().iloc[0])
    resp.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    json.dump(schema, open(os.path.join(out, "schema.json"), "w"), ensure_ascii=False)
    npred = sum(r["class"] == "PREDICT" for r in items.values())
    ng = sum(1 for r in items.values() if r["class"] == "PREDICT" and r["gate"])
    ngp = sum(1 for r in items.values() if r["class"] == "PREDICT" and r["gate"] and r["gate"]["parent"] not in given)
    print(f"wrote {out}: {n} rows, {len(given)} GIVEN, {npred} PREDICT ({ng} gated, {ngp} with PREDICT parent), split {split}")


if __name__ == "__main__":
    main()
