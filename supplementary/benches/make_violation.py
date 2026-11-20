"""Violation bench on the UNHCR-shaped synthetic base (benches/make_unhcr_like.py). Local only; synthetic data.

Switches (all optional; realised stats go to <out>/violation_log.json):
  --given/--pred/--branch  make_messy.py mechanisms (imported; thin adapters, see apply_messy)
  --tail   rare violations: for ~20% of gated PREDICT items pick a big GIVEN stratum where the item is never asked and flip a
           LogUniform(0.1%,3%) share of that stratum (concentrated in one level of another GIVEN column) to asked.
  --ghost  wave-2 breakoff rows (2%, stratified by country, Egypt doubled) + 0.8% full ghosts.
usage: python benches/make_violation.py --seed 0 --out viol_s0 --given --pred --branch --tail --ghost
"""
import argparse, json, os, sys, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_messy as mm  # imports make_proxy (ACS) only for GV / helper functions

GV = "NA_GATED"
SB = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))


def agreements(df, items):
    out = {}
    for k, r in items.items():
        if r["class"] == "PREDICT" and r.get("gate"):
            g = r["gate"]
            out[k] = float(((df[k] != GV).to_numpy() == df[g["parent"]].isin(g["observed_if"]).to_numpy()).mean())
    return out


def stats(ag):
    v = np.array(list(ag.values()))
    return dict(n=len(v), median=float(np.median(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)), min=float(v.min()))


def apply_messy(df, items, rng, a, log):
    """Adapters: mm.* assume ACS only through the module-level GV (same sentinel) and GIVEN/PREDICT classes, so they run as is.
    UNHCR has a single GIVEN-parent gated item, so --given uses share=1.0 to make sure it is hit."""
    ml = []
    if a.given:
        mm.violate_given(df, items, rng, 1.0, ml, 0.85, 0.95)
    if a.pred:
        mm.degrade_pred(df, items, rng, ml)
    if a.branch:
        mm.branch_block(df, items, rng, a.branch_items, 0.12, ml)
    log["messy"] = [list(map(str, l)) for l in ml]


def tail(df0, df, items, rng, log, frac=0.2, min_rows=1000):
    given = [k for k, r in items.items() if r["class"] == "GIVEN" and k != "panel_hh_key"]
    gated = [k for k, r in items.items() if r["class"] == "PREDICT" and r.get("gate")]
    vis = df0[df0.role.isin(["TRAIN", "DEV"])]
    cand = {}
    for thr in (min_rows, 300):
        for k in gated:
            opts = []
            for g in given:
                vc = vis[g].value_counts()
                for lv, c in vc.items():
                    if c >= thr and (vis.loc[vis[g] == lv, k] == GV).all():
                        opts.append((g, lv, int(c)))
            if opts:
                cand[k] = opts
        if cand:
            used = thr
            break
    log["tail_min_rows_used"] = used if cand else None
    log["tail_candidate_items"] = len(cand)
    if not cand:
        return
    keys = sorted(cand)
    nsel = max(1, int(round(frac * len(gated))))
    sel = list(rng.choice(keys, size=min(nsel, len(keys)), replace=False))
    entries = []
    for k in sel:
        g, lv, _ = cand[k][rng.integers(len(cand[k]))]
        S = (df[g] == lv).to_numpy()
        nS = int(S.sum())
        pi = float(np.exp(rng.uniform(np.log(0.001), np.log(0.03))))
        target = pi * nS
        others = [c for c in given if c != g]
        random_order = rng.permutation(len(others))
        sub = None
        for oi in random_order:
            c2 = others[oi]
            lvls = [l for l in df.loc[S, c2].unique() if int(((df[c2] == l).to_numpy() & S).sum()) >= max(target * 1.5, 20)]
            if lvls:
                l2 = lvls[rng.integers(len(lvls))]
                sub = (c2, l2)
                break
        if sub is None:
            continue
        sub_mask = S & (df[sub[0]] == sub[1]).to_numpy()
        q = min(1.0, target / sub_mask.sum())
        flip = sub_mask & (rng.random(len(df)) < q) & (df[k] == GV).to_numpy()
        pool = df.loc[df[k] != GV, k].to_numpy()
        df.loc[flip, k] = pool[rng.integers(len(pool), size=int(flip.sum()))]
        roles = df.loc[flip, "role"].value_counts().to_dict()
        entries.append(dict(item=k, stratum=[g, str(lv)], stratum_rows=nS, sub=[sub[0], str(sub[1])], sub_rows=int(sub_mask.sum()),
                            pi_target=round(pi, 5), pi_realised=round(float(flip.sum()) / nS, 5), n_flipped=int(flip.sum()), flipped_by_role=roles))
    log["tail"] = entries


def ghost(df, items, rng, log, share=0.02, full=0.008):
    pred = [k for k, r in items.items() if r["class"] == "PREDICT"]
    gated = {k for k in pred if items[k].get("gate")}
    w2 = (df["wave"] == "2").to_numpy()
    cty = df["s1_2_country_asylum_lbl"]
    n2 = int(w2.sum())
    wt = np.where(cty == "Egypt", 2.0, 1.0)
    # per-country rate r_c so that overall wave-2 share == share with Egypt doubled
    base = share * n2 / wt[w2].sum()
    br_rows, taken = [], np.zeros(len(df), bool)
    per_country = {}
    for c in sorted(cty.unique()):
        idx = np.where(w2 & (cty == c).to_numpy())[0]
        m = int(round(base * (2.0 if c == "Egypt" else 1.0) * len(idx)))
        pick = rng.choice(idx, size=m, replace=False)
        taken[pick] = True; br_rows.append(pick); per_country[c] = dict(wave2_rows=len(idx), breakoff=m)
    br = np.concatenate(br_rows)
    rest = np.where(w2 & ~taken)[0]
    fg = rng.choice(rest, size=int(round(full * n2)), replace=False)
    mode = {k: df.loc[df[k] != GV, k].mode().iloc[0] for k in pred if k not in gated}
    J = len(pred)
    pos = {k: i for i, k in enumerate(pred)}
    ks = np.concatenate([rng.integers(1, J, size=len(br)), np.zeros(len(fg), int)])  # k = number of items answered before break
    rows = np.concatenate([br, fg])
    for k in pred:
        sel = rows[ks <= pos[k]]
        df.loc[df.index[sel], k] = GV if k in gated else mode[k]
    log["ghost"] = dict(wave2_rows=n2, breakoff_rows=len(br), full_ghost_rows=len(fg),
                        breakoff_share_wave2=round(len(br) / n2, 5), full_ghost_share_wave2=round(len(fg) / n2, 5),
                        total_share_wave2=round((len(br) + len(fg)) / n2, 5), per_country=per_country,
                        roles=df.loc[df.index[rows], "role"].value_counts().to_dict())
    log["ghost_rows"] = rows.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    for f in ("given", "pred", "branch", "tail", "ghost"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--branch-items", type=int, default=9)
    ap.add_argument("--base", default=None, help="base sandbox name (default: generated to _violbase_<out> at --seed)")
    a = ap.parse_args()
    rng = np.random.default_rng(5000 + a.seed)
    base = a.base
    tmp = None
    if base is None:
        base = tmp = f"_violbase_{a.out}"
        subprocess.check_call([sys.executable, "-I", os.path.join(HERE, "make_unhcr_like.py"), "--seed", str(a.seed), "--out", base])
    df0 = pd.read_parquet(os.path.join(SB, base, "respondents.parquet"))
    schema = json.load(open(os.path.join(SB, base, "schema.json")))
    items = schema["items"]
    df = df0.copy()
    log = dict(seed=a.seed, switches={f: getattr(a, f) for f in ("given", "pred", "branch", "tail", "ghost")})
    log["agreement_base"] = stats(agreements(df0, items))
    apply_messy(df, items, rng, a, log)          # may edit items[k]["gate"] (branch)
    log["agreement_after_messy"] = stats(agreements(df, items))
    if a.tail:
        tail(df0, df, items, rng, log)
    if a.ghost:
        ghost(df, items, rng, log)
    ag = agreements(df, items)
    log["agreement_final"] = stats(ag)
    log["agreement_final_per_item"] = {k: round(v, 4) for k, v in ag.items()}
    schema["dataset"]["description"] = "UNHCR-like synthetic with violations: " + ",".join(f for f, v in log["switches"].items() if v)
    out = os.path.join(SB, a.out)
    os.makedirs(out, exist_ok=True)
    assert (df.role.values == df0.role.values).all() and len(df) == len(df0)
    df.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    json.dump(schema, open(os.path.join(out, "schema.json"), "w"), ensure_ascii=False)
    json.dump(log, open(os.path.join(out, "violation_log.json"), "w"), indent=1)
    if tmp:
        import shutil
        shutil.rmtree(os.path.join(SB, tmp))
    print("wrote", out, {k: log[k] for k in ("agreement_base", "agreement_after_messy", "agreement_final")})
    if "tail" in log:
        print("tail items", len(log["tail"]), [ (e["item"], e["pi_realised"]) for e in log["tail"]][:8])
    if "ghost" in log:
        print({k: v for k, v in log["ghost"].items() if "share" in k})


if __name__ == "__main__":
    main()
