"""Ordinal benchmark: world_bank sandbox with every banded PREDICT item regenerated from a latent continuous score.

Latent traits are random smooth functions of the GIVEN columns plus hidden noise; each ordinal item's score is a loading on
1-2 traits plus item noise, cut into the item's declared bands at the quantiles of that item's sandbox marginal (so marginals
are preserved; 'Not answered' / special cells stay where they were). Adjacent bands are therefore genuinely neighbours.
usage: python benches/make_ordinal.py --seed 0 --out wb_ordinal
"""
import argparse, json, os, sys, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SANDBOX = os.environ.get("SB_SANDBOX_ROOT", os.path.join(ROOT, "_sandbox"))   # where benches are read and written
sys.path.insert(0, HERE)
from ord_detect import ordinal_block

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--src", default="world_bank")
ap.add_argument("--out", default="wb_ordinal")
ap.add_argument("--given-weight", type=float, default=0.8)
a = ap.parse_args()
rng = np.random.default_rng(500 + a.seed)
src = os.path.join(SANDBOX, a.src)
df = pd.read_parquet(os.path.join(src, "respondents.parquet"))
schema = json.load(open(os.path.join(src, "schema.json")))
items = schema["items"]
given = [k for k, r in items.items() if r["class"] == "GIVEN"]
n = len(df)
# trait scores: sum of random level effects of the GIVEN columns + hidden noise
NT = 5
T = np.zeros((n, NT))
for g in given:
    codes = pd.factorize(df[g])[0]
    eff = rng.normal(size=(codes.max() + 1, NT))
    T += eff[codes] / np.sqrt(len(given))
T = a.given_weight * T / T.std(0) + np.sqrt(1 - a.given_weight ** 2) * rng.normal(size=(n, NT))
log = []
for k, r in items.items():
    if r["class"] != "PREDICT" or not r["values"]:
        continue
    blk = ordinal_block(r["values"])
    if blk is None:
        continue
    vals = r["values"]
    levels = [vals[i] for i in blk]
    inblk = df[k].isin(levels).to_numpy()
    if inblk.sum() < 50:
        continue
    cnt = df.loc[inblk, k].value_counts().reindex(levels).fillna(0).to_numpy()
    load = np.zeros(NT)
    sel = rng.choice(NT, size=2, replace=False)
    load[sel] = rng.normal(size=2) * [1.0, 0.5]
    z = T @ load / np.linalg.norm(load) + 0.7 * rng.normal(size=n)
    zi = z[inblk]
    order = np.argsort(zi, kind="stable")
    bands = np.repeat(np.arange(len(levels)), cnt.astype(int))
    bands = bands[: len(order)] if len(bands) >= len(order) else np.concatenate([bands, np.full(len(order) - len(bands), len(levels) - 1)])
    newv = np.empty(len(order), dtype=object)
    newv[order] = np.array(levels, dtype=object)[bands]
    col = df[k].astype(object).copy()
    col[inblk] = newv
    df[k] = col
    log.append((k, len(levels)))
out = os.path.join(SANDBOX, a.out)
os.makedirs(out, exist_ok=True)
df.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
schema["dataset"]["description"] = "world_bank schema with banded items regenerated from latent traits"
json.dump(schema, open(os.path.join(out, "schema.json"), "w"), ensure_ascii=False)
print("regenerated", len(log), "ordinal items ->", out)
