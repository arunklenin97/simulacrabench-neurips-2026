"""Probabilistic survey completion: predict the hidden answers of held-out respondents from their GIVEN block.

Per instrument:
  1. Features: one-hot of every GIVEN item plus an ordinal position column.
  2. Candidates, each fitted with 5-fold cross-validation on the visible respondents (out-of-fold predictions kept):
       marginal   smoothed crowd shares
       linear     one softmax head per PREDICT item on the features
       mlp        2-layer trunk shared by all items, one softmax head per item
       mlp_big    3x512 trunk, seed-averaged (large instruments)
       lca        (large instruments) latent-class mixture: GIVEN block -> distribution over latent respondent types, every item answered
                  from its type's distribution; fitted on the joint likelihood of all answers plus the per-item one
       mlp_xl     4x1024 trunk, seed-averaged (large instruments)
       mlp_round  trunk on the features crossed with the survey round (when a round item is GIVEN)
       profile    empirical answer counts of respondents with an identical GIVEN profile, shrunk toward the MLP
       cells      hierarchical back-off over informative GIVEN prefixes
       tabpfn     pre-trained in-context tabular model (TabPFN-3.5, bundled weights) on the items with the most
                  remaining loss, until the instrument's deadline
       tabicl     pre-trained in-context tabular model (TabICLv2) on the items TabPFN left, or in its place
       kumo       second, independently pre-trained in-context tabular model (Kumo-Tabular, bundled weights) on items with
                  at most 10 classes, breadth first over the shared folds, in time taken from TabPFN's repeat passes
       gbdt       per-item histogram gradient-boosted trees with native categorical splits (large instruments),
                  fitted on a subset of the folds in order of nonlinearity, under a hard time cap
  3. Per-item blend weights fitted on out-of-fold log score with the grader's flat mixing; candidates that cover only
     some folds are blended on the rows they cover.
  4. Skip logic: a gated item whose parent is visible is determined; for gates whose parent is itself predicted, the
     gate probability is a learned blend of the direct and the composed estimate, accepted only on out-of-fold gain.
  5. Support caps: where a GIVEN level with at least 1000 visible respondents never shows the gated slot of an item (or
     always shows it), the gated slot is capped near 0 (or the substantive answers are), since such a stratum is
     deterministic in the data.

No subprocesses. Runs on CUDA when available, otherwise CPU threads. Any failure on the modelling path falls back to
marginal plus skip logic, so a run always returns valid output.
"""

import os
import re
import sys
import time
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never touch the network; weights come from the pre-fetched HF cache

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Compact bundled weights: a large weight matrix may be stored as int8 values `name` plus one float scale per output row
# under `name::qs` (symmetric per-row quantization). They are restored to float32 when a checkpoint is read, before any
# model is built, so inference runs at full precision. Checkpoints without such keys pass through unchanged.
_QS, _QH = "::qs", "::f16"         # int8 + per-row scales; half-precision copy (marker key) of a float32 tensor


def _dequant(obj):
    if not isinstance(obj, dict):
        return obj
    for k in [k for k in obj if isinstance(k, str) and k.endswith(_QH)]:
        base = k[:-len(_QH)]
        obj.pop(k)
        obj[base] = obj[base].to(torch.float32)
    for k in [k for k in obj if isinstance(k, str) and k.endswith(_QS)]:
        base, s = k[:-len(_QS)], obj.pop(k)
        q = obj[base]
        obj[base] = (q.to(torch.float32).reshape(q.shape[0], -1) * s.to(device=q.device, dtype=torch.float32).reshape(-1, 1)).reshape(q.shape)
    for v in obj.values():
        if isinstance(v, dict):
            _dequant(v)
    return obj


if not getattr(torch.load, "_sb_dequant", False):
    _torch_load_raw = torch.load

    def _torch_load_dequant(*a, **k):
        return _dequant(_torch_load_raw(*a, **k))
    _torch_load_dequant._sb_dequant = True
    torch.load = _torch_load_dequant

T0 = time.time()
BUDGET = float(os.environ.get("SB_BUDGET", 780))   # seconds for all three instruments (limit 900 in dev)
BUDGET_FIXED = "SB_BUDGET" in os.environ
V70HARD = os.environ.get("SB_NO_V70HARD") != "1"   # v70 final hardening: hard-clock guards, 2880 s final budget, safety-net snapshots
FINAL_BUDGET = float(os.environ.get("SB_FINAL_BUDGET", 2880 if V70HARD else 3000))   # seconds for all three instruments in the final phase (limit 3600)
FM_RESERVE_CAP = 300                # final phase: upper bound of the share-based reserve, seconds
N_INSTRUMENTS = 3
VERBOSE = os.environ.get("SB_VERBOSE") == "1"
DEVICE = torch.device(os.environ.get("SB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
torch.set_num_threads(8)

LEGACY = os.environ.get("SB_V15") == "1"   # reproduce the previous stage set and equal per-instrument split
FIXED = os.environ.get("SB_V16") == "1"     # fixed TabPFN folds, no strict check, no final-phase budget
STRICT = os.environ.get("SB_STRICT", "0") == "1" and not FIXED   # a missing or broken TabPFN is an error, not a fallback
_OFF = not LEGACY                           # near-duplicate trunk variants and trees: off unless LEGACY

EPS = 1e-3          # grader's flat-mixing epsilon
ALPHA = 0.5         # marginal smoothing
N_FOLDS = 5
PROFILE_K = (1.0, 3.0, 10.0, 30.0, 100.0)
CELL_ALPHAS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
FOREST_TREES, FOREST_LEAF, FOREST_MAX_ITEMS = 200, 8, 120
GATE_PRIOR, GATE_MIN_GAIN = 20.0, 3.0      # prior strength (pseudo-rows) toward the direct gate; min log-lik gain (nats) to accept
NO_CELLS = os.environ.get("SB_NO_CELLS") == "1"
NO_FOREST = os.environ.get("SB_FOREST") != "1"
NO_GATEHYB = os.environ.get("SB_NO_GATEHYB") == "1"
NO_GHOST = os.environ.get("SB_NO_GHOST") == "1"
NO_TAILCAL = os.environ.get("SB_NO_TAILCAL") == "1"
NO_TEMPER = os.environ.get("SB_NO_TEMPER") == "1"
NO_V68FIX = os.environ.get("SB_NO_V68FIX") == "1"          # v68 audit fix B (unmix fitted on tempered OOF); 1 = v67 behaviour
GHOST_LAMS = (0.25, 0.5, 0.75, 1.0)   # candidate strengths of the hazard mixture
GHOST_RATE_CAP, GHOST_NATS, GHOST_MIN_ROWS, GHOST_PRIOR, GHOST_MIN_GAIN = 0.999, 3 * np.log(100.0), 10, 50.0, 2.0   # asked-rate cap, surprise of an unexplained sentinel run (nats), rows needed, hazard prior (rows), nested gain (nats)
_ghost_last = {}                    # diagnostics of the last ghost component run
NO_ROUTE = os.environ.get("SB_NO_ROUTE") == "1"
NO_GATE_KT = os.environ.get("SB_GATE_KT") != "1"
NO_TABICL = os.environ.get("SB_NO_TABICL") == "1"
NO_TABICL_ADD = os.environ.get("SB_NO_TABICL_ADD") == "1"
TABICL_ADD_TOTAL = float(os.environ.get("SB_TABICL_ADD_TOTAL", 240))   # seconds cap for TabICL next to TabPFN (small instruments)
TABICL_ADD_LINEAR = os.environ.get("SB_TABICL_ADD_LINEAR") == "1"   # test only: linear probe extrapolation
TABICL_ADD_FOLDS = int(os.environ.get("SB_TABICL_ADD_FOLDS", 2))
NO_BIG = os.environ.get("SB_NO_BIG") == "1"
BIG_MIN_ITEMS = 150                 # only the UNHCR-sized instrument (213 items) gets the big trunk
BIG_CFG = (512, 0.3, 1e-4, 1.5e-3, 80, 3)   # hidden, dropout, wd, lr, max epochs, depth
BIG_SEEDS = 2
TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_FOLDS, TABICL_EST, TABICL_MIN_ROWS, TABICL_MAX_K, TABICL_MAX_CTX = 2, 2, 500, 40, 8000
# TabPFN context cap: the whole training fold on a GPU, the TabICL cap elsewhere
PFN_MAX_CTX = int(os.environ.get("SB_PFN_MAX_CTX") or (20000 if DEVICE.type == "cuda" else TABICL_MAX_CTX))
TABICL_TOTAL = float(os.environ.get("SB_TABICL_TOTAL", 200))   # seconds of TabICL across the smaller instruments
NO_TABICL_BIG = os.environ.get("SB_NO_TABICL_BIG") == "1" or _OFF
TABICL_BIG_TOTAL = float(os.environ.get("SB_TABICL_BIG_TOTAL", 260))   # own TabICL budget for a UNHCR-sized instrument
TABICL_BIG_FOLDS = 1                # one context fold per item: twice the item coverage per second
TABICL_BIG_MIN_LOSS = 0.02          # skip items whose remaining OOF loss (nats/cell) leaves nothing to gain
NO_PH = os.environ.get("SB_NO_PH") == "1" or _OFF
PH_CFG = (512, 0.3, 1e-4, 1.5e-3, 3)            # hidden, dropout, wd, lr, depth
PH_GRID = (6, 10, 14, 20, 28, 38, 50)            # epochs at which every head's predictions are snapshotted
PH_SEL = 0.12                                    # share of each training fold held back to choose the epoch per head
NO_CAPS = os.environ.get("SB_NO_SZ") == "1"
NO_XFIT = os.environ.get("SB_NO_XFIT") == "1"          # rules judged on the rows that fit the blend (instead of cross-fitted)
NO_BBFIT = os.environ.get("SB_NO_BBFIT") == "1"        # fixed residuals instead of beta-binomial fitted ones
RES_MIN, RES_MAX = 1e-5, 0.2
CAP_MIN_ROWS, CAP_Q = 1000, 5e-4            # smallest stratum treated as deterministic; residual mass left on the capped side
NO_GBDT = os.environ.get("SB_NO_GBDT") == "1" or _OFF
GBDT_MIN_ITEMS = int(os.environ.get("SB_GBDT_MIN_ITEMS", 150))
GBDT_FOLDS, GBDT_MIN_ROWS, GBDT_BATCH, GBDT_FIXED_ITERS = 3, 500, 8, 60
GBDT_TOTAL = float(os.environ.get("SB_GBDT_TOTAL", 240))       # seconds of tree fitting for the instrument
GBDT_PARAMS = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
NO_LCA = os.environ.get("SB_NO_LCA") == "1" or _OFF
LCA_CLASSES, LCA_HIDDEN, LCA_DROP, LCA_WD, LCA_LR, LCA_EPOCHS, LCA_SEEDS = 40, 256, 0.25, 1e-4, 3e-3, 80, 2
LCA_MARG_W = 1.0                    # weight of the direct per-item likelihood next to the joint mixture likelihood
NO_AGG = os.environ.get("SB_AGG") != "1"
NO_MR = os.environ.get("SB_NO_MR") == "1"   # mask-and-recover pretraining candidate; off reproduces the previous build exactly
NO_INTER2 = os.environ.get("SB_NO_INTER2") == "1"   # explicit interaction columns for linear/mlp (UNICEF, World Bank); off reproduces the previous build exactly
INTER2_MIN_ROWS = 30                # smallest interaction column kept (rarer combinations stay with the main effects)
INTER2_MAX_COLS = 400               # cap on added columns (largest first)
INTER2_GAIN = 0.0002                # nats/cell the linear+mlp average must gain on BOTH nested halves (fold parity)
INTER2_TIME_FRAC = 0.3              # the stage retrains linear and mlp once; skipped unless that costs less than this share of the time left
# interaction pairs per instrument, chosen from the schemas alone: UNICEF religion codes are country-tailored (country x religion),
# plus country x age band and country x religious-community role; World Bank: randomised feedback arm x country and x the
# demographic GIVEN items an arm effect would plausibly differ by (age, education, employment status)
INTER2_SPECS = (
    (("Country", "Q05", "Q01", "Q04"), (("Country", "Q05"), ("Country", "Q01"), ("Country", "Q04"))),
    (("group", "user.country", "user.age", "user.educationLevel", "user.employmentStatus"),
     (("group", "user.country"), ("group", "user.age"), ("group", "user.educationLevel"), ("group", "user.employmentStatus"))),
)
NO_INTER_UNHCR = os.environ.get("SB_NO_INTER_UNHCR") == "1"   # wave-aware GIVEN x GIVEN interaction columns for the UNHCR-sized schema; off reproduces the previous build exactly
INTERU_FAIL = os.environ.get("SB_INTERU_FAIL") == "1"         # test only: raise inside the UNHCR interaction path
_UC, _UW = "s1_2_country_asylum_lbl", "wave"
# plain GIVEN-value combinations (never record keys): wave x country x governorate / origin, country x household and demographic items
INTER2_SPECS_WAVE = (
    (("wave", "s1_2_country_asylum_lbl", "s2_1_hh_relationship_lbl", "s2_1b_age", "s2_1c_education_level_lbl", "s2_1d_marital_status_lbl",
      "s2_2_single_parent_lbl", "governorate_col", "s2_4a_origin_governorate_lbl", "s2_3_boys_0_5", "s2_3_girls_0_5", "s2_3_boys_6_17",
      "s2_3_girls_6_17", "s2_3_men_18_59", "s2_3_women_18_59"),
     ((_UW, _UC), (_UW, "governorate_col"), (_UW, "s2_4a_origin_governorate_lbl"), (_UW, "s2_1b_age"), (_UW, "s2_1c_education_level_lbl"),
      (_UW, "s2_1d_marital_status_lbl"), (_UW, "s2_1_hh_relationship_lbl"),
      (_UC, "s2_4a_origin_governorate_lbl"), (_UC, "s2_1_hh_relationship_lbl"), (_UC, "s2_1b_age"), (_UC, "s2_1c_education_level_lbl"),
      (_UC, "s2_1d_marital_status_lbl"), (_UC, "s2_2_single_parent_lbl"), (_UC, "s2_3_boys_0_5"), (_UC, "s2_3_girls_0_5"),
      (_UC, "s2_3_boys_6_17"), (_UC, "s2_3_girls_6_17"), (_UC, "s2_3_men_18_59"), (_UC, "s2_3_women_18_59"),
      ("s2_1d_marital_status_lbl", "s2_2_single_parent_lbl"), ("s2_1b_age", "s2_1d_marital_status_lbl"),
      ("s2_3_men_18_59", "s2_3_women_18_59"), ("s2_3_boys_6_17", "s2_3_girls_6_17"), ("s2_3_boys_0_5", "s2_3_girls_0_5"))),
)
INTER_UNHCR_GAIN = float(os.environ.get("SB_INTERU_GAIN", 0.001))   # UNHCR path: both nested halves must gain this much (the base INTER2_GAIN is within noise there)
V65_WB_SAFE = os.environ.get("SB_V65_WB_SAFE", "1") != "0"   # default on: the World Bank instrument runs without the v62 distill and v63 mlp_mr stages (v61 behaviour there); SB_V65_WB_SAFE=0 = v64
def _wb_safe(given):
    return V65_WB_SAFE and "group" in given and "user.country" in given
NO_EBCELLS = os.environ.get("SB_NO_EBCELLS") == "1"   # empirical-Bayes back-off `cells` (per-item per-level strength); off = the single-alpha cells candidate
EBCELLS_FAIL = os.environ.get("SB_EBCELLS_FAIL") == "1"   # test only
EB_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
NO_UNMIX = os.environ.get("SB_NO_UNMIX") == "1"       # deconvolve the grader's flat 1e-3 mix from final vectors (UNHCR-sized only, nested-half gated); off = previous
UNMIX_FAIL = os.environ.get("SB_UNMIX_FAIL") == "1"   # test only
UNMIX_ALPHAS = (0.25, 0.5, 0.75, 1.0)
UNMIX_MIN_ITEMS = 150
NO_SIBPOOL = os.environ.get("SB_NO_SIBPOOL") == "1"   # shared-event sibling gate pooling (UNHCR-sized, nested-half gated per group); off = v65
SIBPOOL_FAIL = os.environ.get("SB_SIBPOOL_FAIL") == "1"   # test only
SIBPOOL_MIN_ITEMS = 150
SIBPOOL_AGREE, SIBPOOL_MIN_NATS, SIBPOOL_LAMS = 0.90, 1.0, (0.0, 0.25, 0.5, 0.75, 1.0)
UNMIX_MIN_GAIN = 0.5    # nats on each nested half
NO_DISTILL = os.environ.get("SB_NO_DISTILL") == "1" or LEGACY or FIXED
DISTILL_CFG = (256, 0.25, 1e-4, 2e-3, 60, 2)   # student trunk: the small MLP (hidden, dropout, wd, lr, max epochs, depth)
DISTILL_ALPHA = float(os.environ.get("SB_DISTILL_ALPHA", 0.7))   # weight of the soft (teacher) term on rows that have both targets
DISTILL_SHARE = float(os.environ.get("SB_DISTILL_SHARE", 0.12))   # share of the foundation-model window held back for the student
DISTILL_TEACHER = os.environ.get("SB_DISTILL_TEACHER", "pfn")     # test only: "blend" = teacher is the in-sample-weighted blend of the other candidates
DISTILL_FAIL = os.environ.get("SB_DISTILL_FAIL") == "1"           # test only: raise inside the stage
DISTILL_TRAIN_SOFT = os.environ.get("SB_DISTILL_TRAIN_SOFT") == "1"   # soft teacher targets on the training rows too (their OOF probabilities saw the validation fold's labels)
_distill_hold = {"s": 0.0}
_distill_rec = {"on": False, "d": {}}      # per-fold teacher probabilities on the rows to score (context without that fold)
NO_XL = os.environ.get("SB_NO_XL") == "1" or _OFF
XL_CFG = (1024, 0.35, 1e-4, 1e-3, 100, 3)      # hidden, dropout, wd, lr, max epochs, depth
XL_SEEDS = 2
NO_ROUND = os.environ.get("SB_NO_ROUND") == "1" or _OFF
ROUND_CFG = (384, 0.3, 1e-4, 1.5e-3, 80, 2)
NO_TABPFN = os.environ.get("SB_NO_TABPFN") == "1" or LEGACY
TABPFN_CKPT = os.environ.get("SB_TABPFN_CKPT", "tabpfn-v3.5-fast-20260909.safetensors")
TABPFN_FOLDS = int(os.environ.get("SB_TABPFN_FOLDS", 2))
TABPFN_MAX_FOLDS = int(os.environ.get("SB_TABPFN_MAX_FOLDS", 4))
TABPFN_ADAPTIVE = not FIXED and "SB_TABPFN_FOLDS" not in os.environ   # folds follow the measured call cost
TABPFN_EST = int(os.environ.get("SB_TABPFN_EST", 1))
NO_ANYTIME = os.environ.get("SB_NO_ANYTIME") == "1"
NO_KUMO = os.environ.get("SB_NO_KUMO") == "1" or LEGACY or FIXED
KUMO_SIZE = os.environ.get("SB_KUMO_SIZE", "large")
KUMO_EST = int(os.environ.get("SB_KUMO_EST", 2))
KUMO_SHARE = float(os.environ.get("SB_KUMO_SHARE", 0.5))   # share of an instrument's foundation-model window held back from TabPFN's repeat passes
KUMO_MAX_K = 10                     # the network's class head; items with more classes stay with the other candidates
KUMO_MAX_FOLDS = int(os.environ.get("SB_KUMO_MAX_FOLDS", 4))
KUMO_MIN_FOLDS = 2                  # fewer completed folds would shrink the rows the whole blend is fitted on
KUMO_MIN_ROWS = 500
KUMO_MAX_CTX = int(os.environ.get("SB_KUMO_MAX_CTX", 3000))      # context rows per call: the regime of the large-model probe; attention cost is superlinear in rows
KUMO_MAX_ITEMS = int(os.environ.get("SB_KUMO_MAX_ITEMS", 10))    # only the items with the most remaining MLP OOF loss
KUMO_MIN_ITEMS = 3                  # fewer items than this fitting two folds is not worth the stage
KUMO_SAFETY = 1.5                   # a call starts only if now + predicted time * this is inside the stage and instrument deadlines
_kumo_hold = {"s": 0.0}             # seconds of the TabPFN stage kept free for the Kumo stage
NO_ORDFEAT = os.environ.get("SB_NO_ORDFEAT") == "1"   # in-context learners get the categorical GIVEN columns only
ORD_VARIANT = os.environ.get("SB_ORDVAR", "C")        # A numeric copies, B + household total, C + numbered labels, E rank copies, F numeric replaces
ORD_MLP = os.environ.get("SB_NO_ORDMLP") != "1" and not NO_ORDFEAT   # MLP ordinal position from parsed lower bounds (specials 0.5)
ANYTIME_PFN_PASSES = int(os.environ.get("SB_ANYTIME_PASSES", 4))   # time left in a stage goes to repeat TabPFN passes (new random preprocessing, averaged) up to this many
NO_FULLCTX = os.environ.get("SB_NO_FULLCTX") == "1"   # test rows of the in-context learners also scored with all visible rows as context
FULLCTX_MODE = os.environ.get("SB_FULLCTX_MODE", "full")   # test only: "full" or "mix" (mean of fold average and full-context, in probability space)
FULLCTX_SEEDS = int(os.environ.get("SB_FULLCTX_SEEDS", 2))   # preprocessing seeds averaged in the full-context pass
FULLCTX_FRAC = 0.5                  # share of the time left after the fold passes that the full-context pass may use
_fullctx_dbg = {}                   # test only: fold average and full-context probabilities (SB_FULLCTX_DBG=1)
ANYTIME_PFN_FRAC = 0.65             # share of the time left after the first pass the repeat passes may use when the big trunk wants the rest
ANYTIME_BIG_SEEDS = int(os.environ.get("SB_ANYTIME_SEEDS", 6))   # ... and to further whole seeds of the big trunk, up to this many in the seed average
TABPFN_MIN_LOSS = 0.01              # skip items whose remaining OOF loss (nats/cell) leaves nothing to gain
FM_RESERVE = 45                     # seconds kept after the foundation-model stages for blend, gates and output
FM_RESERVE_FRAC = 0.25              # ... or this share of the instrument's allowance, whichever is larger
PROBE_ROWS = 256                    # context and query rows of the timing probe before the first in-context call
INSTR_J = (12, 76, 213)             # PREDICT item counts of the three instruments, to split the budget by workload
_calls = {"n": 0, "phase": 1}
_snap = {"P": None}                 # blended, gated test probabilities: the fallback if a later stage raises


PROBE = os.environ.get("SB_PROBE") == "1"   # raise when the TabPFN stage of the largest instrument is not fully covered
KSTAGE = os.environ.get("SB_KSTAGE", "off")   # which Kumo failure raises (v58 ladder, v59 default call = raise only if the model loaded but a probe/real call raised, incl. both autocast attempts; never on timing_skip): load | call | time | any | off
KPROBE = os.environ.get("SB_KPROBE", "1") == "1"   # raise when the Kumo stage adds no candidate on an instrument (development check of host timing)
_pfn_cov = {"k_done": 0, "covered": 0, "order": 0}   # folds completed and items covered by the last TabPFN stage
NO_GATEPFN = os.environ.get("SB_NO_GATEPFN") == "1" or NO_TABPFN or FIXED
GATEPFN_SHARE = float(os.environ.get("SB_GATEPFN_SHARE", 0.12))   # share of an instrument's foundation-model window kept for the gate-event model
GATEPFN_MIN_ROWS = 200              # labeled rows an event needs
GATEPFN_MIN_MINOR = 20              # rows of the rarer class an event needs
GATEPFN_AGREE = 0.98                # children of one event must agree on asked / not asked in this share of the rows where two or more are known
GATEPFN_MAX_EVENTS = 24
GATEPFN_MIN_FOLDS = 1               # completed folds needed for the event model's out-of-fold column (rows of the other folds keep the baseline gate)
GATEPFN_MIN_DELTA = 2.0             # nats: gain of the stacker with the event column over the same stacker without it, on the same rows
_gp_hold = {"s": 0.0}               # seconds of the TabPFN stage kept free for the gate-event stage


class _StrictError(RuntimeError):
    """TabPFN is required but unusable; propagates out of predict() instead of falling back."""


def _log(*a):
    if VERBOSE:
        print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- schema helpers

def _layout(frame, schema):
    gv = schema.get("gated_value", "NA_GATED")
    rec = schema["items"]
    items = [n for n, r in rec.items() if r["class"] in ("GIVEN", "PREDICT")]
    opts = {n: list(rec[n]["values"]) + ([gv] if rec[n].get("gate") else []) for n in items}
    idx = {n: {v: i for i, v in enumerate(opts[n])} for n in items}
    given = [n for n in items if rec[n]["class"] == "GIVEN"]
    pred = [n for n in items if rec[n]["class"] == "PREDICT"]
    return rec, items, opts, idx, given, pred


def _features(frame, given, opts, idx, nvals):
    cols = []
    n = len(frame)
    for g in given:
        k = len(opts[g])
        code = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
        oh = np.zeros((n, k), np.float32)
        ok = code >= 0
        oh[np.flatnonzero(ok), code[ok]] = 1.0
        cols.append(oh)
        nv = len(nvals[g])                     # real levels, excluding a gated slot
        if nv > 2:
            # ordinal position (many GIVEN items are ordered bands/counts); harmless for nominal ones
            pos = np.where(ok & (code < nv), code / (nv - 1), 0.5).astype(np.float32)
            lv = _ord_levels(opts[g][:nv]) if ORD_MLP else None
            if lv is not None:
                rk = _ord_rank_map(lv)
                tab = np.array([rk.get(v, 0.5) for v in opts[g]] + [0.5], np.float32)
                pos = tab[np.where(code >= 0, code, len(opts[g]))]
            cols.append(pos[:, None])
    X = np.hstack(cols) if cols else np.zeros((n, 1), np.float32)
    return X


def _inter2_cols(frame, given, opts, idx, vis):
    """Product one-hots of the instrument's interaction pairs (None when the instrument is not UNICEF / World Bank).
    Combinations with fewer than INTER2_MIN_ROWS visible rows are dropped, then the largest INTER2_MAX_COLS kept."""
    if NO_INTER2:
        return None
    if "wave" in given and NO_INTER_UNHCR:
        return None
    gset = set(given)
    pairs = None
    for need, prs in (INTER2_SPECS_WAVE if "wave" in given else INTER2_SPECS):
        if all(g in gset for g in need):
            pairs = prs
            break
    if pairs is None:
        return None
    if INTERU_FAIL and "wave" in given:
        raise RuntimeError("SB_INTERU_FAIL")
    cols, cnts = [], []
    code = {}
    for g in {x for pr in pairs for x in pr}:
        code[g] = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
    for a, b in pairs:
        for i in range(len(opts[a])):
            ca = code[a] == i
            if not ca.any():
                continue
            for j in range(len(opts[b])):
                c = ca & (code[b] == j)
                k = int(c[vis].sum())
                if k >= INTER2_MIN_ROWS:
                    cols.append(c.astype(np.float32))
                    cnts.append(k)
    if not cols:
        return None
    keep = np.argsort(-np.asarray(cnts), kind="stable")[:INTER2_MAX_COLS]
    return np.stack([cols[i] for i in sorted(keep)], 1)


def _mean_ll(Pflat, Y, heads, rows):
    """Mean log score per observed cell (nats/cell) of the rows."""
    tot, cnt = 0.0, 0
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        y = Y[rows, j]
        ok = y >= 0
        if not ok.any():
            continue
        p = _scored(Pflat[rows][ok][:, s].astype(np.float64) + 1e-12, heads.widths[j])
        tot += np.log(p[np.arange(ok.sum()), y[ok]]).sum()
        cnt += ok.sum()
    return tot / max(cnt, 1)


def _inter2_stage(W, Xv, Xh, Y, fold, heads, configs, oof, test, deadline, t_loop, gain_min=INTER2_GAIN):
    """Retrain linear and mlp on [X, W] with the same folds and seeds. Swap them in only if the average OOF gain is at least
    INTER2_GAIN nats/cell in both nested halves (fold parity). Returns True when swapped."""
    if time.time() + 1.3 * t_loop > time.time() + INTER2_TIME_FRAC * (deadline - time.time()):
        _log("inter2 skipped (time)")
        return False
    Xi, Xhi = np.hstack([Xv, W[0]]), np.hstack([Xh, W[1]])
    o2 = {c: np.zeros_like(oof[c]) for c in ("linear", "mlp")}
    t2 = {c: np.zeros_like(test[c]) for c in ("linear", "mlp")}
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        for c in ("linear", "mlp"):
            h, d, wd, lr, ep = configs[c]
            net = _train(Xi[tr], Y[tr], Xi[va], Y[va], heads, h, d, wd, lr, ep, deadline, seed=f)
            o2[c][va] = _predict_flat(net, Xi[va], heads)
            t2[c] += _predict_flat(net, Xhi, heads) / N_FOLDS
        if time.time() > deadline:
            _log("inter2 aborted (deadline)")
            return False
    gains = []
    for half in (0, 1):
        rows = np.flatnonzero(fold % 2 == half)
        gains.append(np.mean([_mean_ll(o2[c], Y, heads, rows) - _mean_ll(oof[c], Y, heads, rows) for c in ("linear", "mlp")]))
    ok = min(gains) >= gain_min and all(np.isfinite(o2[c]).all() and np.isfinite(t2[c]).all() for c in o2)
    _log(f"inter2 {W[0].shape[1]} cols, nested gains {gains[0]:.5f} / {gains[1]:.5f} nats/cell -> {'USED' if ok else 'off'}")
    if ok:
        for c in o2:
            oof[c], test[c] = o2[c], t2[c]
    return ok


def _lower_bound(v):
    m = re.match(r"^(\d+)", str(v))
    return int(m.group(1)) if m else None


_ORD_NUM = re.compile(r"^\s*(\d+)(?=\s*$|\s*[-+,]|\s+(?:to|and|or|years?)\b)")
_ORD_NUMBERED = re.compile(r"^\s*(\d+)\.\s")


def _ord_levels(values, numbered=False):
    """Order from the labels alone: label -> lower bound for the levels whose label starts with a number (a band,
    a count, a numbered scale step when numbered=True); None unless at least 3 levels parse."""
    out = {}
    for v in values:
        sv = str(v)
        m = _ORD_NUM.match(sv) or (_ORD_NUMBERED.match(sv) if numbered else None)
        if m:
            out[v] = float(m.group(1))
    return out if len(out) >= 3 else None


def _ord_rank_map(lv):
    """Level -> rank in [0, 1] among the parsed levels, by lower bound."""
    u = sorted(set(lv.values()))
    r = {b: i / (len(u) - 1) for i, b in enumerate(u)} if len(u) > 1 else {u[0]: 0.5}
    return {v: r[b] for v, b in lv.items()}


def _ord_frame(frame, given, opts, rec=None):
    """Feature frame for the in-context learners: a categorical column per GIVEN item, plus (where the labels are
    ordered) a float column of the lower bound, so the learner can use the order its category encoding discards.
    Levels that do not parse are NaN in the float copy. Any failure returns the categorical-only frame."""
    cat = {g: pd.Categorical(frame[g], categories=opts[g]) for g in given}
    if NO_ORDFEAT:
        return pd.DataFrame(cat)
    try:
        var = ORD_VARIANT
        numbered = var in ("C", "F", "G")
        cols, num_cols, replaced = dict(cat), {}, set()
        for g in given:
            lv = _ord_levels(opts[g], numbered=numbered)
            if lv is None:
                continue
            mp = _ord_rank_map(lv) if var in ("E", "G") else lv
            col = frame[g].map(mp).astype(float).to_numpy()
            num_cols[g] = col
            if var == "F":
                replaced.add(g)
        out = {g: (num_cols[g] if g in replaced else cols[g]) for g in given}
        for g, col in num_cols.items():
            if g not in replaced:
                out[g + "__ord"] = col
        if var in ("B", "C", "G"):
            cnt = [g for g in given if len(opts[g]) >= 3 and all(re.match(r"^\d+(\+|-\d+)?$", str(v)) for v in opts[g])]
            if len(cnt) >= 4:
                out["__hh_total"] = sum(np.nan_to_num(frame[g].map({v: _lower_bound(v) for v in opts[g]}).astype(float).to_numpy())
                                        for g in cnt)
        return pd.DataFrame(out)
    except Exception as e:
        _log("ordfeat failed:", repr(e))
        return pd.DataFrame(cat)


def _agg_features(frame, given, rec):
    """Household-composition aggregates from banded count items (several GIVEN counts that sum to a household size):
    totals and any/none indicators per demographic group, as one-hots, so sum-type routing is a single feature."""
    cnt = [g for g in given if len(rec[g]["values"]) >= 3 and all(re.match(r"^\d+(\+|-\d+)?$", str(v)) for v in rec[g]["values"])]
    if len(cnt) < 4:
        return None
    val = {g: frame[g].map({v: _lower_bound(v) for v in rec[g]["values"]}).fillna(0).to_numpy(np.float32) for g in cnt}
    groups = {"all": cnt}
    for name, keys in (("child", ("boys", "girls")), ("u6", ("_0_5",)), ("school", ("_6_17",)), ("adult", ("18_59",)),
                       ("old", ("60",)), ("female", ("girls", "women")), ("male", ("boys", "men"))):
        mem = [g for g in cnt if any(k in g.lower() for k in keys)]
        if len(mem) >= 2 or (name in ("u6", "school", "adult", "old") and mem):
            groups[name] = mem
    cols, tot = [], {}
    for name, mem in groups.items():
        t = sum(val[g] for g in mem)
        tot[name] = t
        top = 12 if name == "all" else 6
        code = np.minimum(t, top).astype(int)
        oh = np.zeros((len(frame), top + 1), np.float32)
        oh[np.arange(len(frame)), code] = 1.0
        cols += [oh, (t / max(top, 1)).clip(0, 1.5)[:, None].astype(np.float32)]
    if "all" in tot and "adult" in tot:
        dep = (tot["all"] - tot["adult"]) / np.maximum(tot["adult"], 1)
        b = np.digitize(dep, [0.01, 0.5, 1.0, 1.5, 2.5])
        oh = np.zeros((len(frame), 6), np.float32)
        oh[np.arange(len(frame)), b] = 1.0
        cols.append(oh)
    return np.hstack(cols)


def _profiles(frame, given):
    if not given:
        return np.zeros(len(frame), np.int64)
    key = frame[given].astype(str).agg("\x1f".join, axis=1)
    return pd.factorize(key)[0].astype(np.int64)


# ----------------------------------------------------------------------------- multi-head nets

class _Heads:
    """Index maps between the flat logit vector (sum K) and a padded (J, Kmax) view."""

    def __init__(self, widths):
        self.widths = np.asarray(widths)
        self.offsets = np.concatenate([[0], np.cumsum(self.widths)])
        self.J, self.kmax, self.total = len(widths), int(self.widths.max()), int(self.widths.sum())
        gather = np.zeros((self.J, self.kmax), np.int64)
        mask = np.zeros((self.J, self.kmax), bool)
        for j, w in enumerate(self.widths):
            gather[j, :w] = self.offsets[j] + np.arange(w)
            mask[j, :w] = True
        self.gather = torch.tensor(gather, device=DEVICE)
        self.mask = torch.tensor(mask, device=DEVICE)

    def logp(self, logits):                        # (B, total) -> (B, J, Kmax) log-probs
        z = logits[:, self.gather].masked_fill(~self.mask, -1e9)
        return torch.log_softmax(z, dim=-1)

    def flat(self, logp):                          # (B, J, Kmax) log-probs -> (B, total) probs
        p = logp.exp()
        return p[:, self.mask]


def _make_net(d, total, hidden, drop, depth=2):
    if hidden == 0:
        return nn.Linear(d, total)
    layers, din = [], d
    for _ in range(depth):
        layers += [nn.Linear(din, hidden), nn.GELU(), nn.Dropout(drop)]
        din = hidden
    return nn.Sequential(*layers, nn.Linear(hidden, total))


def _train(Xtr, Ytr, Xva, Yva, heads, hidden, drop, wd, lr, max_epochs, deadline, seed, depth=2):
    torch.manual_seed(seed)
    net = _make_net(Xtr.shape[1], heads.total, hidden, drop, depth).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xtr_t, Ytr_t = torch.tensor(Xtr, device=DEVICE), torch.tensor(Ytr, device=DEVICE)
    Xva_t, Yva_t = torch.tensor(Xva, device=DEVICE), torch.tensor(Yva, device=DEVICE)
    n = len(Xtr)
    bs = 512 if n > 4000 else 256
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def loss_of(lp, Y):
        valid = Y >= 0
        nll = -lp.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    best, best_state, bad, patience = float("inf"), None, 0, 6
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = loss_of(heads.logp(net(Xtr_t[b])), Ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_of(heads.logp(net(Xva_t)), Yva_t).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time.time() > deadline:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


class _LCA(nn.Module):
    """P(y_j | x) = sum_z pi_z(x) theta_zj(y_j): a gating trunk over latent types and one answer distribution per type and item."""

    def __init__(self, d, heads, L, hidden, drop, base_logit):
        super().__init__()
        self.heads, self.L = heads, L
        self.gate = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hidden, L))
        self.delta = nn.Parameter(0.5 * torch.randn(L, heads.total))
        self.register_buffer("base", torch.tensor(base_logit, dtype=torch.float32)[None, :])

    def type_logp(self):                                   # (L, J, Kmax) log answer distributions per type
        return self.heads.logp(self.base + self.delta)

    def forward(self, X):
        lt = self.type_logp()
        lpi = torch.log_softmax(self.gate(X), dim=-1)      # (B, L)
        mix = (lpi.exp() @ lt.exp().reshape(self.L, -1)).reshape(len(X), self.heads.J, self.heads.kmax)
        return lt, lpi, mix


def _train_lca(Xtr, Ytr, Xva, Yva, heads, base_logit, deadline, seed):
    torch.manual_seed(seed)
    net = _LCA(Xtr.shape[1], heads, LCA_CLASSES, LCA_HIDDEN, LCA_DROP, base_logit).to(DEVICE)
    opt = torch.optim.AdamW([{"params": net.gate.parameters(), "weight_decay": LCA_WD},
                             {"params": [net.delta], "weight_decay": 1e-3}], lr=LCA_LR)
    Xtr_t, Ytr_t = torch.tensor(Xtr, device=DEVICE), torch.tensor(Ytr, device=DEVICE)
    Xva_t, Yva_t = torch.tensor(Xva, device=DEVICE), torch.tensor(Yva, device=DEVICE)
    n, J = len(Xtr), heads.J
    jj = torch.arange(J, device=DEVICE)[None, :]
    bs = 512 if n > 4000 else 256
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def marg_nll(mix, Y):
        valid = Y >= 0
        p = mix.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
        return -(torch.log(p + 1e-9) * valid).sum() / valid.sum().clamp(min=1)

    def loss_of(X, Y):
        lt, lpi, mix = net(X)
        valid = (Y >= 0).float()
        lp = lt[:, jj, Y.clamp(min=0)]                     # (L, B, J)
        joint = (lp * valid[None]).sum(-1).T + lpi         # (B, L)
        nll_joint = -torch.logsumexp(joint, dim=1).mean() / J
        return nll_joint + LCA_MARG_W * marg_nll(mix, Y)

    best, best_state, bad, patience = float("inf"), None, 0, 6
    for _ in range(LCA_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = loss_of(Xtr_t[b], Ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = marg_nll(net(Xva_t)[2], Yva_t).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time.time() > deadline:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _predict_lca(net, X, heads):
    out = []
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEVICE)
        for i in range(0, len(Xt), 2048):
            out.append(net(Xt[i:i + 2048])[2][:, heads.mask].float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, heads.total), np.float32)


def _predict_flat(net, X, heads):
    out = []
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEVICE)
        for i in range(0, len(Xt), 4096):
            out.append(heads.flat(heads.logp(net(Xt[i:i + 4096]))).float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, heads.total), np.float32)


# ----------------------------------------------------------------------------- simple estimators

def _onehot_flat(Y, heads):
    n = len(Y)
    M = np.zeros((n, heads.total), np.float32)
    for j in range(heads.J):
        ok = Y[:, j] >= 0
        M[np.flatnonzero(ok), heads.offsets[j] + Y[ok, j]] = 1.0
    return M


def _marginal_flat(Yoh, heads):
    m = Yoh.sum(0) + ALPHA
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        m[s] /= m[s].sum()
    return m.astype(np.float32)


def _profile_flat(prof_tr, Yoh_tr, prof_q, base_q, k):
    n_prof = int(max(prof_tr.max(initial=0), prof_q.max(initial=0))) + 1
    C = np.zeros((n_prof, Yoh_tr.shape[1]), np.float32)
    np.add.at(C, prof_tr, Yoh_tr)
    N = np.bincount(prof_tr, minlength=n_prof).astype(np.float32)
    return (C[prof_q] + k * base_q) / (N[prof_q][:, None] + k)


# ----------------------------------------------------------------------------- blending

def _scored(p, K):
    p = p / p.sum(1, keepdims=True)
    return EPS + (1 - K * EPS) * p


def _fit_weights(cands, y, scale=1.0):
    """cands: list of (n, K) OOF prob arrays; y: (n,) labels. Softmax weights maximising mean log score
    (scale multiplies the log-score term against the fixed shrinkage penalty; 1.0 = the standard objective)."""
    K = cands[0].shape[1]
    P = np.stack([_scored(c, K)[np.arange(len(y)), y] for c in cands], 1)   # (n, C)

    def f(a):
        w = np.exp(a - a.max())
        w /= w.sum()
        mix = P @ w
        val = -scale * np.log(mix).mean() + 1e-4 * (a ** 2).sum()
        g_w = -scale * (P / mix[:, None]).mean(0)
        grad = w * (g_w - (w * g_w).sum()) + 2e-4 * a
        return val, grad

    res = minimize(f, np.zeros(len(cands)), jac=True, method="L-BFGS-B")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


NO_LOGPOOL = os.environ.get("SB_NO_LOGPOOL") == "1"
LOGPOOL_MARGIN = 0.5               # nats of held-out gain (summed over the item's rows) needed to prefer the log pool
LOGPOOL_TINY = 1e-6
LOGPOOL_STATS = []                 # (total weight, switched) per item, for diagnostics


def _lp_stack(cands):
    L = []
    for c in cands:
        c = np.asarray(c, np.float64)
        c = c / np.maximum(c.sum(1, keepdims=True), 1e-300)
        L.append(np.log(c + LOGPOOL_TINY))
    return np.stack(L)


def _lp_probs(a, LP):
    z = np.tensordot(a, LP, axes=1)
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def _fit_logpool(LP, y, w0):
    """Weights a_c >= 0 (free total) of p ~ exp(sum_c a_c log p_c), maximising the floored mean log score."""
    C, n, K = LP.shape
    idx = np.arange(n)
    LPy = LP[:, idx, y]

    def f(a):
        q = _lp_probs(a, LP)
        qy = q[idx, y]
        s = EPS + (1 - K * EPS) * qy
        val = -np.log(s).mean() + 1e-4 * ((a - w0) ** 2).sum()
        g = (1 - K * EPS) * qy / s / n
        gz = -g[:, None] * (np.eye(K)[y] - q)
        grad = np.tensordot(LP, gz, axes=([1, 2], [0, 1])) + 2e-4 * (a - w0)
        return val, grad

    res = minimize(f, np.maximum(w0, 1e-3), jac=True, method="L-BFGS-B", bounds=[(0, 30)] * C, options={"maxiter": 40})
    return res.x


def _logpool_choice(cands, y, half, w_lin):
    """Pool the candidates (OOF probabilities on the item's rows) in the log domain when a nested comparison
    (fit on one half of the rows, score on the other) shows a held-out gain over the linear pool; else None."""
    if len(cands) < 2 or len(y) < 100 or min(int((half == 0).sum()), int((half == 1).sum())) < 40:
        return None
    K = cands[0].shape[1]
    idx = np.arange(len(y))
    LP = _lp_stack(cands)
    gain = 0.0
    for h in (0, 1):
        tr, te = half != h, half == h
        wl = _fit_weights([c[tr] for c in cands], y[tr])
        a = _fit_logpool(LP[:, tr], y[tr], wl)
        lin = sum(w * c[te] for w, c in zip(wl, cands))
        yt = y[te]
        ti = np.arange(len(yt))
        gain += np.log(_scored(_lp_probs(a, LP[:, te]), K)[ti, yt]).sum() - np.log(_scored(lin, K)[ti, yt]).sum()
    if gain <= LOGPOOL_MARGIN:
        LOGPOOL_STATS.append((float(np.sum(w_lin)), False))
        return None
    a = _fit_logpool(LP, y, w_lin)
    LOGPOOL_STATS.append((float(a.sum()), True))
    return a


# ----------------------------------------------------------------------------- skip logic

GATE_MIN_ROWS = 30                  # visible rows needed on a side of a gate before its routing is trusted as exact
HARD_GATES = os.environ.get("SB_HARD_GATES") == "1"


def _gate_rules(frame, vis, Y, pred, rec, heads, given_set, sub=None, min_rows=GATE_MIN_ROWS):
    """Per item with a GIVEN gate parent: ((asked side exact, skipped side exact), (rows on each side)), judged on the
    visible rows `vis[sub]` (all visible rows when `sub` is None). A side is exact only if it has >= GATE_MIN_ROWS
    labelled rows and not one of them contradicts the routing; a stated gate that only approximates who was asked is
    left to the learned models instead of being forced."""
    rows = vis if sub is None else vis[sub]
    Ys = Y if sub is None else Y[sub]
    trust, nside = {}, {}
    for j, n in enumerate(pred):
        g = rec[n].get("gate")
        if not g or g["parent"] not in given_set:
            continue
        asked = frame[g["parent"]].iloc[rows].isin(g["observed_if"]).to_numpy()
        known = Ys[:, j] >= 0
        isg = Ys[:, j] == heads.widths[j] - 1
        a, b = asked & known, ~asked & known
        nside[n] = (int(a.sum()), int(b.sum()))
        trust[n] = (HARD_GATES or (a.sum() >= min_rows and not isg[a].any()),
                    HARD_GATES or (b.sum() >= min_rows and isg[b].all()))
    return trust, nside


def _gate_trust(frame, vis, Y, pred, rec, heads, given_set):
    trust, _gate_override.nside = _gate_rules(frame, vis, Y, pred, rec, heads, given_set)
    return trust


def _bb_res(ab, n):
    """Posterior residual (a + 0) / (n + a + b) of a rule held on n rows with no exception, clipped."""
    a, b = ab
    return np.clip(a / (np.asarray(n, np.float64) + a + b), RES_MIN, RES_MAX)


def _gate_override(P, rows_frame, pred, rec, heads, given_set, trust=None, nside=None, ab=None):
    """Apply gating in place for items whose gate parent is a GIVEN (always visible) item, on each side of the gate
    the visible rows show to be exact. An exact side keeps a residual on the other outcome (the gate slot on the
    asked side, the substantive options on the skipped side): rule of succession 0.5 / (n_side + 1), or the fitted
    posterior residual `ab[...]` when given."""
    if trust is None:
        trust = getattr(_gate_override, "trust", {})
    if nside is None:
        nside = getattr(_gate_override, "nside", {})
    ab = ab or {}
    for j, n in enumerate(pred):
        g = rec[n].get("gate")
        if not g or g["parent"] not in given_set:
            continue
        t_asked, t_skip = trust.get(n, (True, True))
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        asked = rows_frame[g["parent"]].isin(g["observed_if"]).to_numpy()
        block = P[:, s]
        base_soft = not (NO_GATE_KT or HARD_GATES) and n in nside
        ab_a = ab.get("gate_asked") if (n in nside and not HARD_GATES) else None
        ab_s = ab.get("gate_skip") if (n in nside and not HARD_GATES) else None
        soft_a = base_soft or ab_a is not None
        soft_s = base_soft or ab_s is not None
        if soft_a or soft_s:
            res_a = float(_bb_res(ab_a, nside[n][0])) if ab_a is not None else 0.5 / (nside[n][0] + 1.0)
            res_s = float(_bb_res(ab_s, nside[n][1])) if ab_s is not None else 0.5 / (nside[n][1] + 1.0)
        if t_asked:
            if soft_a:
                sub = block[asked, :-1]
                tot = sub.sum(1, keepdims=True)
                sub = np.where(tot > 0, sub / np.maximum(tot, 1e-30), 1.0 / sub.shape[1])
                block[np.ix_(asked, np.arange(block.shape[1] - 1))] = sub * (1.0 - res_a)
                block[asked, -1] = res_a
            else:
                block[asked, -1] = 0.0
                empty = asked & (block[:, :-1].sum(1) <= 0)
                block[empty, :-1] = 1.0
        if t_skip:
            if soft_s:
                sub = block[~asked, :-1]
                tot = sub.sum(1, keepdims=True)
                sub = np.where(tot > 0, sub / np.maximum(tot, 1e-30), 1.0 / sub.shape[1])
                block[np.ix_(~asked, np.arange(block.shape[1] - 1))] = sub * res_s
                block[~asked, -1] = 1.0 - res_s
            else:
                block[~asked, :] = 0.0
                block[~asked, -1] = 1.0
        P[:, s] = block
    return P


# ----------------------------------------------------------------------------- hierarchical cells

def _mi_rank(codes_list, Yoh, heads):
    """Mean mutual information between each GIVEN column and the targets (training rows). Higher = more informative."""
    import scipy.sparse as sp
    scores = []
    for c in codes_list:
        K = int(c.max()) + 1
        S = sp.csr_matrix((np.ones(len(c), np.float32), (c, np.arange(len(c)))), shape=(K, len(c)))
        Cnt = np.asarray(S @ Yoh, dtype=np.float64)
        mis = []
        for j in range(heads.J):
            N = Cnt[:, heads.offsets[j]:heads.offsets[j + 1]]
            t = N.sum()
            if t <= 0:
                continue
            p = N / t
            pc = p.sum(1, keepdims=True)
            pk = p.sum(0, keepdims=True)
            nz = p > 0
            mis.append(float((p[nz] * np.log(p[nz] / (pc @ pk)[nz])).sum()))
        scores.append(np.mean(mis) if mis else 0.0)
    return np.array(scores)


def _cells_predict(codes_tr, Yoh_tr, codes_q, marg, alphas):
    """Back-off hierarchy: p_L = (C_L[cell] + a * p_{L-1}) / (N_L[cell] + a), from the global marginal down to the full profile."""
    import scipy.sparse as sp
    nq = len(codes_q[0])
    P = {a: np.tile(marg, (nq, 1)).astype(np.float32) for a in alphas}
    for ctr, cq in zip(codes_tr, codes_q):
        ncodes = int(max(ctr.max(initial=0), cq.max(initial=0))) + 1
        S = sp.csr_matrix((np.ones(len(ctr), np.float32), (ctr, np.arange(len(ctr)))), shape=(ncodes, len(ctr)))
        C = np.asarray(S @ Yoh_tr, dtype=np.float32)
        N = np.bincount(ctr, minlength=ncodes).astype(np.float32)
        Cq, Nq = C[cq], N[cq][:, None]
        for a in alphas:
            P[a] = (Cq + a * P[a]) / (Nq + a)
    return P


def _cells_candidate(frame, given, idx, vis, held_pos, Yoh, Y, fold, covered, heads):
    codes = [frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy() + 1 for g in given]
    order = np.argsort(-_mi_rank([c[vis] for c in codes], Yoh, heads))
    levels, cur = [], None
    for g in order:
        c = codes[g]
        cur = pd.factorize(c if cur is None else cur.astype(np.int64) * (int(c.max()) + 1) + c)[0]
        levels.append(cur)
    lv_vis = [l[vis] for l in levels]
    lv_held = [l[held_pos] for l in levels]
    oof = {a: np.zeros((len(vis), heads.total), np.float32) for a in CELL_ALPHAS}
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(covered & (fold != f)), np.flatnonzero(fold == f)
        if len(va) == 0 or len(tr) == 0:
            continue
        P = _cells_predict([l[tr] for l in lv_vis], Yoh[tr], [l[va] for l in lv_vis], _marginal_flat(Yoh[tr], heads), CELL_ALPHAS)
        for a in CELL_ALPHAS:
            oof[a][va] = P[a]
    lls = {a: _total_ll(oof[a][covered], Y[covered], heads) for a in CELL_ALPHAS}
    best = max(lls, key=lls.get)
    tr = np.flatnonzero(covered)
    test = _cells_predict([l[tr] for l in lv_vis], Yoh[tr], lv_held, _marginal_flat(Yoh[tr], heads), (best,))[best]
    _log(f"  cells: {len(levels)} levels, MI order {list(order[:5])}, best alpha {best}")
    return oof[best], test


def _eb_cells_candidate(frame, given, idx, vis, held_pos, Yoh, Y, fold, covered, heads):
    """Back-off hierarchy of the cells candidate with the strength chosen per item and per level by leave-one-out likelihood on the
    training rows of each fit (so the out-of-fold vectors never see their fold's labels)."""
    if EBCELLS_FAIL:
        raise RuntimeError("SB_EBCELLS_FAIL")
    codes = [frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy() + 1 for g in given]
    order = np.argsort(-_mi_rank([c[vis] for c in codes], Yoh, heads))
    levels, cur = [], None
    for g in order:
        c = codes[g]
        cur = pd.factorize(c if cur is None else cur.astype(np.int64) * (int(c.max()) + 1) + c)[0]
        levels.append(cur)
    lv_vis = [l[vis] for l in levels]
    lv_held = [l[held_pos] for l in levels]
    nlev = [int(l.max()) + 1 for l in levels]
    alphas = np.asarray(EB_ALPHAS)
    n = len(vis)

    def fit_predict(tr, qlevs):
        nq = len(qlevs[0])
        out = np.zeros((nq, heads.total), np.float32)
        for j in range(heads.J):
            K = int(heads.widths[j])
            okr = tr[Y[tr, j] >= 0]
            if len(okr) == 0:
                out[:, heads.offsets[j]:heads.offsets[j + 1]] = 1.0 / K
                continue
            y = Y[okr, j]
            m0 = np.bincount(y, minlength=K) + 0.5
            m0 = m0 / m0.sum()
            p_tr = np.tile(m0, (len(okr), 1))
            p_q = np.tile(m0, (nq, 1))
            ar = np.arange(len(y))
            for L in range(len(lv_vis)):
                m = nlev[L]
                ctr = lv_vis[L][okr]
                C = np.bincount(ctr * K + y, minlength=m * K).reshape(m, K).astype(np.float64)
                N = C.sum(1)
                Cr = C[ctr]
                Cr[ar, y] -= 1.0
                Nr = N[ctr] - 1.0
                py = p_tr[ar, y]
                ll = np.log((Cr[ar, y][None, :] + alphas[:, None] * py[None, :]) / (Nr[None, :] + alphas[:, None])).sum(1)
                a = alphas[int(np.argmax(ll))]
                p_tr = (Cr + a * p_tr) / (Nr[:, None] + a)
                lq = qlevs[L]
                p_q = (C[lq] + a * p_q) / (N[lq][:, None] + a)
            out[:, heads.offsets[j]:heads.offsets[j + 1]] = p_q
        return out

    oof = np.zeros((n, heads.total), np.float32)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(covered & (fold != f)), np.flatnonzero(fold == f)
        if len(va) == 0 or len(tr) == 0:
            continue
        oof[va] = fit_predict(tr, [l[va] for l in lv_vis])
    tr = np.flatnonzero(covered)
    test = fit_predict(tr, lv_held)
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        raise ValueError("non-finite eb cells")
    _log(f"  eb cells: {len(levels)} levels, MI order {list(order[:5])}")
    return oof, test


# ----------------------------------------------------------------------------- multi-output forest

def _forest_candidate(Xv, Y, Xh, fold, heads, deadline):
    """Multi-output ExtraTrees on the one-hot GIVEN block (threads only, no new packages). None if it cannot finish in time."""
    from sklearn.ensemble import ExtraTreesClassifier
    n = len(Xv)
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((len(Xh), heads.total), np.float32)
    t0 = time.time()

    def fill(clf, X):
        out = np.zeros((len(X), heads.total), np.float32)
        probs = clf.predict_proba(X)
        if not isinstance(probs, list):
            probs = [probs]
        for j in range(heads.J):
            cls = np.asarray(clf.classes_[j] if isinstance(clf.classes_, list) else clf.classes_)
            keep = cls >= 0
            out[:, heads.offsets[j] + cls[keep]] = probs[j][:, keep]
            s = slice(heads.offsets[j], heads.offsets[j + 1])
            out[:, s] /= np.maximum(out[:, s].sum(1, keepdims=True), 1e-12)
        return out

    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        clf = ExtraTreesClassifier(n_estimators=FOREST_TREES, min_samples_leaf=FOREST_LEAF, max_features=0.6,
                                   n_jobs=8, random_state=f)
        clf.fit(Xv[tr], Y[tr])
        m = np.tile(_marginal_flat(_onehot_flat(Y[tr], heads), heads), (1, 1))
        oof[va] = 0.95 * fill(clf, Xv[va]) + 0.05 * m
        test += (0.95 * fill(clf, Xh) + 0.05 * m) / N_FOLDS
        per_fold = (time.time() - t0) / (f + 1)
        _log(f"  forest fold {f} ({time.time() - t0:.0f}s)")
        if time.time() + (N_FOLDS - f - 1) * per_fold * 1.2 > deadline:
            return None
    return oof, test


# ----------------------------------------------------------------------------- TabICLv2 candidate

_tabicl_state = {"cls": None, "tried": False, "ckpt": None, "spent": 0.0}


def _find_ckpt():
    """Locate the TabICLv2 checkpoint: bundled in the zip first, else the pre-fetched Hugging Face cache (never downloads)."""
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", TABICL_CKPT)
    if os.path.isfile(bundled):                       # weights shipped inside the zip: no dependence on the HF pre-fetch
        return bundled
    roots = [os.environ.get("HF_HUB_CACHE"),
             os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
             os.path.expanduser("~/.cache/huggingface/hub")]
    for r in roots:
        if r and os.path.isdir(r):
            for dirpath, _, files in os.walk(r):
                if TABICL_CKPT in files:
                    return os.path.join(dirpath, TABICL_CKPT)
    return None


def _tabicl_ready():
    """Import tabicl and find its weights once; any failure just means the candidate is absent."""
    st = _tabicl_state
    if st["tried"]:
        return st["cls"] is not None
    st["tried"] = True
    try:
        vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        if os.path.isdir(vendor) and vendor not in sys.path:
            sys.path.insert(0, vendor)            # bundled tabicl + einops (pure python): no pip install needed
        from tabicl import TabICLClassifier
        ckpt = _find_ckpt()
        if ckpt is None:
            _log("tabicl: checkpoint not found in the HF cache")
            return False
        st["cls"], st["ckpt"] = TabICLClassifier, ckpt
        _log(f"tabicl ready: {ckpt}")
        return True
    except Exception as e:
        _log("tabicl unavailable:", repr(e))
        return False


_tabpfn_state = {"tried": False, "cls": None, "ckpt": None}


def _tabpfn_ready():
    """Import tabpfn and find the bundled TabPFN-3.5 weights once; any failure just means the candidate is absent."""
    st = _tabpfn_state
    if st["tried"]:
        return st["cls"] is not None
    st["tried"] = True
    try:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")   # build and place the network once, not per fit
        here = os.path.dirname(os.path.abspath(__file__))
        ckpt = os.path.join(here, "weights", TABPFN_CKPT)
        if not os.path.isfile(ckpt):
            _log("tabpfn: checkpoint not bundled")
            return False
        # bundled pure-python tabpfn first; its pydantic stack last, so an installed pydantic pair takes precedence
        vendor, vendor_pd = os.path.join(here, "vendor"), os.path.join(here, "vendor_pd")
        if os.path.isdir(vendor) and vendor not in sys.path:
            sys.path.insert(0, vendor)
        if os.path.isdir(vendor_pd) and vendor_pd not in sys.path:
            sys.path.append(vendor_pd)
        # skrub is only used for datetime/text columns (none here); a stub keeps it, matplotlib and their cache
        # directories and font-scan subprocess out of the run
        import types
        sk = types.ModuleType("skrub")

        class _Absent:
            def __init__(self, *a, **k):
                raise RuntimeError("skrub is not bundled")
        sk.DatetimeEncoder = sk.StringEncoder = _Absent
        sys.modules.setdefault("skrub", sk)
        from tabpfn import TabPFNClassifier
        import tabpfn.checkpoint as tck
        if not getattr(tck.Checkpoint._load_safetensors, "_sb_dequant", False):
            raw = tck.Checkpoint._load_safetensors

            def _load_st(self):
                c = raw(self)
                _dequant(c.get("state_dict"))
                return c
            _load_st._sb_dequant = True
            tck.Checkpoint._load_safetensors = _load_st
        st["cls"], st["ckpt"] = TabPFNClassifier, ckpt
        _log(f"tabpfn ready: {ckpt}")
        return True
    except Exception as e:
        _log("tabpfn unavailable:", repr(e))
        return False


def _free_device():
    import gc
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------- TabPFN, targets sharing one context
# Items whose context rows and query rows are identical (all GIVEN columns are shared; only the target differs) share the
# input cleaning, preprocessing fit and query transform; each target keeps its own label encoding and class permutation,
# and the prepared ensemble members of several targets go through one forward pass (the library's own batched call).
# A target whose generated ensemble config differs from the base one in anything but the class permutation is fitted in
# full, so predictions equal the per-item fit/predict_proba path.

def _pfn_predict_inputs(clf, Xq):
    """The input cleaning of TabPFNClassifier._raw_predict, up to the model call."""
    from tabpfn.validation import check_input_shape_matches, ensure_compatible_predict_input_sklearn
    from tabpfn.preprocessing.clean import clean_data_transform
    from tabpfn.preprocessing.datamodel import FeatureModality
    check_input_shape_matches(Xq, estimator=clf)
    X = clf.date_transformer_.transform(Xq)
    X = clf.text_transformer_.transform(X)
    X = ensure_compatible_predict_input_sklearn(X, clf)
    return clean_data_transform(
        X, cat_indices=clf.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL),
        ord_encoder=getattr(clf, "ordinal_encoder_", None), passthrough_inf=clf.get_inference_config().PASSTHROUGH_INF)


def _pfn_eq(a, b):
    try:
        return bool(np.all(a == b))
    except Exception:
        return a == b


class _PfnStub:
    """Executor stand-in carrying precomputed per-member outputs."""
    def __init__(self, pre):
        self.pre = pre

    def use_torch_inference_mode(self, use_inference):
        pass

    def iter_outputs(self, X, autocast=None, task_type=None, only_return_standard_out=True):
        return iter(self.pre)


def _pfn_clone_for_target(base, y, static_seed):
    import copy
    import dataclasses
    from tabpfn.preprocessing.label_encoder import TabPFNLabelEncoder
    from tabpfn.preprocessing.ensemble import generate_classification_ensemble_configs
    c = copy.copy(base)
    c.label_encoder_ = TabPFNLabelEncoder(original_target_name=None)
    y_enc, meta = c.label_encoder_.fit_transform(y=np.asarray(y), max_num_classes=c.inference_config_.MAX_NUMBER_OF_CLASSES)
    c.classes_, c.n_classes_, c.class_counts_ = meta.classes, meta.n_classes, meta.class_counts
    cfgs = generate_classification_ensemble_configs(
        num_estimators=c.n_estimators_, add_fingerprint_feature=c.inference_config_.FINGERPRINT_FEATURE,
        feature_shift_decoder=c.inference_config_.FEATURE_SHIFT_METHOD, polynomial_features=c.inference_config_.POLYNOMIAL_FEATURES,
        preprocessor_configs=c.inference_config_.PREPROCESS_TRANSFORMS, class_shift_method=c.inference_config_.CLASS_SHIFT_METHOD,
        n_classes=c.n_classes_, random_state=static_seed, num_models=len(c.models_),
        outlier_removal_std=c.inference_config_.get_resolved_outlier_removal_std(estimator_type=c.estimator_type),
        passthrough_inf=c.get_inference_config().PASSTHROUGH_INF)
    bm = base.executor_.ensemble_members
    if len(cfgs) != len(bm):
        return None
    members = []
    for cfg, m in zip(cfgs, bm):
        o = m.config
        same = all(getattr(cfg, f.name) is getattr(o, f.name) or _pfn_eq(getattr(cfg, f.name), getattr(o, f.name))
                   for f in dataclasses.fields(cfg) if f.name != "class_permutation")
        if not same:
            return None
        yt = cfg.class_permutation[y_enc] if cfg.class_permutation is not None else y_enc
        members.append(dataclasses.replace(m, config=cfg, y_train=yt))
    c.ensemble_configs_ = cfgs
    c._shared_members = members
    return c


def _pfn_fit_shared(Cls, ckpt, X_ctx, ys, X_query, random_state):
    """One shared context and query, several targets ys (each with >= 2 classes). Returns [(classes, proba), ...]."""
    from tabpfn import inference as I
    from tabpfn.utils import infer_random_state
    base = Cls(model_path=ckpt, device=str(DEVICE), n_estimators=TABPFN_EST, random_state=random_state,
               n_preprocessing_jobs=1, enable_torch_compile=False, show_progress_bar=False)
    base.fit(X_ctx, ys[0])
    static_seed, _ = infer_random_state(random_state)
    clfs = [base]
    base._shared_members = list(base.executor_.ensemble_members)
    for y in ys[1:]:
        c = _pfn_clone_for_target(base, y, static_seed)
        if c is None:
            c = Cls(model_path=ckpt, device=str(DEVICE), n_estimators=TABPFN_EST, random_state=random_state,
                    n_preprocessing_jobs=1, enable_torch_compile=False, show_progress_bar=False)
            c.fit(X_ctx, y)
            c._shared_members = None
        clfs.append(c)
    Xc = _pfn_predict_inputs(base, X_query)
    eng0 = base.executor_
    devices = eng0.get_devices()
    xt_cache = {}

    def xtest(m):
        k = id(m.cpu_preprocessor)
        if k not in xt_cache:
            xt_cache[k] = m.transform_X_test(Xc)
        return xt_cache[k]
    members, X_tests, owner = [], [], []
    for ci, c in enumerate(clfs):
        for m in (c._shared_members if c._shared_members is not None else c.executor_.ensemble_members):
            members.append(m)
            X_tests.append(xtest(m))
            owner.append(ci)
    save_peak = I.should_save_peak_mem(
        memory_saving_mode=eng0.save_peak_mem, X_train_shape=tuple(eng0.X_train_shape_before_preprocessing),
        X_test_shape=tuple(Xc.shape), devices=devices, dtype_byte_size=eng0.dtype_byte_size)
    cap = I._members_per_forward(members[0].X_train.shape[0] + np.shape(X_tests[0])[0], members[0].X_train.shape[1],
                                 I._shared_estimator_batch_budget(eng0.model_caches))
    outs = [None] * len(members)
    order = sorted(range(len(members)), key=lambda i: (tuple(members[i].X_train.shape), tuple(np.shape(X_tests[i]))))
    chunks, cur, key = [], [], None
    for i in order:
        k = (tuple(members[i].X_train.shape), tuple(np.shape(X_tests[i])))
        if cur and (k != key or len(cur) >= cap):
            chunks.append(cur)
            cur = []
        cur.append(i)
        key = k
    if cur:
        chunks.append(cur)
    for ch in chunks:
        res = I._call_model_on_members(
            eng0.model_caches, device=devices[0], members=[members[i] for i in ch], X_tests=[X_tests[i] for i in ch],
            force_inference_dtype=eng0.force_inference_dtype, autocast=base.use_autocast_,
            enable_torch_compile=eng0.enable_torch_compile, inference_mode=True, only_return_standard_out=True,
            save_peak_mem=save_peak, task_type="multiclass")
        for positions, output in res:
            for col, p in enumerate(positions):
                outs[ch[p]] = I._move_and_squeeze_output(I._member_output(output, col), devices[0])
    per = {ci: [] for ci in range(len(clfs))}
    for i, ci in enumerate(owner):
        per[ci].append((outs[i], members[i].config))
    res = []
    for ci, c in enumerate(clfs):
        pre = per[ci]
        if c is not base and c._shared_members is not None:
            c.executor_ = _PfnStub(pre)
        else:
            c.executor_.iter_outputs = lambda X, autocast=None, task_type=None, only_return_standard_out=True, _p=pre: iter(_p)
        p = c.forward(Xc, use_inference_mode=True, return_logits=False).float().detach().cpu().numpy()
        p = c._maybe_reweight_probas(probas=p)
        res.append((c.classes_, p / p.sum(axis=1, keepdims=True)))
    return res


def _pfn_clear():
    try:
        from tabpfn.model_loading import clear_built_model_cache
        clear_built_model_cache()
    except Exception:
        pass


def _pfn_item_fold(Cls, ckpt, Xdf, vis, held_pos, Y, fold, f, items, widths, rs=0):
    """TabPFN fits of `items` on fold f. Per item the context is the other folds' rows with a known answer (a random
    subsample of PFN_MAX_CTX rows when larger); the query is fold f's rows followed by the rows to score. Items with
    an identical context share one preprocessing. Returns {j: (pr, va)}, pr = (len(va)+len(held_pos), widths[j])
    class probabilities; items whose context has < 2 classes are left out."""
    va = np.flatnonzero(fold == f)
    q = np.r_[vis[va], held_pos]
    Xq = Xdf.iloc[q]
    ctxs = {}
    for j in items:
        tr = np.flatnonzero((fold != f) & (Y[:, j] >= 0))
        if len(tr) > PFN_MAX_CTX:
            tr = np.sort(np.random.default_rng(1000 * j + f).choice(tr, PFN_MAX_CTX, replace=False))
        if len(np.unique(Y[tr, j])) >= 2:
            ctxs[j] = tr
    groups = {}
    for j, tr in ctxs.items():
        groups.setdefault(tr.tobytes(), []).append(j)
    out = {}
    for js in groups.values():
        tr = ctxs[js[0]]
        Xc = Xdf.iloc[vis[tr]]
        res = _pfn_fit_shared(Cls, ckpt, Xc, [Y[tr, j] for j in js], Xq, f + rs)
        for j, (cl, p) in zip(js, res):
            pr = np.zeros((len(q), widths[j]))
            pr[:, cl] = p
            out[j] = (pr, va)
    return out


PFN_CHUNK = 16        # items per shared-context call


def _pfn_repeat_passes(run_chunk, plan_cost, state, S, k_done, fold, heads, oof, test, stage_deadline, frac):
    """Variance reduction with the time a stage has left: repeat the covered items on the completed folds with a different
    random preprocessing (random_state offset per pass) and average the probabilities of all passes. An item is updated
    only once it has finished every completed fold in the pass, so its OOF rows and its test rows always average the
    same number of passes; chunks that do not finish are discarded. Items go in the priority order of the first pass,
    at most ANYTIME_PFN_PASSES passes in all, and no chunk is started that the measured cost says would overrun."""
    t_x = time.time()
    end = t_x + frac * max(stage_deadline - t_x, 0.0)
    if _kumo_hold["s"] + _gp_hold["s"] > 0:
        end = min(end, stage_deadline - _kumo_hold["s"] - _gp_hold["s"])
    cnt = {j: 1 for j in S}
    for p in range(1, ANYTIME_PFN_PASSES):
        pos, stop = 0, False
        todo = [j for j in S if cnt[j] == p]
        while pos < len(todo) and not stop:
            c = plan_cost() * 1.3 * k_done
            size = min(state["cap"], int((end - time.time()) / c)) if c > 0 else 0
            if size < 1:
                break
            chunk = todo[pos:pos + size]
            pos += len(chunk)
            got = {f: None for f in range(k_done)}
            for f in range(k_done):
                res, _ = run_chunk(f, chunk, 1000 * p)
                if len(res) < len(chunk) or not all(np.isfinite(q).all() for q in res.values()):
                    stop = True
                    break
                got[f] = res
            if any(g is None for g in got.values()):
                break
            for j in chunk:
                s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
                n_j = cnt[j]
                tsum = np.zeros((test.shape[0], s1 - s0), np.float64)
                for f in range(k_done):
                    va = np.flatnonzero(fold == f)
                    q = got[f][j]
                    oof[va, s0:s1] = (oof[va, s0:s1] * n_j + q[:len(va)]) / (n_j + 1)
                    tsum += q[len(va):]
                test[:, s0:s1] = (test[:, s0:s1] * n_j + tsum / k_done) / (n_j + 1)
                cnt[j] = n_j + 1
        if not [j for j in S if cnt[j] == p + 1]:
            break
    _log(f"tabpfn: repeat passes {time.time() - t_x:.0f}s, items by pass count {np.bincount(list(cnt.values())).tolist()}")


def _fullctx_merge(test_fold, test_full):
    """Test-row probabilities from the fold average and the full-context pass (per FULLCTX_MODE)."""
    if FULLCTX_MODE == "mix":
        return 0.5 * (test_fold + test_full)
    return test_full


def _pfn_full_pass(Cls, ckpt, Xdf, vis, held_pos, Y, heads, S, plan_cost, state, stage_deadline):
    """One extra TabPFN pass over the covered items with all visible rows (with a known answer; a random subsample of
    PFN_MAX_CTX when larger) as context and only the rows to score as queries. Items with an identical context share one
    preprocessing; chunks are sized from the measured per-item cost and none is started that would overrun the window
    (FULLCTX_FRAC of the time left). Probabilities are averaged over FULLCTX_SEEDS preprocessing seeds; an item whose
    pass fails or does not finish is absent. Returns {j: (len(held_pos), width) probabilities}."""
    t_x = time.time()
    end = t_x + FULLCTX_FRAC * max(stage_deadline - t_x, 0.0)
    out, widths = {}, heads.widths
    ctxs = {}
    for j in S:
        tr = np.flatnonzero(Y[:, j] >= 0)
        if len(tr) > PFN_MAX_CTX:
            tr = np.sort(np.random.default_rng(1000 * j + 77).choice(tr, PFN_MAX_CTX, replace=False))
        if len(np.unique(Y[tr, j])) >= 2:
            ctxs[j] = tr
    todo = [j for j in S if j in ctxs]
    Xq = Xdf.iloc[held_pos]
    pos, stop, fails = 0, False, 0
    while pos < len(todo) and not stop:
        c = plan_cost() * 1.5 * FULLCTX_SEEDS
        size = min(state["cap"], int((end - time.time()) / c)) if c > 0 else 0
        if size < 1:
            break
        chunk = todo[pos:pos + size]
        pos += len(chunk)
        acc = {}
        try:
            for sd in range(FULLCTX_SEEDS):
                groups = {}
                for j in chunk:
                    groups.setdefault(ctxs[j].tobytes(), []).append(j)
                for js in groups.values():
                    tr = ctxs[js[0]]
                    res = _pfn_fit_shared(Cls, ckpt, Xdf.iloc[vis[tr]], [Y[tr, j] for j in js], Xq, 5000 + sd)
                    for j, (cl, p) in zip(js, res):
                        pr = np.zeros((len(held_pos), widths[j]))
                        pr[:, cl] = p
                        marg = np.bincount(Y[tr, j], minlength=widths[j]) + ALPHA
                        pr = 0.97 * pr + 0.03 * marg / marg.sum()
                        acc[j] = acc.get(j, 0.0) + pr / FULLCTX_SEEDS
                        if os.environ.get("SB_FULLCTX_DBG") == "1":
                            _fullctx_dbg.setdefault("seeds", {}).setdefault(j, {})[sd] = pr.copy()
        except Exception as e:
            _log("tabpfn: full-context chunk failed:", repr(e)[:200])
            _pfn_clear()
            if len(chunk) > 1:                      # e.g. out of memory: retry the chunk in halves
                state["cap"] = max(1, len(chunk) // 2)
                pos -= len(chunk)
            else:
                fails += 1
                stop = fails >= 3
            continue
        finally:
            _free_device()
        fails = 0
        for j, p in acc.items():
            if p.shape == (len(held_pos), widths[j]) and np.isfinite(p).all():
                out[j] = p
    _log(f"tabpfn: full-context pass {len(out)}/{len(S)} items in {time.time() - t_x:.0f}s")
    return out


def _pfn_foldmajor(Cls, ckpt, make, Xdf, vis, held_pos, Y, fold, heads, order, stage_deadline, K, anytime_frac=1.0):
    """TabPFN breadth first. Fold 0 goes over the items in priority order in chunks sized from the measured per-item cost
    (robust estimate over successful calls, x1.3) until the stage deadline; the items done form the covered set S. Each later fold is started only
    if the measured cost says all of S can finish it before the deadline, and a fold that does not complete is dropped,
    so the result is S x k_done complete folds. Returns (oof, test, have, rows_mask) or None."""
    n, nh = len(vis), len(held_pos)
    oof = np.zeros((n, heads.total), np.float32)
    acc = np.zeros((nh, heads.total), np.float64)       # test predictions summed over the completed folds
    have = np.zeros(heads.J, bool)
    widths = heads.widths
    t0 = time.time()
    # items per shared-context call shrink as the context grows beyond the TabICL cap (memory per call)
    ctx_rows = min(PFN_MAX_CTX, int(n * (1.0 - 1.0 / N_FOLDS)))
    chunk0 = PFN_CHUNK if ctx_rows <= TABICL_MAX_CTX else max(4, PFN_CHUNK * TABICL_MAX_CTX // ctx_rows)
    state = {"fails": 0, "calls": 0, "hist": [], "cap": chunk0}

    def blend(j, f, pr):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        tr = np.flatnonzero((fold != f) & (Y[:, j] >= 0))
        if len(tr) > PFN_MAX_CTX:
            tr = np.sort(np.random.default_rng(1000 * j + f).choice(tr, PFN_MAX_CTX, replace=False))
        marg = np.bincount(Y[tr, j], minlength=s1 - s0) + ALPHA
        marg = marg / marg.sum()
        return 0.97 * pr + 0.03 * marg if pr is not None else None, marg

    def run_chunk(f, chunk, rs=0):
        """-> {j: p} for the chunk's items (p = (len(va)+nh, width)), items that failed are absent."""
        va = np.flatnonzero(fold == f)
        res, got = {}, None
        t_c = t_fb = time.time()
        try:
            got = _pfn_item_fold(Cls, ckpt, Xdf, vis, held_pos, Y, fold, f, chunk, widths, rs)
        except Exception as e:
            _log(f"tabpfn: chunk of {len(chunk)} failed, per item:", repr(e)[:200])
            _pfn_clear()
            got = None
            state["cap"] = max(1, len(chunk) // 2)          # a failed (e.g. out-of-memory) chunk halves the chunk size
            state["hist"] = []                              # its time says nothing about the path now in use
            t_fb = time.time()
        finally:
            _free_device()
        if got is not None:
            for j in chunk:
                if j in got:
                    res[j] = blend(j, f, got[j][0])[0]
                else:                                           # < 2 classes in context: marginal
                    res[j] = np.tile(blend(j, f, None)[1], (len(va) + nh, 1))
        else:
            for j in chunk:
                s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
                tr = np.flatnonzero((fold != f) & (Y[:, j] >= 0))
                if len(tr) > PFN_MAX_CTX:
                    tr = np.sort(np.random.default_rng(1000 * j + f).choice(tr, PFN_MAX_CTX, replace=False))
                ytr = Y[tr, j]
                if len(np.unique(ytr)) < 2:
                    res[j] = np.tile(blend(j, f, None)[1], (len(va) + nh, 1))
                    continue
                try:
                    clf = make(f + rs)
                    clf.fit(Xdf.iloc[vis[tr]], ytr)
                    pr = np.zeros((len(va) + nh, s1 - s0))
                    pr[:, clf.classes_] = clf.predict_proba(Xdf.iloc[np.r_[vis[va], held_pos]])
                    res[j] = blend(j, f, pr)[0]
                except Exception as e:
                    _log(f"tabpfn: item {j} failed:", repr(e)[:200])
                    state["fails"] += 1
                    if state["fails"] >= 3:
                        break
                finally:
                    clf = None
                    _free_device()
        dt = time.time() - t_c
        real = [j for j in res]
        if real:
            state["fails"] = 0
        state["calls"] += 1
        if got is not None:                                 # a successful shared call
            state["hist"].append(dt / max(len(chunk), 1))
        elif real and (state["cap"] == 1 or not state["hist"]):     # per-item fallback calls, without the failed attempt
            state["hist"].append((time.time() - t_fb) / len(real))
        return res, dt

    def plan_cost():
        """Per-item cost of the path in use: the larger of the median of the last 3 successful calls (the first call, which
        carries the model load, left out once others exist) and the last one."""
        h = state["hist"][1:] if len(state["hist"]) > 1 else state["hist"]
        return max(float(np.median(h[-3:])), h[-1]) if h else 0.0

    # fold 0: anytime over the priority order
    S, sizes, f0_t = [], [], time.time()
    pos = 0
    while pos < len(order) and state["fails"] < 3:
        left = stage_deadline - time.time()
        if state["calls"] == 0:
            size = 1
        else:
            c = plan_cost() * 1.3
            size = min(state["cap"], int(left / c)) if c > 0 else state["cap"]
        if size < 1:
            break
        chunk = order[pos:pos + size]
        pos += len(chunk)
        res, dt = run_chunk(0, chunk)
        sizes.append(len(chunk))
        va = np.flatnonzero(fold == 0)
        for j, p in res.items():
            if not np.isfinite(p).all():
                continue
            s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
            oof[va, s0:s1] = p[:len(va)]
            acc[:, s0:s1] += p[len(va):]
            if _distill_rec["on"]:
                _distill_rec["d"].setdefault(0, np.zeros((nh, heads.total), np.float32))[:, s0:s1] = p[len(va):]
            have[j] = True
            S.append(j)
        _log(f"tabpfn: fold 0 chunk {len(chunk)} items {dt:.1f}s ({dt / len(chunk):.2f}s/item, {stage_deadline - time.time():.0f}s left)")
    _log(f"tabpfn: fold 0 pass {len(S)}/{len(order)} items in {time.time() - f0_t:.0f}s, chunks {sizes}")
    if not S:
        if STRICT and order and state["fails"] >= 3:
            raise _StrictError("tabpfn: every call failed")
        return None
    k_done = 1
    for f in range(1, K):
        c = plan_cost() * 1.3
        if time.time() + len(S) * c > stage_deadline:
            _log(f"tabpfn: fold {f} needs ~{len(S) * c:.0f}s, {stage_deadline - time.time():.0f}s left; stopping at {k_done} fold(s)")
            break
        va = np.flatnonzero(fold == f)
        pend_oof, pend_acc, complete, sizes, f_t = {}, {}, True, [], time.time()
        pos = 0
        while pos < len(S):
            left = stage_deadline - time.time()
            c = plan_cost() * 1.3
            size = min(state["cap"], int(left / c)) if c > 0 else state["cap"]
            if size < 1:
                complete = False
                break
            chunk = S[pos:pos + size]
            pos += len(chunk)
            res, dt = run_chunk(f, chunk)
            sizes.append(len(chunk))
            if len(res) < len(chunk) or not all(np.isfinite(p).all() for p in res.values()):
                complete = False
                break
            pend_oof.update({j: p[:len(va)] for j, p in res.items()})
            pend_acc.update({j: p[len(va):] for j, p in res.items()})
            _log(f"tabpfn: fold {f} chunk {len(chunk)} items {dt:.1f}s ({stage_deadline - time.time():.0f}s left)")
        _log(f"tabpfn: fold {f} {'done' if complete else 'dropped'}: {len(pend_oof)}/{len(S)} items in {time.time() - f_t:.0f}s, chunks {sizes}")
        if not complete:
            break
        for j in S:
            s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
            oof[va, s0:s1] = pend_oof[j]
            acc[:, s0:s1] += pend_acc[j]
            if _distill_rec["on"]:
                _distill_rec["d"].setdefault(f, np.zeros((nh, heads.total), np.float32))[:, s0:s1] = pend_acc[j]
        k_done += 1
    test = np.zeros((nh, heads.total), np.float32)
    for j in S:
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        test[:, s0:s1] = acc[:, s0:s1] / k_done
    full = {}
    if not NO_FULLCTX and nh > 0 and state["hist"]:
        try:
            full = _pfn_full_pass(Cls, ckpt, Xdf, vis, held_pos, Y, heads, S, plan_cost, state, stage_deadline)
        except Exception as e:
            _log("tabpfn: full-context pass failed:", repr(e)[:200])
            full = {}
    if not NO_ANYTIME and state["hist"]:
        _pfn_repeat_passes(run_chunk, plan_cost, state, S, k_done, fold, heads, oof, test, stage_deadline, anytime_frac)
    if full:
        if os.environ.get("SB_FULLCTX_DBG") == "1":
            _fullctx_dbg["tabpfn"] = (test.copy(), {j: p.copy() for j, p in full.items()}, heads)
        for j, p in full.items():
            s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
            test[:, s0:s1] = _fullctx_merge(test[:, s0:s1], p)
    _log(f"tabpfn: {len(S)}/{len(order)} items x {k_done} fold(s) in {time.time() - t0:.0f}s")
    _pfn_cov.update(k_done=k_done, covered=len(S), order=len(order))
    return oof, test, have, fold < k_done


def _tabicl_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof_mlp, oof_marg, stage_deadline,
                      n_folds=TABICL_FOLDS, by_loss=False, kind="tabicl", min_loss=TABICL_BIG_MIN_LOSS, anytime_frac=1.0, conservative=False):
    """Per item: a pre-trained in-context model (kind "tabicl" = TabICLv2, "tabpfn" = TabPFN-3.5) fitted on n_folds of
    the 5 shared folds; the same call predicts the held-out fold (OOF) and the rows to score (test, averaged over the
    fold models). Items go in priority order until the stage deadline: by how much the MLP beat the marginal (most
    learnable first), or with by_loss by the MLP's remaining OOF loss (most room first, items below min_loss skipped).
    A failed item is skipped; three failures in a row end the stage. Returns (oof, test, have_item, rows_mask) or None."""
    if kind == "tabpfn":
        if not _tabpfn_ready():
            return None
        Cls, ckpt = _tabpfn_state["cls"], _tabpfn_state["ckpt"]
        make = lambda f: Cls(model_path=ckpt, device=str(DEVICE), n_estimators=TABPFN_EST, random_state=f,
                             n_preprocessing_jobs=1, enable_torch_compile=False, show_progress_bar=False)
    else:
        if not _tabicl_ready():
            return None
        Cls, ckpt = _tabicl_state["cls"], _tabicl_state["ckpt"]
        make = lambda f: Cls(n_estimators=TABICL_EST, device=str(DEVICE), random_state=f, allow_auto_download=False,
                             model_path=ckpt, verbose=False)
    Xdf = _ord_frame(frame, given, opts)
    n, nh = len(vis), len(held_pos)
    fast = kind == "tabpfn" and not FIXED and not LEGACY        # fold-major, breadth-first TabPFN
    rows_mask = fold < n_folds
    rank_rows = np.ones(n, bool) if fast else rows_mask      # rows on which the items are ranked and filtered
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((nh, heads.total), np.float32)
    have = np.zeros(heads.J, bool)

    gain = np.full(heads.J, -np.inf)
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = (Y[:, j] >= 0) & rank_rows
        if ok.sum() < TABICL_MIN_ROWS or heads.widths[j] > TABICL_MAX_K:
            continue
        yj = Y[ok, j]
        a = np.log(_scored(oof_mlp[ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), yj]).mean()
        if by_loss:
            if -a >= min_loss:
                gain[j] = -a * _fanout_w.get(j, 1.0)
            continue
        b = np.log(_scored(oof_marg[ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), yj]).mean()
        gain[j] = a - b
    order = [j for j in np.argsort(-gain) if np.isfinite(gain[j])]

    # a call cannot be interrupted, so before the first full-size one, time small calls at two sizes (after a warm-up
    # that pays for loading the weights), fit overhead + per-row cost, and skip the stage if one item would not fit
    if order and not LEGACY:
        j = order[0]
        okr = np.flatnonzero(Y[:, j] >= 0)
        if len(np.unique(Y[okr, j])) >= 2:
            pr_rng = np.random.default_rng(7)
            sizes, times = [], []
            try:
                for m in (PROBE_ROWS, PROBE_ROWS, 4 * PROBE_ROWS):
                    tr = np.sort(pr_rng.choice(okr, min(m, len(okr)), replace=False))
                    q = vis[np.sort(pr_rng.choice(n, min(m, n), replace=False))]
                    t_p = time.time()
                    clf = make(0)
                    clf.fit(Xdf.iloc[vis[tr]], Y[tr, j])
                    clf.predict_proba(Xdf.iloc[q])
                    sizes.append(len(tr) + len(q))
                    times.append(time.time() - t_p)
                    if time.time() > stage_deadline:
                        break
            except Exception as e:
                _log(f"{kind}: probe failed:", repr(e)[:200])
                if kind == "tabpfn":
                    if STRICT:
                        raise _StrictError(f"tabpfn probe failed: {e!r}"[:300])
                    _tabpfn_state["cls"] = None        # treat as unavailable, so TabICL takes the window
                return None
            finally:
                clf = None
                if kind == "tabpfn":
                    _free_device()
            full = min(len(okr) * (N_FOLDS - 1) / N_FOLDS, PFN_MAX_CTX if kind == "tabpfn" else TABICL_MAX_CTX) + n / N_FOLDS + nh
            if len(times) == 3 and sizes[2] > sizes[1]:
                per_row = max(times[2] - times[1], 0.0) / (sizes[2] - sizes[1])
                est = times[2] + per_row * max(full - sizes[2], 0.0)
            else:
                est = np.inf
            if conservative and len(times) == 3:
                # attention cost grows faster than linearly in the context: take the worse of linear and quadratic scaling
                est = max(est, times[2] * (max(full, sizes[2]) / sizes[2]) ** 2)
            _log(f"{kind}: probe {' / '.join(f'{t:.1f}' for t in times)}s, full call est {est:.0f}s, {stage_deadline - time.time():.0f}s in stage")
            if time.time() + est * (1 if fast else n_folds) * 1.3 > stage_deadline:
                _log(f"{kind}: one item would not fit the stage window; skipping")
                return None

    if fast:
        return _pfn_foldmajor(Cls, ckpt, make, Xdf, vis, held_pos, Y, fold, heads, order, stage_deadline,
                              TABPFN_MAX_FOLDS if TABPFN_ADAPTIVE else n_folds, anytime_frac)

    t0, calls, done_items, stop_stage, fails, dt_max = time.time(), 0, 0, False, 0, 0.0
    for j in order:
        if stop_stage:
            break
        per_call = (time.time() - t0) / calls if calls else 0.0
        if not LEGACY:
            per_call = max(per_call, dt_max)       # plan on the slowest call seen, not the average
        if calls and time.time() + per_call * n_folds * 1.3 > stage_deadline:
            break
        if time.time() > stage_deadline:
            break
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        okrow = Y[:, j] >= 0
        acc = np.zeros((nh, s1 - s0), np.float64)
        ok_item = True
        for f in range(n_folds):
            tr = np.flatnonzero((fold != f) & okrow)
            ctx_cap = PFN_MAX_CTX if kind == "tabpfn" else TABICL_MAX_CTX
            if len(tr) > ctx_cap:                             # bound the cost of one in-context call
                tr = np.sort(np.random.default_rng(1000 * j + f).choice(tr, ctx_cap, replace=False))
            va = np.flatnonzero(fold == f)
            ytr = Y[tr, j]
            marg = np.bincount(ytr, minlength=s1 - s0) + ALPHA
            marg = marg / marg.sum()
            if len(np.unique(ytr)) < 2:
                p = np.tile(marg, (len(va) + nh, 1))
            else:
                t_call = time.time()
                try:
                    clf = make(f)
                    clf.fit(Xdf.iloc[vis[tr]], ytr)
                    q = np.r_[vis[va], held_pos]
                    pr = np.zeros((len(q), s1 - s0))
                    pr[:, clf.classes_] = clf.predict_proba(Xdf.iloc[q])
                    p = 0.97 * pr + 0.03 * marg
                except Exception as e:
                    _log(f"{kind}: item {j} failed:", repr(e)[:200])
                    p = None
                clf = None
                if kind == "tabpfn":
                    _free_device()
                if p is None:
                    ok_item = False
                    fails += 1
                    stop_stage = fails >= 3
                    break
                dt = time.time() - t_call
                dt_max = max(dt_max, dt)
                _log(f"{kind}: item {j} fold {f} {dt:.1f}s ({stage_deadline - time.time():.0f}s left in stage)")
                # a call cannot be interrupted: do not start the next fold if it would overrun the stage by > 15 s
                if f < n_folds - 1 and time.time() + dt * 1.1 > stage_deadline + 15:
                    _log(f"{kind}: call took {dt:.0f}s, too slow for the stage window; stopping")
                    ok_item = False
                    stop_stage = True
                    break
            calls += 1
            if not np.isfinite(p).all():
                ok_item = False
                break
            oof[va, s0:s1] = p[:len(va)]
            acc += p[len(va):]
        if ok_item:
            test[:, s0:s1] = acc / n_folds
            if not NO_FULLCTX and nh > 0 and len(np.unique(Y[okrow, j])) >= 2 and time.time() + max(dt_max, 1.0) * 1.3 <= stage_deadline:
                try:
                    trf = np.flatnonzero(okrow)
                    ctx_cap = PFN_MAX_CTX if kind == "tabpfn" else TABICL_MAX_CTX
                    if len(trf) > ctx_cap:
                        trf = np.sort(np.random.default_rng(1000 * j + 77).choice(trf, ctx_cap, replace=False))
                    pf = 0.0
                    for sd in range(FULLCTX_SEEDS):
                        clf = make(77 + sd)
                        clf.fit(Xdf.iloc[vis[trf]], Y[trf, j])
                        pr = np.zeros((nh, s1 - s0))
                        pr[:, clf.classes_] = clf.predict_proba(Xdf.iloc[held_pos])
                        mg = np.bincount(Y[trf, j], minlength=s1 - s0) + ALPHA
                        pf = pf + (0.97 * pr + 0.03 * mg / mg.sum()) / FULLCTX_SEEDS
                    if np.isfinite(pf).all():
                        if os.environ.get("SB_FULLCTX_DBG") == "1":
                            _fullctx_dbg.setdefault(kind, {})[j] = (test[:, s0:s1].copy(), pf.copy())
                        test[:, s0:s1] = _fullctx_merge(test[:, s0:s1], pf)
                except Exception as e:
                    _log(f"{kind}: full-context item {j} failed:", repr(e)[:200])
                clf = None
            have[j] = True
            done_items += 1
            fails = 0
        else:
            oof[rows_mask, s0:s1] = 0.0
    _log(f"{kind}: {done_items}/{len(order)} items in {time.time() - t0:.0f}s ({calls} calls)")
    if STRICT and kind == "tabpfn" and order and done_items == 0 and fails >= 3:
        raise _StrictError("tabpfn: every call failed")
    if not have.any():
        return None
    return oof, test, have, rows_mask


# ----------------------------------------------------------------------------- Kumo-Tabular
# A second in-context tabular model, pre-trained independently of TabPFN, as one more candidate for the per-item blend.
# Bundled pure-python package and weights; the network is loaded from the bundled file only, and its JIT kernels
# (which compile into a cache directory) are replaced by the equivalent eager operations before import.

_kumo_state = {"tried": False, "model": None, "sdm": None, "dev": None, "stage": "untried", "call_fail": False, "autocast_fallback": False}
_KUMO_TRITON = ("sdm._kernels.triton.rmsnorm_cast", "sdm._kernels.triton.segment_multi_reduce")


def _kumo_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "vendor"), os.path.join(here, "weights", "kumo")


def _kumo_present():
    """Cheap check (no import) that the bundled package and the weights for the chosen size exist."""
    vendor, wdir = _kumo_paths()
    return os.path.isdir(os.path.join(vendor, "sdm")) and os.path.isfile(os.path.join(wdir, KUMO_SIZE, "classifier.pt"))


def _kumo_ready():
    """Import the bundled package and build the classifier once; any failure just means the candidate is absent."""
    st = _kumo_state
    if st["tried"]:
        return st["model"] is not None
    st["tried"] = True
    try:
        if not _kumo_present():
            _log("kumo: package or weights not bundled")
            st["stage"] = "absent"
            return False
        vendor, wdir = _kumo_paths()
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        for k in _KUMO_TRITON:
            if k not in sys.modules:
                sys.modules[k] = None
        import sdm
        import sdm.models.kumo.tabular.model as km
        km.download_checkpoint = lambda repo_id, filename, **kw: os.path.join(wdir, filename)
        dev = "cuda" if DEVICE.type == "cuda" else "cpu"
        st["model"] = km.KumoTabular(task="classification", size=KUMO_SIZE, device=dev)
        st["sdm"], st["dev"] = sdm, dev
        st["stage"] = "loaded"
        _log(f"kumo ready: {KUMO_SIZE} on {dev}")
        return True
    except Exception as e:
        st["model"] = None
        st["stage"] = "load_failed"
        _log("kumo unavailable:", repr(e)[:200])
        return False


def _kumo_release():
    _kumo_state["model"] = None
    _free_device()


def _kumo_frame(Xdf):
    """Numeric copy of the feature frame: category codes (NaN where missing) for categorical columns."""
    cols, cats = {}, []
    for c in Xdf.columns:
        if isinstance(Xdf[c].dtype, pd.CategoricalDtype):
            codes = Xdf[c].cat.codes
            cols[c] = codes.astype(float).where(codes >= 0)
            cats.append(c)
        else:
            cols[c] = Xdf[c].astype(float)
    return pd.DataFrame(cols), cats


def _kumo_fold_call(Xk, stypes, ytr_codes, rows_ctx, rows_q, width):
    """One in-context call: context rows with their class codes, query rows unlabelled -> (len(rows_q), width)."""
    st = _kumo_state
    sdm, dev, model = st["sdm"], st["dev"], st["model"]
    df = Xk.iloc[np.r_[rows_ctx, rows_q]].reset_index(drop=True)
    y = np.full(len(df), np.nan)
    y[:len(rows_ctx)] = ytr_codes
    df["__y"] = y
    if dev == "cuda":
        df = df.astype(np.float32)   # no float64 tensors on the cuda device (cpu path unchanged)
    tab = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=dev)
    nc = len(rows_ctx)

    def _go(amp):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x_context=tab[:nc].drop_columns("__y"), y_context=tab[:nc, "__y"],
                         x_query=tab[nc:].drop_columns("__y"), num_estimators=KUMO_EST)

    if dev == "cuda":
        try:
            out = _go(True)
        except Exception as e:
            _log("kumo: autocast call failed, retrying without autocast:", repr(e)[:200])
            st["autocast_fallback"] = True
            _free_device()
            out = _go(False)
    else:
        out = _go(False)
    pk = out.numerical.float().cpu().numpy()
    pr = np.zeros((len(rows_q), width))
    for ci, c in enumerate(out.columns["numerical"]):
        pr[:, int(float(c))] = pk[:, ci]
    return pr


def _kumo_check(kum, done):
    """v58 diagnostic ladder (development only; SB_KPROBE=0 or SB_KSTAGE=off disables): raise _StrictError for exactly the
    Kumo failure stage selected by SB_KSTAGE, so a failed upload says which stage broke on the host."""
    st = _kumo_state
    if not (KPROBE and KSTAGE != "off" and not NO_KUMO and _kumo_present() and done == N_FOLDS):
        return
    if st["stage"] == "untried" and kum is None:
        _kumo_ready()   # the stage window was too short to try it; still report whether the model loads
    stage = st["stage"]
    _log(f"kumo stage={stage} autocast_fallback={st['autocast_fallback']} call_fail={st['call_fail']} kstage={KSTAGE}")
    if kum is not None:
        return
    if KSTAGE == "any":
        raise _StrictError("kumo: no candidate on this instrument")
    if KSTAGE == "load" and stage in ("absent", "load_failed"):
        raise _StrictError(f"kumo: load failed (stage={stage})")
    if KSTAGE == "call" and (stage == "call_failed" or st["call_fail"]):
        raise _StrictError("kumo: call failed")
    if KSTAGE == "time" and stage not in ("absent", "load_failed", "call_failed") and not st["call_fail"]:
        raise _StrictError(f"kumo: no candidate after load and calls were fine (stage={stage})")


def _kumo_extrapolate(r1, t1, r2, t2, r_full):
    """Power law t ~ rows^a through two timed calls (a clamped to [1, 2]), evaluated at r_full rows, never below the larger timing."""
    if r2 <= r1 or t1 <= 0 or t2 <= 0:
        a = 2.0
    else:
        a = float(np.clip(np.log(t2 / t1) / np.log(r2 / r1), 1.0, 2.0))
    return max(t2 * (max(r_full, r2) / r2) ** a, t2)


def _kumo_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof_mlp, stage_deadline, inst_deadline=None):
    """Per item with at most KUMO_MAX_K classes, breadth first over the shared folds: fold 0 over the items in order of the
    MLP's remaining OOF loss, within the share of the window that still leaves room for KUMO_MIN_FOLDS folds of the same
    items; later folds over the items done, each started only if the measured cost says it can finish. A fold that does
    not finish is dropped. Returns (oof, test, have, rows_mask) or None when fewer than KUMO_MIN_FOLDS folds complete."""
    if not _kumo_ready():
        return None
    t0 = time.time()
    n, nh = len(vis), len(held_pos)
    Xk, cats = _kumo_frame(_ord_frame(frame, given, opts))
    stypes = {c: ("categorical" if c in cats else "numerical") for c in Xk.columns}
    stypes["__y"] = "categorical"
    order = []
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Y[:, j] >= 0
        if ok.sum() < KUMO_MIN_ROWS or len(np.unique(Y[ok, j])) < 2 or len(np.unique(Y[ok, j])) > KUMO_MAX_K:
            continue
        loss = -np.log(_scored(oof_mlp[ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), Y[ok, j]]).mean()
        if loss >= TABPFN_MIN_LOSS:
            order.append((loss, j))
    order = [j for _, j in sorted(order, key=lambda x: -x[0])]
    order = order[:KUMO_MAX_ITEMS]
    hard_end = stage_deadline if inst_deadline is None else min(stage_deadline, inst_deadline)
    # a call cannot be interrupted: after a warm-up time two small calls, fit t ~ rows^a (a clamped to [1, 2]) and extrapolate
    # to the real call (capped context + queries); the number of items is then set by what fits KUMO_MIN_FOLDS folds
    est = 0.0
    if order:
        j = order[0]
        okr = np.flatnonzero(Y[:, j] >= 0)
        pr_rng = np.random.default_rng(11)
        full_rows = min(len(okr) * (N_FOLDS - 1) / N_FOLDS, KUMO_MAX_CTX) + n / N_FOLDS + nh
        try:
            sizes, times = [], []
            for m in (PROBE_ROWS // 2, PROBE_ROWS // 2, 2 * PROBE_ROWS):
                tr_p = np.sort(pr_rng.choice(okr, min(m, len(okr)), replace=False))
                q_p = np.sort(pr_rng.choice(n, min(m, n), replace=False))
                if len(np.unique(Y[tr_p, j])) < 2:
                    continue
                t_p = time.time()
                _kumo_fold_call(Xk, stypes, Y[tr_p, j].astype(float), vis[tr_p], vis[q_p], int(heads.widths[j]))
                sizes.append(len(tr_p) + len(q_p))
                times.append(time.time() - t_p)
                if time.time() > hard_end:
                    break
        except Exception as e:
            _log("kumo: probe failed:", repr(e)[:200])
            _kumo_state["call_fail"] = True
            _kumo_state["stage"] = "call_failed"
            _kumo_release()
            return None
        if len(times) < 3:
            _log("kumo: probe incomplete; skipping")
            _kumo_state["stage"] = "timing_skip"
            _kumo_release()
            return None
        est = _kumo_extrapolate(sizes[1], times[1], sizes[2], times[2], full_rows)
        n_fit = int((hard_end - time.time()) / (est * KUMO_SAFETY * KUMO_MIN_FOLDS)) if est > 0 else 0
        _log(f"kumo: probe {times[0]:.1f} / {times[1]:.1f} / {times[2]:.1f}s at {sizes} rows, full call ({full_rows:.0f} rows) est {est:.1f}s, "
             f"{hard_end - time.time():.0f}s in stage, {n_fit} item(s) fit {KUMO_MIN_FOLDS} folds")
        if n_fit < min(KUMO_MIN_ITEMS, len(order)):
            _log("kumo: too few calls would fit the stage window; skipping")
            _kumo_state["stage"] = "timing_skip"
            _kumo_release()
            return None
        order = order[:n_fit]
    oof = np.zeros((n, heads.total), np.float32)
    acc = np.zeros((nh, heads.total), np.float64)
    have = np.zeros(heads.J, bool)
    costs, fails = [], 0

    def run(j, f):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        tr = np.flatnonzero((fold != f) & (Y[:, j] >= 0))
        if len(tr) > KUMO_MAX_CTX:
            tr = np.sort(np.random.default_rng(1000 * j + f).choice(tr, KUMO_MAX_CTX, replace=False))
        va = np.flatnonzero(fold == f)
        marg = np.bincount(Y[tr, j], minlength=s1 - s0) + ALPHA
        marg = marg / marg.sum()
        t_c = time.time()
        pr = _kumo_fold_call(Xk, stypes, Y[tr, j].astype(float), vis[tr], np.r_[vis[va], held_pos], s1 - s0)
        p = 0.97 * pr + 0.03 * marg
        if not np.isfinite(p).all():
            raise ValueError("non-finite output")
        costs.append(time.time() - t_c)
        return p, va

    def cost():
        return max(costs[-3:]) if costs else 0.0


    def can_start():
        """A call cannot be interrupted: start it only if it would end well inside the stage and the instrument deadline."""
        pred = max(cost(), est)
        return time.time() + pred * KUMO_SAFETY < hard_end

    S = []
    f0_end = t0 + (stage_deadline - t0) / (1.0 + 1.3 * (KUMO_MIN_FOLDS - 1))   # leaves room for the safety factor on the later folds
    try:
        for j in order:
            if time.time() + 1.3 * max(cost(), est) > f0_end or not can_start():
                break
            try:
                p, va = run(j, 0)
            except Exception as e:
                _log(f"kumo: item {j} failed:", repr(e)[:200])
                _kumo_state["call_fail"] = True
                _free_device()
                fails += 1
                if fails >= 3:
                    break
                continue
            fails = 0
            s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
            oof[va, s0:s1] = p[:len(va)]
            acc[:, s0:s1] += p[len(va):]
            S.append(j)
        _log(f"kumo: fold 0 {len(S)}/{len(order)} items in {time.time() - t0:.0f}s ({cost():.2f}s/call)")
        k_done = 1 if S else 0
        for f in range(1, KUMO_MAX_FOLDS):
            if not S or time.time() + len(S) * 1.3 * float(np.mean(costs)) > hard_end:
                break
            pend, complete = {}, True
            for j in S:
                if not can_start():
                    complete = False
                    break
                try:
                    pend[j] = run(j, f)
                except Exception as e:
                    _log(f"kumo: item {j} fold {f} failed:", repr(e)[:200])
                    _kumo_state["call_fail"] = True
                    _free_device()
                    complete = False
                    break
            if not complete:
                _log(f"kumo: fold {f} dropped")
                break
            for j, (p, va) in pend.items():
                s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
                oof[va, s0:s1] = p[:len(va)]
                acc[:, s0:s1] += p[len(va):]
            k_done += 1
    finally:
        _kumo_release()
    _log(f"kumo: {len(S)} items x {k_done} fold(s) in {time.time() - t0:.0f}s")
    if k_done < KUMO_MIN_FOLDS:
        _kumo_state["stage"] = "call_failed" if _kumo_state["call_fail"] else "timing_skip"
        return None
    _kumo_state["stage"] = "ok"
    for j in S:
        have[j] = True
    test = (acc / k_done).astype(np.float32)
    return oof, test, have, fold < k_done


# ----------------------------------------------------------------------------- support caps

# ----------------------------------------------------------------------------- distillation with transduction

def _train_distill(Xall, S, Yh, WS, WH, Xva, Yva, heads, cfg, deadline, seed):
    """Student MLP on rows that carry a soft target (teacher probabilities, flat layout), a hard label, or both.
    Cell loss = WS * soft cross-entropy + WH * hard NLL; the epoch is chosen on the hard NLL of the validation rows."""
    hidden, drop, wd, lr, max_epochs, depth = cfg
    torch.manual_seed(seed)
    net = _make_net(Xall.shape[1], heads.total, hidden, drop, depth).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    X_t, S_t = torch.tensor(Xall, device=DEVICE), torch.tensor(S, device=DEVICE)
    Y_t = torch.tensor(Yh, device=DEVICE)
    WS_t, WH_t = torch.tensor(WS, device=DEVICE), torch.tensor(WH, device=DEVICE)
    Xva_t, Yva_t = torch.tensor(Xva, device=DEVICE), torch.tensor(Yva, device=DEVICE)
    n = len(Xall)
    bs = 512 if n > 4000 else 256
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def hard_nll(lp, Y):
        valid = Y >= 0
        nll = -lp.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    best, best_state, bad, patience = float("inf"), None, 0, 6
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            lp = heads.logp(net(X_t[b]))
            sb = S_t[b][:, heads.gather] * heads.mask
            soft = -(sb * lp).sum(2)
            hard = -lp.gather(2, Y_t[b].clamp(min=0).unsqueeze(2)).squeeze(2)
            w_s, w_h = WS_t[b], WH_t[b]
            loss = (w_s * soft + w_h * hard).sum() / (w_s + w_h).sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = hard_nll(heads.logp(net(Xva_t)), Yva_t).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time.time() > deadline:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _distill_blend_teacher(oof, test, ft, Y, heads, n):
    """Test-only teacher: per-item weighted blend of marginal, linear and mlp (weights fitted on the OOF rows); its rows-to-score
    probabilities for fold f come from the fold-f models only (ft[f][c])."""
    names = ["marginal", "linear", "mlp"]
    t_oof = np.zeros((n, heads.total), np.float32)
    t_test = np.zeros_like(test["mlp"])
    t_fold = {f: np.zeros_like(test["mlp"]) for f in ft}
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Y[:, j] >= 0
        w = _fit_weights([oof[c][ok, s] for c in names], Y[ok, j]) if ok.sum() >= 20 else np.eye(len(names))[1]
        t_oof[:, s] = sum(wi * oof[c][:, s] for wi, c in zip(w, names))
        t_test[:, s] = sum(wi * test[c][:, s] for wi, c in zip(w, names))
        for f in ft:
            t_fold[f][:, s] = w[0] * test["marginal"][:, s] + w[1] * ft[f]["linear"][:, s] + w[2] * ft[f]["mlp"][:, s]
    return t_oof, t_test, np.ones(heads.J, bool), np.ones(n, bool), t_fold


def _distill_candidate(Xv, Xh, Y, fold, heads, teacher, stage_deadline):
    """Student MLP distilled from an in-context teacher with transduction. Teacher = (oof, test, have, rows): out-of-fold
    probabilities on the visible rows of the folds in `rows`, probabilities on the rows to score, the items it covers.
    Student for fold f trains on the other folds' rows (soft teacher OOF target where the teacher has one, plus the hard
    label) and on the rows to score (soft teacher target only, no labels); fold f's labels are used only to pick the epoch.
    The rows to score carry, for student fold f, the teacher's probabilities from its own fold-f call (context without fold f), so
    nothing about fold f's labels reaches the student except through the epoch choice. Returns (oof, test, have, rows) in the partial-candidate layout, or None (all folds of the teacher or nothing)."""
    t_oof, t_test, have, rows, t_fold = teacher
    if DISTILL_FAIL:
        raise RuntimeError("injected distill failure")
    n, nh = len(Xv), len(Xh)
    k = len(np.unique(fold[rows]))
    if k < 2 or nh == 0 or not have.any() or any(f not in t_fold for f in range(k)):
        return None
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((nh, heads.total), np.float32)
    cell_of = np.repeat(np.arange(heads.J), heads.widths)               # flat column -> item
    soft_cols = have[cell_of]
    S_tr = np.where(soft_cols[None, :] & rows[:, None], t_oof, 0.0).astype(np.float32)
    has_soft = (rows[:, None] & have[None, :]) if DISTILL_TRAIN_SOFT else np.zeros((n, heads.J), bool)   # (n, J) train cells with a teacher target
    a = DISTILL_ALPHA
    Xall = np.vstack([Xv, Xh]).astype(np.float32)
    t0 = time.time()
    for f in range(k):
        S_te = np.where(soft_cols[None, :], t_fold[f], 0.0).astype(np.float32)   # teacher whose context excluded fold f
        S = np.vstack([S_tr, S_te])
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        idx = np.r_[tr, n + np.arange(nh)]
        Yh = np.r_[Y[tr], -np.ones((nh, heads.J), Y.dtype)]
        ws_tr = a * has_soft[tr]
        wh_tr = np.where(has_soft[tr], 1.0 - a, 1.0) * (Y[tr] >= 0)
        WS = np.vstack([ws_tr, np.tile(have[None, :], (nh, 1)).astype(np.float64)]).astype(np.float32)
        WH = np.vstack([wh_tr, np.zeros((nh, heads.J))]).astype(np.float32)
        net = _train_distill(Xall[idx], S[idx], Yh, WS, WH, Xv[va], Y[va], heads, DISTILL_CFG, stage_deadline, seed=700 + f)
        oof[va] = _predict_flat(net, Xv[va], heads)
        test += _predict_flat(net, Xh, heads) / k
        del net
        per = (time.time() - t0) / (f + 1)
        _log(f"distill fold {f} ({time.time() - t0:.0f}s)")
        if f < k - 1 and time.time() + 1.3 * per > stage_deadline:
            _log("distill: next fold does not fit, dropped")
            return None
    if not (np.isfinite(oof[rows]).all() and np.isfinite(test).all()):
        return None
    return oof, test, have.copy(), rows.copy()


def _lca_candidate(Xv, Y, Xh, Yoh, fold, heads, stage_deadline):
    """Latent-class mixture, 5 shared folds x LCA_SEEDS; None if it cannot finish every fold before the deadline."""
    t0 = time.time()
    ob = np.zeros((len(Xv), heads.total), np.float32)
    tb = np.zeros((len(Xh), heads.total), np.float32)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        base = np.log(_marginal_flat(Yoh[tr], heads))
        for sd in range(LCA_SEEDS):
            net = _train_lca(Xv[tr], Y[tr], Xv[va], Y[va], heads, base, stage_deadline, seed=500 + 10 * sd + f)
            ob[va] += _predict_lca(net, Xv[va], heads) / LCA_SEEDS
            tb += _predict_lca(net, Xh, heads) / (LCA_SEEDS * N_FOLDS)
            del net
        per = (time.time() - t0) / (f + 1)
        _log(f"lca fold {f} ({time.time() - t0:.0f}s)")
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    return (ob, tb) if np.isfinite(ob).all() and np.isfinite(tb).all() else None


def _wide_candidate(Xv, Y, Xh, fold, heads, cfg, seeds, stage_deadline, seed0, tag):
    """Seed-averaged wide trunk over the 5 shared folds; None if it cannot finish every fold before the deadline."""
    t0 = time.time()
    h, d_, wd_, lr_, ep_, dep_ = cfg
    ob = np.zeros((len(Xv), heads.total), np.float32)
    tb = np.zeros((len(Xh), heads.total), np.float32)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        for sd in range(seeds):
            net = _train(Xv[tr], Y[tr], Xv[va], Y[va], heads, h, d_, wd_, lr_, ep_, stage_deadline, seed=seed0 + 100 * sd + f, depth=dep_)
            ob[va] += _predict_flat(net, Xv[va], heads) / seeds
            tb += _predict_flat(net, Xh, heads) / (seeds * N_FOLDS)
            del net
        per = (time.time() - t0) / (f + 1)
        _log(f"{tag} fold {f} ({time.time() - t0:.0f}s)")
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    return (ob, tb) if np.isfinite(ob).all() and np.isfinite(tb).all() else None


MR_CFG = (256, 0.25, 1e-4, 2e-3)    # hidden, dropout, wd, lr (same trunk as the small MLP)
MR_PRE_EPOCHS, MR_FT_EPOCHS = 12, 40
MR_SHARE = float(os.environ.get("SB_MR_SHARE", 0.12))   # share of the remaining budget for the mlp_mr stage (distill holds another 12% of the foundation window)


def _mr_train(Xtr, Ytr, Ytr_oh, Xva, Yva, heads, deadline, seed):
    """Mask-and-recover pretraining on the fitting rows only, then fine-tuning on the full-mask objective.
    Input = GIVEN features + one-hot of the visible answers + per-item observed indicators."""
    h, d, wd, lr = MR_CFG
    J, tot = heads.J, heads.total
    torch.manual_seed(seed)
    net = _make_net(Xtr.shape[1] + tot + J, tot, h, d, 2).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt, Yt, Ot = (torch.tensor(a, device=DEVICE) for a in (Xtr, Ytr, Ytr_oh))
    Xv_, Yv_ = torch.tensor(Xva, device=DEVICE), torch.tensor(Yva, device=DEVICE)
    n = len(Xtr)
    bs = 512 if n > 4000 else 256
    gen = torch.Generator(device="cpu").manual_seed(seed)
    owner = torch.tensor(np.repeat(np.arange(J), heads.widths), device=DEVICE)

    def loss_of(lp, Y, w):
        valid = (Y >= 0) & w
        nll = -lp.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    def build(xb, ob, obs):                                  # obs: (B, J) bool, True = answer visible
        return torch.cat([xb, ob * obs[:, owner].float(), obs.float()], 1)

    def none_obs(m):
        return torch.zeros(m, J, dtype=torch.bool, device=DEVICE)

    for _ in range(MR_PRE_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            keep = torch.rand(len(b), 1, device=DEVICE) * 0.8
            obs = torch.rand(len(b), J, device=DEVICE) < keep
            loss = loss_of(heads.logp(net(build(Xt[b], Ot[b], obs))), Yt[b], ~obs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if time.time() > deadline:
            break
    for g in opt.param_groups:
        g["lr"] = lr * 0.5
    Xva_in = build(Xv_, torch.zeros(len(Xva), tot, device=DEVICE), none_obs(len(Xva)))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(MR_FT_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            inp = build(Xt[b], torch.zeros(len(b), tot, device=DEVICE), none_obs(len(b)))
            loss = loss_of(heads.logp(net(inp)), Yt[b], torch.ones(len(b), J, dtype=torch.bool, device=DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_of(heads.logp(net(Xva_in)), Yv_, torch.ones(len(Xva), J, dtype=torch.bool, device=DEVICE)).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
        if time.time() > deadline:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _mr_predict(net, X, heads):
    pad = np.zeros((len(X), heads.total + heads.J), np.float32)
    return _predict_flat(net, np.hstack([X, pad]), heads)


def _mr_candidate(Xv, Y, Xh, Yoh, fold, heads, stage_deadline):
    """Per fold, pretraining sees only the rows outside the scored fold; None if it cannot finish every fold in time."""
    t0 = time.time()
    if os.environ.get("SB_MR_FAIL") == "1":
        raise RuntimeError("injected")
    ob = np.zeros((len(Xv), heads.total), np.float32)
    tb = np.zeros((len(Xh), heads.total), np.float32)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        net = _mr_train(Xv[tr], Y[tr], Yoh[tr], Xv[va], Y[va], heads, stage_deadline, seed=700 + f)
        ob[va] = _mr_predict(net, Xv[va], heads)
        tb += _mr_predict(net, Xh, heads) / N_FOLDS
        del net
        per = (time.time() - t0) / (f + 1)
        _log(f"mlp_mr fold {f} ({time.time() - t0:.0f}s)")
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    return (ob, tb) if np.isfinite(ob).all() and np.isfinite(tb).all() else None


def _round_columns(given, rec, nvals):
    """Index of a GIVEN item that records the survey round / wave (a small categorical), else None."""
    for g in given:
        if any(t in g.lower() for t in ("wave", "round")) and 2 <= len(nvals[g]) <= 4:
            return g
    return None


def _round_features(frame, X, g, idx, nlev):
    code = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
    parts = [X * (code == k)[:, None].astype(np.float32) for k in range(nlev)]
    return np.hstack([X] + parts)


def _round_candidate(Xr_v, Y, Xr_h, fold, heads, stage_deadline):
    t0 = time.time()
    h, d_, wd_, lr_, ep_, dep_ = ROUND_CFG
    ob = np.zeros((len(Xr_v), heads.total), np.float32)
    tb = np.zeros((len(Xr_h), heads.total), np.float32)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        net = _train(Xr_v[tr], Y[tr], Xr_v[va], Y[va], heads, h, d_, wd_, lr_, ep_, stage_deadline, seed=700 + f, depth=dep_)
        ob[va] = _predict_flat(net, Xr_v[va], heads)
        tb += _predict_flat(net, Xr_h, heads) / N_FOLDS
        del net
        per = (time.time() - t0) / (f + 1)
        _log(f"mlp_round fold {f} ({time.time() - t0:.0f}s)")
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    return (ob, tb) if np.isfinite(ob).all() and np.isfinite(tb).all() else None


def _train_ph(Xtr, Ytr, Xsel, Ysel, Xva, Xh, heads, stage_deadline, seed):
    """Train one shared trunk and, at every epoch in PH_GRID, record each head's loss on a held-back selection set and
    its predictions on the OOF and held-out rows. Returns (sel_loss[grid, J], oof[grid, n_va, total], test[grid, nh, total])
    so the caller can pick, per head, the grid epoch that generalised best. None if the first checkpoint overruns."""
    torch.manual_seed(seed)
    h, d_, wd_, lr_, dep_ = PH_CFG
    net = _make_net(Xtr.shape[1], heads.total, h, d_, dep_).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_, weight_decay=wd_)
    Xtr_t, Ytr_t = torch.tensor(Xtr, device=DEVICE), torch.tensor(Ytr, device=DEVICE)
    Xsel_t, Ysel_t = torch.tensor(Xsel, device=DEVICE), torch.tensor(Ysel, device=DEVICE)
    n, bs = len(Xtr), 512 if len(Xtr) > 4000 else 256
    gen = torch.Generator(device="cpu").manual_seed(seed)
    grid = PH_GRID
    sel_loss = np.full((len(grid), heads.J), np.inf, np.float32)
    oof = np.zeros((len(grid), len(Xva), heads.total), np.float32)
    test = np.zeros((len(grid), len(Xh), heads.total), np.float32)
    gi, ep = 0, 0
    while gi < len(grid) and ep < grid[-1]:
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            Yb = Ytr_t[b]
            lp = heads.logp(net(Xtr_t[b]))
            valid = Yb >= 0
            nll = -lp.gather(2, Yb.clamp(min=0).unsqueeze(2)).squeeze(2)
            loss = (nll * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ep += 1
        if ep == grid[gi]:
            net.eval()
            with torch.no_grad():
                lps = heads.logp(net(Xsel_t))
                v = (Ysel_t >= 0)
                nll = -lps.gather(2, Ysel_t.clamp(min=0).unsqueeze(2)).squeeze(2)
                sel_loss[gi] = ((nll * v).sum(0) / v.sum(0).clamp(min=1)).float().cpu().numpy()
            oof[gi] = _predict_flat(net, Xva, heads)
            test[gi] = _predict_flat(net, Xh, heads)
            gi += 1
            if gi == 1 and time.time() > stage_deadline:
                return None
    return sel_loss[:gi], oof[:gi], test[:gi]


def _ph_candidate(Xv, Y, Xh, fold, heads, stage_deadline):
    """Per-head checkpoint selection: one trunk per fold, each head taking the epoch that minimised its own selection
    loss, so 213 heterogeneous heads are not forced to share one early stop. None if it cannot finish every fold."""
    t0 = time.time()
    ob = np.zeros((len(Xv), heads.total), np.float32)
    tb = np.zeros((len(Xh), heads.total), np.float32)
    rng = np.random.default_rng(321)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        msk = rng.random(len(tr)) < PH_SEL
        inner, sel = tr[~msk], tr[msk]
        if len(sel) < 50 or len(inner) < 50:
            return None
        r = _train_ph(Xv[inner], Y[inner], Xv[sel], Y[sel], Xv[va], Xh, heads, stage_deadline, seed=1500 + f)
        if r is None:
            return None
        sel_loss, oof_g, test_g = r
        best = sel_loss.argmin(0)
        for j in range(heads.J):
            s = slice(heads.offsets[j], heads.offsets[j + 1])
            ob[va, s] = oof_g[best[j], :, s]
            tb[:, s] += test_g[best[j], :, s] / N_FOLDS
        per = (time.time() - t0) / (f + 1)
        _log(f"ph fold {f} ({time.time() - t0:.0f}s)")
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    return (ob, tb) if np.isfinite(ob).all() and np.isfinite(tb).all() else None


def _cap_flags(frame, given, idx, rec, pred, vis, Y, heads):
    """Per gated item, the rows (of `frame`) that fall in a GIVEN stratum of >= CAP_MIN_ROWS visible respondents where
    the gated slot never occurs (`never`) or is the only answer (`always`). Returns (never, always) bool (rows, J)."""
    n = len(frame)
    never = np.zeros((n, heads.J), bool)
    always = np.zeros((n, heads.J), bool)
    gated = [j for j, p in enumerate(pred) if rec[p].get("gate")]
    if not gated:
        return never, always
    known = (Y[:, gated] >= 0)
    isg = np.stack([Y[:, j] == heads.widths[j] - 1 for j in gated], 1) & known
    for g in given:
        codes = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
        cv = codes[vis]
        L = len(idx[g])
        C = np.zeros((len(vis), L), np.float32)
        ok = cv >= 0
        C[np.flatnonzero(ok), cv[ok]] = 1.0
        tot = C.T @ known.astype(np.float32)
        gat = C.T @ isg.astype(np.float32)
        for col, j in enumerate(gated):
            lv_never = np.flatnonzero((tot[:, col] >= CAP_MIN_ROWS) & (gat[:, col] == 0))
            lv_always = np.flatnonzero((tot[:, col] >= CAP_MIN_ROWS) & (gat[:, col] == tot[:, col]))
            if len(lv_never):
                never[:, j] |= np.isin(codes, lv_never)
            if len(lv_always):
                always[:, j] |= np.isin(codes, lv_always)
    both = never & always
    never &= ~both
    always &= ~both
    return never, always


def _apply_caps(M, rows, flags, heads):
    """Cap, in place, the gated slot (never-strata) or the substantive mass (always-strata) of the vectors in M.
    `rows` are the frame positions of M's rows."""
    never, always = flags
    nv, al = never[rows], always[rows]
    for j in np.flatnonzero(nv.any(0) | al.any(0)):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        sub = M[:, s0:s1 - 1]
        ssum = sub.sum(1)
        sel = nv[:, j] & (M[:, s1 - 1] > CAP_Q)
        if sel.any():
            scale = np.where(ssum[sel] > 1e-12, (1.0 - CAP_Q) / np.maximum(ssum[sel], 1e-12), 0.0)
            flat = (1.0 - CAP_Q) / (s1 - s0 - 1)
            M[sel, s0:s1 - 1] = np.where(ssum[sel, None] > 1e-12, sub[sel] * scale[:, None], flat)
            M[sel, s1 - 1] = CAP_Q
        sel = al[:, j] & (ssum > CAP_Q)
        if sel.any():
            M[sel, s0:s1 - 1] = sub[sel] * (CAP_Q / ssum[sel])[:, None]
            M[sel, s1 - 1] = 1.0 - CAP_Q
    return M


def _cap_core(frame, given, idx, rec, pred, vis, Y, heads, sub=None, out_rows=None, ho=None, ab=None, cache=None, min_rows=CAP_MIN_ROWS):
    """Support-cap strata derived on the visible rows `vis[sub]` (all when None). Returns (never, always, qn, qa, pairs)
    over the frame rows `out_rows` (all rows when None). With `ab` (fitted rule posteriors) qn / qa hold the residual
    left on the capped side of each flagged cell; with `ho` (positions in `vis` of held-out rows) pairs collects, per
    rule type, arrays (n derivation rows, m covered held-out rows, v violations among them) per item x stratum."""
    J = heads.J
    out = np.arange(len(frame)) if out_rows is None else np.asarray(out_rows)
    never = np.zeros((len(out), J), bool)
    always = np.zeros((len(out), J), bool)
    qn = np.ones((len(out), J), np.float32) if ab is not None else None
    qa = np.ones((len(out), J), np.float32) if ab is not None else None
    pairs = {"cap_never": [], "cap_always": [], "n_cap_never": 0, "n_cap_always": 0}
    gated = [j for j, p in enumerate(pred) if rec[p].get("gate")]
    if not gated:
        return never, always, qn, qa, pairs
    cache = {} if cache is None else cache
    tr_rows = vis if sub is None else vis[sub]
    Ys = Y if sub is None else Y[sub]
    known = (Ys[:, gated] >= 0)
    isg = np.stack([Ys[:, j] == heads.widths[j] - 1 for j in gated], 1) & known
    if ho is not None:
        known_h = (Y[ho][:, gated] >= 0)
        isg_h = np.stack([Y[ho][:, j] == heads.widths[j] - 1 for j in gated], 1) & known_h

    def onehot(cv, L):
        C = np.zeros((len(cv), L), np.float32)
        ok = cv >= 0
        C[np.flatnonzero(ok), cv[ok]] = 1.0
        return C

    def res(typ, n):
        return _bb_res(ab[typ], n) if ab.get(typ) is not None else np.full(len(n), CAP_Q)

    for g in given:
        if g not in cache:
            cache[g] = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
        codes = cache[g]
        L = len(idx[g])
        C = onehot(codes[tr_rows], L)
        tot = C.T @ known.astype(np.float32)
        gat = C.T @ isg.astype(np.float32)
        if ho is not None:
            Ch = onehot(codes[vis[ho]], L)
            toth = Ch.T @ known_h.astype(np.float32)
            gath = Ch.T @ isg_h.astype(np.float32)
        co = codes[out]
        for col, j in enumerate(gated):
            lv_never = np.flatnonzero((tot[:, col] >= min_rows) & (gat[:, col] == 0))
            lv_always = np.flatnonzero((tot[:, col] >= min_rows) & (gat[:, col] == tot[:, col]))
            pairs["n_cap_never"] += len(lv_never)
            pairs["n_cap_always"] += len(lv_always)
            for typ, lv in (("cap_never", lv_never), ("cap_always", lv_always)):
                if ho is not None and len(lv):
                    m = toth[lv, col]
                    v = gath[lv, col] if typ == "cap_never" else toth[lv, col] - gath[lv, col]
                    k = m > 0
                    if k.any():
                        pairs[typ].append((tot[lv, col][k].astype(np.float64), m[k].astype(np.float64), v[k].astype(np.float64)))
                if not len(lv) or not len(out):
                    continue
                msk = np.isin(co, lv)
                (never if typ == "cap_never" else always)[:, j] |= msk
                if ab is not None:
                    lev = np.ones(L, np.float32)
                    lev[lv] = res(typ, tot[lv, col])
                    q = qn if typ == "cap_never" else qa
                    q[:, j] = np.where(msk, np.minimum(q[:, j], lev[co]), q[:, j])
    both = never & always
    never &= ~both
    always &= ~both
    return never, always, qn, qa, pairs


def _apply_caps_q(M, flags, heads):
    """As _apply_caps, with a residual per cell: flags = (never, always, qn, qa) aligned with the rows of M."""
    never, always, qn, qa = flags
    for j in np.flatnonzero(never.any(0) | always.any(0)):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        sub = M[:, s0:s1 - 1]
        ssum = sub.sum(1)
        sel = never[:, j] & (M[:, s1 - 1] > qn[:, j])
        if sel.any():
            q = qn[sel, j].astype(np.float64)
            scale = np.where(ssum[sel] > 1e-12, (1.0 - q) / np.maximum(ssum[sel], 1e-12), 0.0)
            flat = (1.0 - q) / (s1 - s0 - 1)
            M[sel, s0:s1 - 1] = np.where(ssum[sel, None] > 1e-12, sub[sel] * scale[:, None], flat[:, None])
            M[sel, s1 - 1] = q
        sel = always[:, j] & (ssum > qa[:, j])
        if sel.any():
            q = qa[sel, j].astype(np.float64)
            M[sel, s0:s1 - 1] = sub[sel] * (q / ssum[sel])[:, None]
            M[sel, s1 - 1] = 1.0 - q
    return M


def _fit_bb(pairs):
    """Beta-binomial posterior for one rule type. A rule derived on n rows without an exception leaves a violation rate
    q ~ Beta(a, b + n); on m covered held-out rows the violations v are then BetaBinomial(m; a, b + n). Maximum marginal
    likelihood over (a, b), a in [0.05, 50], b in [1, 1e6]. Returns (a, b, instances, rows, violations) or None when
    there are < 5 instances or no violation at all."""
    if not pairs:
        return None
    from scipy.optimize import minimize
    from scipy.special import betaln
    n = np.concatenate([p[0] for p in pairs])
    m = np.concatenate([p[1] for p in pairs])
    v = np.concatenate([p[2] for p in pairs])
    if len(n) < 5 or v.sum() == 0:
        return None

    def nll(z):
        a, b = np.exp(z)
        return -(betaln(v + a, m - v + b + n) - betaln(a, b + n)).sum() / len(n)

    lo, hi = np.log([0.05, 1.0]), np.log([50.0, 1e6])
    best = None
    for z0 in ((np.log(0.5), np.log(100.0)), (np.log(1.0), np.log(3000.0)), (np.log(0.1), np.log(10.0))):
        r = minimize(nll, np.array(z0), method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x)
    return float(a), float(b), len(n), float(m.sum()), float(v.sum())


RULE_TYPES = ("cap_never", "cap_always", "gate_asked", "gate_skip")


def _xfit_rules(frame, given, idx, rec, pred, heads, vis, held_pos, Y, fold, covered, given_set, use_caps, use_bb):
    """Rules judged out of fold. For every completed fold f the gate sides and support strata are derived on the
    covered visible rows outside f and applied to f's out-of-fold rows; the held-out violations of those rules give
    per rule type a fitted posterior residual. Returns a dict: per-fold gate rules, per-OOF-row cap flags (rows in the
    order of `vis[covered]`), test cap flags, the fitted (a, b) per rule type, and the pairs log."""
    cov = np.flatnonzero(covered)
    fo = fold[cov]
    folds = [int(f) for f in np.unique(fo)]
    cache = {}
    gate_rules, pairs = {}, {t: [] for t in RULE_TYPES}
    cnt_gate, cnt_cap = {}, {}
    for f in folds:
        tr = np.flatnonzero(covered & (fold != f))
        ho = np.flatnonzero(covered & (fold == f))
        fr = len(tr) / len(vis)                       # thresholds scale with the share of visible rows the rule is derived on
        gmin, cmin = int(np.ceil(GATE_MIN_ROWS * fr)), int(np.ceil(CAP_MIN_ROWS * fr))
        trust, nside = _gate_rules(frame, vis, Y, pred, rec, heads, given_set, tr, gmin)
        cnt_gate[f] = sum(int(t[0]) + int(t[1]) for t in trust.values())
        gate_rules[f] = (trust, nside)
        for j, n in enumerate(pred):
            g = rec[n].get("gate")
            if n not in nside:
                continue
            asked = frame[g["parent"]].iloc[vis[ho]].isin(g["observed_if"]).to_numpy()
            known = Y[ho, j] >= 0
            isg = Y[ho, j] == heads.widths[j] - 1
            for typ, side, ok, viol in (("gate_asked", 0, asked & known, isg), ("gate_skip", 1, ~asked & known, ~isg)):
                if trust[n][side] and ok.any():
                    pairs[typ].append((np.array([nside[n][side]], np.float64), np.array([ok.sum()], np.float64),
                                       np.array([(ok & viol).sum()], np.float64)))
        if use_caps:
            *_, cp = _cap_core(frame, given, idx, rec, pred, vis, Y, heads, sub=tr, out_rows=vis[:0], ho=ho, cache=cache, min_rows=cmin)
            cnt_cap[f] = (cp["n_cap_never"], cp["n_cap_always"])
            for t in ("cap_never", "cap_always"):
                pairs[t] += cp[t]
    fits = {t: _fit_bb(pairs[t]) for t in RULE_TYPES}
    ab = {t: (fits[t][:2] if (use_bb and fits[t] is not None) else None) for t in RULE_TYPES}
    ab_arg = ab if use_bb and any(v is not None for v in ab.values()) else None
    J = heads.J
    oof_flags = None
    test_flags = None
    if use_caps:
        nv = np.zeros((len(cov), J), bool)
        al = np.zeros((len(cov), J), bool)
        qn = np.ones((len(cov), J), np.float32)
        qa = np.ones((len(cov), J), np.float32)
        for f in folds:
            tr = np.flatnonzero(covered & (fold != f))
            pos = np.flatnonzero(fo == f)
            r = _cap_core(frame, given, idx, rec, pred, vis, Y, heads, sub=tr, out_rows=vis[cov[pos]], ab=ab_arg, cache=cache,
                          min_rows=int(np.ceil(CAP_MIN_ROWS * len(tr) / len(vis))))
            nv[pos], al[pos] = r[0], r[1]
            if ab_arg is not None:
                qn[pos], qa[pos] = r[2], r[3]
        oof_flags = (nv, al, qn, qa)
        r = _cap_core(frame, given, idx, rec, pred, vis, Y, heads, out_rows=held_pos, ab=ab_arg, cache=cache)
        test_flags = (r[0], r[1], r[2], r[3])
        cnt_cap["all"] = (r[4]["n_cap_never"], r[4]["n_cap_always"])
    if VERBOSE:
        all_g = sum(int(t[0]) + int(t[1]) for t in getattr(_gate_override, "trust", {}).values())
        _log(f"xfit rule sets: gate sides trusted all-rows {all_g} vs per fold {[cnt_gate[f] for f in folds]}; "
             f"cap strata (never, always) all-rows {cnt_cap.get('all')} vs per fold {[cnt_cap.get(f) for f in folds]}")
        for t in RULE_TYPES:
            ps = pairs[t]
            ni = int(sum(len(p[0]) for p in ps))
            mm = float(sum(p[1].sum() for p in ps))
            vv = float(sum(p[2].sum() for p in ps))
            fit = fits[t]
            if fit is None:
                _log(f"xfit {t}: {ni} instances, {mm:.0f} held-out rows, {vv:.0f} violations; no fit (fixed residual)")
            else:
                res = lambda n: float(_bb_res((fit[0], fit[1]), n))
                _log(f"xfit {t}: {ni} instances, {mm:.0f} held-out rows, {vv:.0f} violations; a={fit[0]:.3f} b={fit[1]:.1f} "
                     f"residual n=30: {res(30):.2e} n=1000: {res(1000):.2e}")
    return {"gate": gate_rules, "ab": {t: v for t, v in ab.items() if v is not None}, "oof_caps": oof_flags,
            "test_caps": test_flags, "q": ab_arg is not None, "fo": fo, "fits": fits}


# ----------------------------------------------------------------------------- gradient-boosted trees candidate

def _gbdt_candidate(frame, given, idx, vis, held_pos, Y, fold, heads, oof_mlp, oof_lin, oof_marg, stage_deadline):
    """Per item: histogram gradient-boosted trees on the integer-coded GIVEN block with native categorical splits,
    fitted on GBDT_FOLDS of the 5 shared folds (out-of-fold rows from those folds; test = mean of the fold models).
    Items go in order of how much the MLP beat the linear model (nonlinear structure first), in batches fitted by
    concurrent threads (each fit single-threaded: the OpenMP thread count is a per-thread setting), until the stage
    deadline; a fit that would start after the deadline is skipped. Returns (oof, test, have_item, rows_mask) or None."""
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.ensemble import HistGradientBoostingClassifier
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None
    cols = []
    for g in given:
        c = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy().astype(np.float32)
        c[c < 0] = np.nan
        cols.append(c)
    X = np.stack(cols, 1) if cols else np.zeros((len(frame), 1), np.float32)
    cat = np.ones(X.shape[1], bool)
    Xv, Xh = X[vis], X[held_pos]
    n, nh = len(vis), len(held_pos)
    rows_mask = fold < GBDT_FOLDS
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((nh, heads.total), np.float32)
    have = np.zeros(heads.J, bool)

    def mean_score(M, ok, j):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        yj = Y[ok, j]
        return np.log(_scored(M[ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), yj]).mean()

    gain = np.full(heads.J, -np.inf)
    for j in range(heads.J):
        ok = (Y[:, j] >= 0) & rows_mask
        if ok.sum() < GBDT_MIN_ROWS or heads.widths[j] < 2:
            continue
        gain[j] = (mean_score(oof_mlp, ok, j) - mean_score(oof_lin, ok, j)) + 1e-3 * (mean_score(oof_mlp, ok, j) - mean_score(oof_marg, ok, j))
    order = [j for j in np.argsort(-gain) if np.isfinite(gain[j])]

    def fit_one(j, f):
        if time.time() > stage_deadline:
            return None, None
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        okrow = Y[:, j] >= 0
        tr, va = np.flatnonzero((fold != f) & okrow), np.flatnonzero(fold == f)
        ytr = Y[tr, j]
        marg = np.bincount(ytr, minlength=s1 - s0) + ALPHA
        marg = marg / marg.sum()
        q = np.vstack([Xv[va], Xh])
        if len(np.unique(ytr)) < 2:
            return np.tile(marg, (len(q), 1)), va
        counts = np.bincount(ytr)
        params = dict(GBDT_PARAMS)
        if counts[counts > 0].min() < 3:            # the stratified early-stopping split needs >= 2 per class
            params.update(early_stopping=False, max_iter=GBDT_FIXED_ITERS)
        try:
            clf = HistGradientBoostingClassifier(categorical_features=cat, random_state=f, **params)
            if threadpool_limits is not None:
                with threadpool_limits(limits=1):
                    clf.fit(Xv[tr], ytr)
            else:
                clf.fit(Xv[tr], ytr)
            pr = np.zeros((len(q), s1 - s0))
            pr[:, clf.classes_] = clf.predict_proba(q)
        except Exception as e:
            _log(f"gbdt item {j} fold {f} failed: {e!r}")
            return None, None
        return 0.97 * pr + 0.03 * marg, va

    t0, done_items = time.time(), 0
    workers = GBDT_BATCH if threadpool_limits is not None else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for b in range(0, len(order), GBDT_BATCH):
            batch = order[b:b + GBDT_BATCH]
            per_batch = (time.time() - t0) / (b // GBDT_BATCH) if b else 0.0
            if time.time() > stage_deadline or (b and time.time() + 1.5 * per_batch > stage_deadline):
                break
            futs = {(j, f): pool.submit(fit_one, j, f) for j in batch for f in range(GBDT_FOLDS)}
            for j in batch:
                s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
                acc = np.zeros((nh, s1 - s0), np.float64)
                good = True
                for f in range(GBDT_FOLDS):
                    p, va = futs[(j, f)].result()
                    if p is None or not np.isfinite(p).all():
                        good = False
                        continue
                    oof[va, s0:s1] = p[:len(va)]
                    acc += p[len(va):]
                if good:
                    test[:, s0:s1] = acc / GBDT_FOLDS
                    have[j] = True
                    done_items += 1
                else:
                    oof[rows_mask, s0:s1] = 0.0
    _log(f"gbdt: {done_items}/{len(order)} items in {time.time() - t0:.0f}s")
    if not have.any():
        return None
    return oof, test, have, rows_mask


# ----------------------------------------------------------------------------- gate hybrid

def _gate_items(rec, pred, idx):
    """Gated items whose parent is another PREDICT item: {j: (jp, obs_idx)} and a parents-first order."""
    pj_of = {n: j for j, n in enumerate(pred)}
    items = {}
    for j, n in enumerate(pred):
        g = rec[n].get("gate")
        if not g or g["parent"] not in pj_of:
            continue
        obs = np.array([idx[g["parent"]][v] for v in g["observed_if"] if v in idx[g["parent"]]], dtype=int)
        if len(obs):
            items[j] = (pj_of[g["parent"]], obs)
    order, seen = [], set()

    def visit(j):
        if j in seen:
            return
        seen.add(j)
        if j in items:
            visit(items[j][0])
            order.append(j)

    for j in items:
        visit(j)
    return items, order


def _set_gate(M, s0, s1, g):
    """Replace the gate slot with 1-g and rescale the substantive part to sum to g (substantive shape unchanged)."""
    block = M[:, s0:s1 - 1]
    ssum = block.sum(1, keepdims=True)
    sub = np.where(ssum > 1e-12, block / np.maximum(ssum, 1e-12), 1.0 / (s1 - s0 - 1))
    M[:, s0:s1 - 1] = sub * g[:, None]
    M[:, s1 - 1] = 1.0 - g


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


ROUTE_TOP, ROUTE_FOLDS, ROUTE_KEEP = 3, 5, 2    # MI candidates per item; cross-fit folds for the routing tables; composed features kept


def _route_mi(Yk, ak, widths, excl):
    """Bias-corrected mutual information (nats) between the gated indicator ak and every PREDICT column of Yk
    (-1 = unknown is its own category); columns in excl get -inf."""
    n, J = Yk.shape
    off = np.concatenate([[0], np.cumsum(np.asarray(widths) + 1)])
    codes = ((Yk + 1 + off[:-1][None, :]) * 2 + ak[:, None].astype(np.int64)).ravel()
    cnt = np.bincount(codes, minlength=2 * int(off[-1])).astype(np.float64)
    mi = np.full(J, -np.inf)
    for p in range(J):
        if p in excl:
            continue
        c = cnt[2 * off[p]:2 * off[p + 1]].reshape(-1, 2)
        c = c[c.sum(1) > 0]
        pj = c / n
        pc, pk = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
        nz = pj > 0
        mi[p] = float((pj[nz] * np.log(pj[nz] / (pc @ pk)[nz])).sum()) - (len(c) - 1) / (2.0 * n)
    return mi


def _route_table(Yp, asked, known, K, folds=None):
    """r(v) = P(asked | parent = v), half-count smoothed. With folds, also the out-of-fold tables (one row per fold)."""
    ok = known & (Yp >= 0)
    a = np.bincount(Yp[ok], weights=asked[ok], minlength=K)
    c = np.bincount(Yp[ok], minlength=K).astype(np.float64)
    full = (a + 0.5) / (c + 1.0)
    if folds is None:
        return full
    out = np.empty((ROUTE_FOLDS, K))
    for f in range(ROUTE_FOLDS):
        m = ok & (folds == f)
        af = np.bincount(Yp[m], weights=asked[m], minlength=K)
        cf = np.bincount(Yp[m], minlength=K).astype(np.float64)
        out[f] = (a - af + 0.5) / (c - cf + 1.0)
    return full, out


# ----------------------------------------------------------------------------- gate events
# Children that sit behind the same parent open set share one asked / not-asked event. Its label is read from every
# visible row on which any of the children is known, so a dedicated binary model sees the pooled label once; its
# out-of-fold probability then enters the gate stacker as one more column.

def _gate_events(Y, rec, pred, idx, heads, oof_mlp):
    """Events from the gated items with a PREDICT parent: groups of children with the same (declared parent, open set).
    Returns (events, dropped); an event is {children, jp, label (n,) in {-1,0,1}, leverage}, ordered by leverage = the
    summed OOF loss (nats/cell) of the children and all their descendants. Groups whose children disagree on asked /
    not asked (< GATEPFN_AGREE where two or more are known), or with too few labeled rows, are dropped."""
    items, _ = _gate_items(rec, pred, idx)
    if not items:
        return [], []
    n = len(Y)
    kids = {}
    for j, (jp, _) in items.items():
        kids.setdefault(jp, []).append(j)

    def desc(j):
        out, stack = set(), [j]
        while stack:
            for k in kids.get(stack.pop(), []):
                if k not in out:
                    out.add(k)
                    stack.append(k)
        return out

    loss = np.zeros(heads.J)
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Y[:, j] >= 0
        if ok.sum() >= 20:
            loss[j] = -np.log(_scored(oof_mlp[ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), Y[ok, j]]).mean()
    groups = {}
    for j, (jp, obs) in items.items():
        groups.setdefault((jp, tuple(sorted(int(x) for x in obs))), []).append(j)
    events, dropped = [], []
    for (jp, obs), ch in groups.items():
        ch = sorted(ch)
        A = np.full((n, len(ch)), -1, np.int8)
        for c, j in enumerate(ch):
            k = Y[:, j] >= 0
            A[k, c] = (Y[k, j] != heads.widths[j] - 1)
        known = A >= 0
        nk = known.sum(1)
        label = np.where(nk > 0, (A == 1).any(1).astype(np.int8), np.int8(-1))
        multi = nk >= 2
        agree = float(((A.max(1) == np.where(known, A, 2).min(1))[multi]).mean()) if multi.any() else 1.0
        lab = label[label >= 0]
        if len(lab) < GATEPFN_MIN_ROWS or min(int(lab.sum()), len(lab) - int(lab.sum())) < GATEPFN_MIN_MINOR:
            dropped.append((ch, "rows", len(lab)))
            continue
        if agree < GATEPFN_AGREE:
            dropped.append((ch, "inconsistent", agree))
            continue
        members = set(ch)
        for j in ch:
            members |= desc(j)
        events.append(dict(children=ch, jp=jp, label=label.astype(np.int64), agree=agree,
                           leverage=float(sum(loss[k] for k in members))))
    events.sort(key=lambda e: -e["leverage"])
    return events[:GATEPFN_MAX_EVENTS], dropped


def _gatepfn_stage(frame, given, opts, vis, held_pos, Y, fold, heads, events, stage_deadline):
    """Binary TabPFN per event, breadth first over the shared folds (the same machinery as the item candidate; the event
    labels play the part of the target columns). Returns {child: (logit_oof (n,), logit_test (nh,), rows (n,) bool)}
    for the children of every event that completed >= GATEPFN_MIN_FOLDS folds, or None."""
    if not _tabpfn_ready():
        return None
    Cls, ckpt = _tabpfn_state["cls"], _tabpfn_state["ckpt"]
    make = lambda f: Cls(model_path=ckpt, device=str(DEVICE), n_estimators=TABPFN_EST, random_state=f,
                         n_preprocessing_jobs=1, enable_torch_compile=False, show_progress_bar=False)
    Xdf = _ord_frame(frame, given, opts)
    E = np.stack([e["label"] for e in events], 1)
    hev = _Heads([2] * len(events))
    saved = dict(_pfn_cov)
    try:
        r = _pfn_foldmajor(Cls, ckpt, make, Xdf, vis, held_pos, E, fold, hev, list(range(len(events))), stage_deadline, N_FOLDS, 1.0)
    finally:
        _pfn_cov.update(saved)
    if r is None:
        return None
    oof, test, have, rows = r
    if int(fold[rows].max() + 1 if rows.any() else 0) < GATEPFN_MIN_FOLDS:
        return None
    out = {}
    for e, ev in enumerate(events):
        if not have[e]:
            continue
        po, pt = oof[:, 2 * e + 1].astype(np.float64), test[:, 2 * e + 1].astype(np.float64)
        if not (np.isfinite(po[rows]).all() and np.isfinite(pt).all()):
            continue
        for j in ev["children"]:
            out[j] = (_logit(po), _logit(pt), rows)
    _gatepfn_stage.last = dict(done=int(have.sum()), total=len(events), folds=int(fold[rows].max() + 1))
    return out or None


def _gate_fit(Z, a, th0):
    """Penalised logistic fit of the gate stacker (prior toward th0). Returns (theta, gain over th0, loss) in nats."""
    nk = len(a)

    def f(th):
        z = Z @ th
        loss = np.mean(np.logaddexp(0, z) - a * z) + GATE_PRIOR / nk * ((th - th0) ** 2).sum()
        g = 1 / (1 + np.exp(-z))
        return loss, Z.T @ (g - a) / nk + 2 * GATE_PRIOR / nk * (th - th0)

    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    base = np.mean(np.logaddexp(0, Z @ th0) - a * (Z @ th0))
    fit = np.mean(np.logaddexp(0, Z @ res.x) - a * (Z @ res.x))
    return res.x, nk * (base - fit), nk * fit


def _gate_hybrid(Bo, P, Y, rec, pred, idx, heads, ev=None):
    """Per gated item with a PREDICT parent: g = sigmoid(w . logit(features) + b), shrunk toward (1,0,...) = the direct
    gate. Features: the direct gate, the gate composed from the declared parent's open set, and (routing) up to
    ROUTE_KEEP gates composed through the per-value asked-rate of the best-scoring candidate parents (the declared
    parent and the PREDICT items most informative about the gated indicator). Bo / P are the blended OOF / test
    vectors (gate slot last); updated in place, parents first. ev: optional {child: (event logit OOF, event logit test,
    rows with an OOF value)}; for such a child the stacker is refitted with that column and used when it beats the
    same stacker without the column on the same rows by GATEPFN_MIN_DELTA nats."""
    items, order = _gate_items(rec, pred, idx)
    stats = []
    _gate_hybrid.routes = {}
    _gate_hybrid.ev_stats = {}
    J = heads.J
    folds = np.random.default_rng(0).integers(0, ROUTE_FOLDS, len(Y))
    kids = {}
    for j, (jp, _) in items.items():
        kids.setdefault(jp, []).append(j)

    def descendants(j):
        out, stack = set(), [j]
        while stack:
            for k in kids.get(stack.pop(), []):
                if k not in out:
                    out.add(k)
                    stack.append(k)
        return out

    for j in order:
        jp, obs = items[j]
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        sp_ = int(heads.offsets[jp])
        known = Y[:, j] >= 0
        if known.sum() < 200:
            continue
        asked = ((Y[:, j] != (s1 - s0 - 1)) & known).astype(np.float64)
        opens = np.isin(Y[:, jp], obs) & (Y[:, jp] >= 0)
        n1, n0 = int((opens & known).sum()), int((~opens & known).sum())
        r1 = (float((asked * (opens & known)).sum()) + 0.01) / (n1 + 0.02) if n1 else 1 - 1e-4
        r0 = (float((asked * (~opens & known)).sum()) + 0.01) / (n0 + 0.02) if n0 else 1e-4
        dir_o, dir_t = 1.0 - Bo[:, s1 - 1].astype(np.float64), 1.0 - P[:, s1 - 1].astype(np.float64)
        pa_o = np.clip(Bo[:, sp_ + obs].sum(1).astype(np.float64), 0, 1)
        pa_t = np.clip(P[:, sp_ + obs].sum(1).astype(np.float64), 0, 1)
        comp_o, comp_t = r0 + (r1 - r0) * pa_o, r0 + (r1 - r0) * pa_t
        cols_o = [_logit(dir_o), _logit(comp_o), np.ones(len(dir_o))]
        cols_t = [_logit(dir_t), _logit(comp_t), np.ones(len(dir_t))]
        a = asked[known]
        route = []
        if not NO_ROUTE:
            kn = np.flatnonzero(known)
            mi = _route_mi(Y[kn], asked[kn], heads.widths, {j} | descendants(j))
            cand = [p for p in np.argsort(-mi)[:ROUTE_TOP] if np.isfinite(mi[p])]
            if jp not in cand:
                cand.append(jp)
            scored = []
            for p in cand:
                K = int(heads.widths[p])
                q0 = int(heads.offsets[p])
                full, oof_r = _route_table(Y[:, p], asked, known, K, folds)
                Bp = Bo[:, q0:q0 + K].astype(np.float64)
                Bp = Bp / np.maximum(Bp.sum(1, keepdims=True), 1e-12)
                Pp = P[:, q0:q0 + K].astype(np.float64)
                Pp = Pp / np.maximum(Pp.sum(1, keepdims=True), 1e-12)
                co = np.clip((Bp * oof_r[folds]).sum(1), 1e-4, 1 - 1e-4)       # out-of-fold table x OOF parent vector
                ct = np.clip(Pp @ full, 1e-4, 1 - 1e-4)
                ll = float((a * np.log(co[known]) + (1 - a) * np.log(1 - co[known])).sum())
                scored.append((ll, p, co, ct))
            scored.sort(key=lambda x: -x[0])
            for ll, p, co, ct in scored[:ROUTE_KEEP]:
                cols_o.insert(-1, _logit(co))
                cols_t.insert(-1, _logit(ct))
                route.append((int(p), ll))
        Zo_all = np.stack(cols_o, 1)
        Zo = Zo_all[known]
        Zt = np.stack(cols_t, 1)
        nk = len(a)
        th0 = np.zeros(Zo.shape[1])
        th0[0] = 1.0

        def f(th):
            z = Zo @ th
            loss = np.mean(np.logaddexp(0, z) - a * z) + GATE_PRIOR / nk * ((th - th0) ** 2).sum()
            g = 1 / (1 + np.exp(-z))
            return loss, Zo.T @ (g - a) / nk + 2 * GATE_PRIOR / nk * (th - th0)

        res = minimize(f, th0, jac=True, method="L-BFGS-B")
        gain = nk * (np.mean(np.logaddexp(0, Zo @ th0) - a * (Zo @ th0)) - np.mean(np.logaddexp(0, Zo @ res.x) - a * (Zo @ res.x)))
        stats.append((j, gain, res.x))
        _gate_hybrid.routes[j] = route
        go = gt = None
        if ev is not None and j in ev:
            try:
                zo_e, zt_e, erow = ev[j]
                me = known & erow
                if int(me.sum()) >= 200:
                    Zo_e = np.insert(Zo_all, Zo_all.shape[1] - 1, zo_e, axis=1)
                    Zt_e = np.insert(Zt, Zt.shape[1] - 1, zt_e, axis=1)
                    th0e = np.insert(th0, len(th0) - 1, 0.0)
                    _, g_b, l_b = _gate_fit(Zo_all[me], asked[me], th0)
                    th_e, g_e, l_e = _gate_fit(Zo_e[me], asked[me], th0e)
                    acc_e = bool(g_e >= GATE_MIN_GAIN and l_b - l_e >= GATEPFN_MIN_DELTA and np.isfinite(th_e).all())
                    _gate_hybrid.ev_stats[j] = dict(rows=int(me.sum()), gain_base=float(g_b), gain_event=float(g_e),
                                                    delta=float(l_b - l_e), w_event=float(th_e[-2]), accepted=acc_e)
                    if acc_e:
                        base_o = (1 / (1 + np.exp(-(Zo_all @ res.x)))) if gain >= GATE_MIN_GAIN else dir_o
                        go = np.clip(np.where(erow, 1 / (1 + np.exp(-(Zo_e @ th_e))), base_o), 1e-4, 1 - 1e-4)
                        gt = np.clip(1 / (1 + np.exp(-(Zt_e @ th_e))), 1e-4, 1 - 1e-4)
                        stats[-1] = (j, float(g_e), th_e)
            except Exception as e_:
                _log("gate event column failed:", repr(e_)[:200])
                go = gt = None
        if go is None:
            if gain < GATE_MIN_GAIN:
                continue
            go = np.clip(1 / (1 + np.exp(-(Zo_all @ res.x))), 1e-4, 1 - 1e-4)
            gt = np.clip(1 / (1 + np.exp(-(Zt @ res.x))), 1e-4, 1 - 1e-4)
        _set_gate(Bo, s0, s1, go)
        _set_gate(P, s0, s1, gt)
    return stats


def _sibpool(Bo, P, Y, rec, pred, idx, heads, par):
    """Children of one parent open set share one asked/not-asked event. Groups (declared PREDICT parent, open set) are found from the
    visible rows and kept when the children agree on asked/not-asked in >= SIBPOOL_AGREE of rows where >= 2 are known. For a kept group
    each child's gate logit (centred on its own OOF mean) is shrunk toward the group's mean centred logit, per row, with strength lam.
    lam is fitted on the OOF vectors; a group is changed only if lam fitted on one parity half gains > SIBPOOL_MIN_NATS on the other,
    for both halves. Bo / P updated in place (gate slot and rescaled substantive part). Uses PREDICT-column probabilities only.
    Returns (groups accepted, groups tested, nested gains)."""
    if SIBPOOL_FAIL:
        raise RuntimeError("SB_SIBPOOL_FAIL")
    items, _ = _gate_items(rec, pred, idx)
    groups = {}
    for j, (jp, obs) in items.items():
        groups.setdefault((jp, tuple(int(o) for o in obs)), []).append(j)
    off = heads.offsets
    tested, acc, gains_all = 0, 0, []
    for key, kids in groups.items():
        if len(kids) < 2:
            continue
        A = np.full((len(Y), len(kids)), -1, np.int8)
        for c, j in enumerate(kids):
            kn = Y[:, j] >= 0
            A[kn, c] = (Y[kn, j] != (int(off[j + 1] - off[j]) - 1)).astype(np.int8)
        nk = (A >= 0).sum(1)
        rows2 = nk >= 2
        if rows2.sum() < 200:
            continue
        Am = np.where(A >= 0, A, 0)
        s_ = Am.sum(1)
        agree = ((s_ == 0) | (s_ == nk))[rows2].mean()     # known children all asked or all not asked
        if agree < SIBPOOL_AGREE:
            continue
        tested += 1
        L_o, L_t, C, ok_o, ok_t = [], [], [], [], []
        for j in kids:
            s1 = int(off[j + 1])
            go, gt = 1.0 - Bo[:, s1 - 1].astype(np.float64), 1.0 - P[:, s1 - 1].astype(np.float64)
            ok_o.append((go > 2e-4) & (go < 1 - 2e-4)); ok_t.append((gt > 2e-4) & (gt < 1 - 2e-4))
            lo, lt = _logit(go), _logit(gt)
            c = float(lo[ok_o[-1]].mean()) if ok_o[-1].any() else 0.0
            L_o.append(lo - c); L_t.append(lt - c); C.append(c)
        Mo, Mt = np.mean(L_o, 0), np.mean(L_t, 0)

        def new_g(lo, c, M, lam):
            return 1 / (1 + np.exp(-(c + (1 - lam) * lo + lam * M)))

        # score table: total log score (nats) of the kids' visible cells per lam per parity half
        S = np.zeros((len(SIBPOOL_LAMS), 2))
        prep = []
        for ci, j in enumerate(kids):
            s0, s1 = int(off[j]), int(off[j + 1])
            K = s1 - s0
            y = Y[:, j]
            r = np.flatnonzero((y >= 0) & ok_o[ci])
            blk = np.clip(Bo[r, s0:s1 - 1].astype(np.float64), 0, None)
            sub = blk / np.maximum(blk.sum(1, keepdims=True), 1e-12)
            sub_y = np.where(y[r] < K - 1, sub[np.arange(len(r)), np.minimum(y[r], K - 2)], 0.0)
            prep.append((r, K, y[r] == K - 1, sub_y))
        for li, lam in enumerate(SIBPOOL_LAMS):
            for ci, j in enumerate(kids):
                r, K, isg, sub_y = prep[ci]
                g = new_g(L_o[ci][r], C[ci], Mo[r], lam)
                p_y = np.where(isg, 1 - g, sub_y * g)
                lp = np.log(EPS + (1 - K * EPS) * p_y)
                for h in (0, 1):
                    S[li, h] += lp[par[r] == h].sum()
        gains = []
        for h in (0, 1):
            best = int(np.argmax(S[:, 1 - h]))
            gains.append(float(S[best, h] - S[0, h]))
        gains_all.append(round(min(gains), 1))
        if min(gains) <= SIBPOOL_MIN_NATS:
            continue
        lam = SIBPOOL_LAMS[int(np.argmax(S.sum(1)))]
        if lam <= 0:
            continue
        acc += 1
        for ci, j in enumerate(kids):
            s0, s1 = int(off[j]), int(off[j + 1])
            go = np.where(ok_o[ci], new_g(L_o[ci], C[ci], Mo, lam), 1.0 - Bo[:, s1 - 1])
            gt = np.where(ok_t[ci], new_g(L_t[ci], C[ci], Mt, lam), 1.0 - P[:, s1 - 1])
            _set_gate(Bo, s0, s1, np.clip(go, 1e-4, 1 - 1e-4))
            _set_gate(P, s0, s1, np.clip(gt, 1e-4, 1 - 1e-4))
    return acc, tested, gains_all


# ----------------------------------------------------------------------------- main entry

def _emit(frame, items, pred, held_pos, P, heads):
    col = {n: frame.columns.get_loc(n) for n in items}
    pj = {n: j for j, n in enumerate(pred)}
    row_of = {r: i for i, r in enumerate(held_pos)}
    vals = frame.to_numpy(dtype=object)
    out = []
    for r in range(len(frame)):
        for n in items:
            v = vals[r, col[n]]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if n in pj and r in row_of:
                    j = pj[n]
                    vec = P[row_of[r], heads.offsets[j]:heads.offsets[j + 1]].astype(np.float64)
                    vec = np.clip(vec, 0, None)
                    if not np.isfinite(vec).all() or vec.sum() <= 0:
                        vec = np.ones_like(vec)
                    out.append((vec / vec.sum()).tolist())
                else:                       # blank GIVEN cell: not expected, return uniform
                    out.append([1.0] * len(_emit.opts[n]))
    return out


def _late(margin=0.0, stage=""):
    """True once the shared wall clock is past the hard end of the budget: BUDGET + 60 s in the development phase (limit 900),
    BUDGET + 300 s in the final phase (limit 3600); `margin` brings the stop earlier for stages that are slow."""
    if not V70HARD:
        return False
    late = os.environ.get("SB_V70_LATE") == "1" or time.time() - T0 > BUDGET + (300.0 if _calls["phase"] == 2 else 60.0) - margin
    if late and stage:
        _log(f"late guard: {stage} skipped (clock {time.time() - T0:.0f}s, budget {BUDGET:.0f}s)")
    return late


def _workload(J):
    return max(J, 40) ** 0.8


def _share(J):
    """Fraction of the remaining budget for this instrument: its workload over that of itself plus the instruments
    not yet seen (matched by item count). Equal split when the item count is not one of the expected ones."""
    seen = _calls.setdefault("seen", [])
    left = list(INSTR_J)
    for s in seen:
        if s in left:
            left.remove(s)
    if J not in left:
        return 1.0 / max(N_INSTRUMENTS - len(seen), 1)
    left.remove(J)
    seen.append(J)
    return _workload(J) / (_workload(J) + sum(_workload(x) for x in left))


def _detect_phase(frame, schema):
    """Final phase (2) when the visible respondents are TRAIN + DEV and the held-out ones are TEST, as the schema's split
    block counts them; the development phase (1) shows TRAIN only and holds out DEV. Anything else counts as phase 1."""
    forced = os.environ.get("SB_PHASE")
    if forced in ("1", "2"):
        return int(forced)
    sp = schema.get("split") or {}
    try:
        cols = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
        held = int(frame[cols].isna().any(axis=1).sum())
        vis = len(frame) - held
        if V70HARD and len(frame) == sp["n_train"] + sp["n_dev"] + sp["n_test"] and held == sp["n_test"]:
            return 2                       # TRAIN + DEV + TEST all present: unambiguous
        if vis == sp["n_train"] and held == sp["n_dev"]:
            return 1
        # phase 2 when either count matches the final split (a one-sided match still signals it); only the exact phase-1 signature stays phase 1
        return 2 if (vis == sp["n_train"] + sp["n_dev"] or (held == sp["n_test"] and held != sp["n_dev"])) else 1
    except Exception:
        return 1


def predict(frame, schema):
    global BUDGET
    if _calls["n"] == 0 and not FIXED:
        _calls["phase"] = _detect_phase(frame, schema)
        if _calls["phase"] == 2 and not BUDGET_FIXED:
            BUDGET = FINAL_BUDGET
        _log(f"phase {_calls['phase']}, budget {BUDGET:.0f}s")
    _calls["n"] += 1
    remaining_calls = max(N_INSTRUMENTS - _calls["n"] + 1, 1)
    if LEGACY:
        allowance = max((BUDGET - (time.time() - T0)) / remaining_calls, 20.0)
    else:
        J = sum(1 for r in schema["items"].values() if r["class"] == "PREDICT")
        allowance = max((BUDGET - (time.time() - T0)) * _share(J), 20.0)
    deadline = time.time() + allowance
    _log(f"instrument {_calls['n']}: allowance {allowance:.0f}s, device {DEVICE}")

    frame = frame.reset_index(drop=True)
    rec, items, opts, idx, given, pred = _layout(frame, schema)
    _emit.opts = opts
    heads = _Heads([len(opts[n]) for n in pred])

    blank = frame[pred].isna().to_numpy()
    held = blank.any(axis=1)
    held_pos = np.flatnonzero(held)
    vis = np.flatnonzero(~held)
    Y = np.stack([frame[n].map(idx[n]).fillna(-1).astype(int).to_numpy() for n in pred], 1)[vis]
    Yoh = _onehot_flat(Y, heads)
    marg = _marginal_flat(Yoh, heads)
    _snap["P"] = None
    P = np.tile(marg, (len(held_pos), 1))
    try:
        _gate_override.trust = _gate_trust(frame, vis, Y, pred, rec, heads, set(given))
        _log("gates trusted (asked, skipped):", {k: v for k, v in _gate_override.trust.items() if not all(v)} or "all exact")
        # safe fallback first: marginal + skip logic
        P = _gate_override(P, frame.iloc[held_pos], pred, rec, heads, set(given))
    except Exception as e:
        _log("gate fallback failed, plain marginal:", repr(e))
        P = np.tile(marg, (len(held_pos), 1))
    P_safe = P
    try:
        P = _model(frame, rec, given, pred, opts, idx, heads, vis, held_pos, Y, Yoh, deadline)
    except _StrictError:
        raise
    except Exception as e:                 # never fail the run on the modelling path
        _log("model path failed, using fallback:", repr(e))
        P = _snap["P"] if _snap["P"] is not None else P_safe
    try:
        if os.environ.get("SB_V70_EMITFAIL") == "1":
            raise RuntimeError("v70 emit failure hook")   # test hook
        return _emit(frame, items, pred, held_pos, P, heads)
    except Exception as e:
        _log("emit failed, retrying with the blend snapshot:", repr(e))
        if _snap["P"] is not None:
            try:
                return _emit(frame, items, pred, held_pos, _snap["P"], heads)
            except Exception as e2:
                _log("snapshot emit failed, plain marginal:", repr(e2))
        return _emit(frame, items, pred, held_pos, np.tile(marg, (len(held_pos), 1)), heads)


def _more_big_seeds(Xv, Y, Xh, fold, heads, seed_cost, limit, oof, test):
    """Add whole seeds (all folds) of the big trunk while the measured per-seed cost fits before `limit`; a seed that
    does not finish is discarded. The seed average of oof/test["mlp_big"] is updated in place."""
    h, d_, wd_, lr_, ep_, dep_ = BIG_CFG
    n_seeds, t0 = BIG_SEEDS, time.time()
    while n_seeds < ANYTIME_BIG_SEEDS and time.time() + 1.3 * seed_cost < limit:
        t_s = time.time()
        o2 = np.zeros_like(oof["mlp_big"])
        t2 = np.zeros_like(test["mlp_big"])
        ok = True
        for f in range(N_FOLDS):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            net = _train(Xv[tr], Y[tr], Xv[va], Y[va], heads, h, d_, wd_, lr_, ep_, limit, seed=100 * (n_seeds + 1) + f, depth=dep_)
            o2[va] = _predict_flat(net, Xv[va], heads)
            t2 += _predict_flat(net, Xh, heads) / N_FOLDS
            del net
            if time.time() > limit:
                ok = False
                break
        if not (ok and np.isfinite(o2).all() and np.isfinite(t2).all()):
            _log(f"mlp_big seed {n_seeds} discarded")
            break
        oof["mlp_big"] = (oof["mlp_big"] * n_seeds + o2) / (n_seeds + 1)
        test["mlp_big"] = (test["mlp_big"] * n_seeds + t2) / (n_seeds + 1)
        n_seeds += 1
        seed_cost = max(seed_cost, time.time() - t_s)
    _log(f"mlp_big seeds {n_seeds} (+{n_seeds - BIG_SEEDS}, {time.time() - t0:.0f}s)")


NO_FANOUT = os.environ.get("SB_NO_FANOUT") == "1"          # fan-out-weighted priority for the high fan-out gate parents (UNHCR-sized); off reproduces v66
FANOUT_FAIL = os.environ.get("SB_FANOUT_FAIL") == "1"      # test only: raise inside the fan-out candidate
FANOUT_MIN_ITEMS = 150
FANOUT_SHARE_MIN = 0.04            # a parent is high fan-out when it has at least this share of all gated children (parent a PREDICT item); 0.04 selects the 5 parents carrying ~77% of the children on the UNHCR schema
FANOUT_W_CAP = 10.0                # cap of the blend-objective weight
FANOUT_SEEDS = int(os.environ.get("SB_FANOUT_SEEDS", 6))
FANOUT_FRAC = float(os.environ.get("SB_FANOUT_FRAC", 0.10))   # share of the time left (before the foundation-model stages) the parent trunk may use
_fanout_w = {}                     # item index -> 1 + children / median children, for the selected parents only
NO_SEEDS67 = os.environ.get("SB_NO_SEEDS67") == "1"        # extra seeds of linear/mlp on non-UNHCR-sized schemas; off reproduces v66
SEEDS67_FAIL = os.environ.get("SB_SEEDS67_FAIL") == "1"    # test only
SEEDS67 = int(os.environ.get("SB_SEEDS67", 4))             # seeds averaged in the end (v66: 1)
SEEDS67_FRAC = float(os.environ.get("SB_SEEDS67_FRAC", 0.25))   # share of the time left the extra seeds may use
NO_RIDGE67 = os.environ.get("SB_NO_RIDGE67") == "1"        # per-item-CV ridge blend candidate on non-UNHCR-sized schemas; off reproduces v66
RIDGE67_FAIL = os.environ.get("SB_RIDGE67_FAIL") == "1"    # test only
RIDGE67_GRID = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)       # penalty 0.5 * c * ||W||^2 added to the summed (not mean) NLL
RIDGE67_FRAC = float(os.environ.get("SB_RIDGE67_FRAC", 0.12))
SMALL67_MAX_ITEMS = 150


def _fanout_parents(rec, pred, idx, heads):
    """{item index: weight} for the gate parents (PREDICT items) holding >= FANOUT_SHARE_MIN of all gated children, weight 1 + children / median
    children over all parents. UNHCR-sized instruments only."""
    if NO_FANOUT or heads.J < FANOUT_MIN_ITEMS:
        return {}
    items, _ = _gate_items(rec, pred, idx)
    cnt = {}
    for j, (jp, _o) in items.items():
        cnt[jp] = cnt.get(jp, 0) + 1
    if not cnt:
        return {}
    med = float(np.median(list(cnt.values())))
    tot = float(len(items))
    return {jp: min(1.0 + c / max(med, 1.0), FANOUT_W_CAP) for jp, c in cnt.items() if c >= FANOUT_SHARE_MIN * tot}


def _fanout_candidate(Xv, Y, Xh, fold, heads, pj, base_oof, base_test, stage_deadline):
    """Own seed-averaged big-trunk MLP trained on the parent heads only. Whole seeds (all five folds) are averaged while the measured
    cost fits before the stage deadline. Returns (oof, test, have, rows) for the per-item blend, active on the parent items only."""
    if FANOUT_FAIL:
        raise RuntimeError("SB_FANOUT_FAIL")
    pj = sorted(pj)
    sub = _Heads([int(heads.widths[j]) for j in pj])
    Ys = Y[:, pj]
    h, d_, wd_, lr_, ep_, dep_ = BIG_CFG
    n = len(Xv)
    acc_o = np.zeros((n, sub.total), np.float32)
    acc_t = np.zeros((len(Xh), sub.total), np.float32)
    n_done, t0 = 0, time.time()
    for sd in range(FANOUT_SEEDS):
        o2 = np.zeros_like(acc_o)
        t2 = np.zeros_like(acc_t)
        ok = True
        for f in range(N_FOLDS):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            net = _train(Xv[tr], Ys[tr], Xv[va], Ys[va], sub, h, d_, wd_, lr_, ep_, stage_deadline, seed=5000 + 100 * sd + f, depth=dep_)
            o2[va] = _predict_flat(net, Xv[va], sub)
            t2 += _predict_flat(net, Xh, sub) / N_FOLDS
            del net
            per = (time.time() - t0) / (n_done * N_FOLDS + f + 1)
            if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
                ok = False
                break
        if not ok or not (np.isfinite(o2).all() and np.isfinite(t2).all()):
            break
        acc_o += o2
        acc_t += t2
        n_done += 1
        if time.time() + 1.3 * (time.time() - t0) / n_done > stage_deadline:
            break
    if n_done == 0:
        return None
    oof = base_oof.copy()
    test = base_test.copy()
    for k, j in enumerate(pj):
        s_full = slice(heads.offsets[j], heads.offsets[j + 1])
        s_sub = slice(sub.offsets[k], sub.offsets[k + 1])
        oof[:, s_full] = acc_o[:, s_sub] / n_done
        test[:, s_full] = acc_t[:, s_sub] / n_done
    have = np.zeros(heads.J, bool)
    have[pj] = True
    _log(f"fanout parents {[int(j) for j in pj]}: {n_done} seeds, {time.time() - t0:.0f}s")
    return oof, test, have, np.ones(n, bool)


def _seeds67_stage(Xs, Xhs, Y, fold, heads, configs, oof, test, deadline, t_loop):
    """Extra whole seeds of linear and mlp (all folds), averaged into oof/test in place. A seed costs about the base loop; seeds are added
    while they fit in SEEDS67_FRAC of the time left. Returns the number of seeds averaged (incl. the base one)."""
    if SEEDS67_FAIL:
        raise RuntimeError("SB_SEEDS67_FAIL")
    t0 = time.time()
    limit = t0 + SEEDS67_FRAC * (deadline - t0)
    n_seeds = 1
    cost = max(t_loop, 1e-3)
    while n_seeds < SEEDS67 and time.time() + 1.3 * cost < limit:
        t_s = time.time()
        o2 = {c: np.zeros_like(oof[c]) for c in ("linear", "mlp")}
        t2 = {c: np.zeros_like(test[c]) for c in ("linear", "mlp")}
        for f in range(N_FOLDS):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            for c in ("linear", "mlp"):
                h, d, wd, lr, ep = configs[c]
                net = _train(Xs[tr], Y[tr], Xs[va], Y[va], heads, h, d, wd, lr, ep, deadline, seed=f + 1000 * n_seeds)
                o2[c][va] = _predict_flat(net, Xs[va], heads)
                t2[c] += _predict_flat(net, Xhs, heads) / N_FOLDS
                del net
            if f < N_FOLDS - 1 and time.time() + 1.3 * (time.time() - t_s) / (f + 1) > limit + 0.5 * cost:
                break
        else:
            if all(np.isfinite(o2[c]).all() and np.isfinite(t2[c]).all() for c in o2):
                for c in o2:
                    oof[c] = (oof[c] * n_seeds + o2[c]) / (n_seeds + 1)
                    test[c] = (test[c] * n_seeds + t2[c]) / (n_seeds + 1)
                n_seeds += 1
                cost = time.time() - t_s
                continue
        break
    _log(f"seeds67: linear/mlp averaged over {n_seeds} seeds ({time.time() - t0:.0f}s)")
    return n_seeds


def _ridge67_candidate(Xv, Y, Xh, fold, heads, stage_deadline):
    """Per-item ridge multinomial logistic: for each penalty c in RIDGE67_GRID a multi-head linear model is fitted per fold by L-BFGS on
    sum NLL + 0.5 c ||W||^2 (heads are separable, so this equals independent per-item fits); each item then takes the c with the best
    5-fold OOF log score. Returns (oof, test) or None when it cannot finish in time."""
    if RIDGE67_FAIL:
        raise RuntimeError("SB_RIDGE67_FAIL")
    t0 = time.time()
    n, d = Xv.shape
    grid = sorted(RIDGE67_GRID, reverse=True)           # strongest first: warm starts
    oofs = {c: np.zeros((n, heads.total), np.float32) for c in grid}
    tests = {c: np.zeros((len(Xh), heads.total), np.float32) for c in grid}
    Xh_t = torch.tensor(Xh, device=DEVICE)
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Xt, Yt = torch.tensor(Xv[tr], device=DEVICE), torch.tensor(Y[tr], device=DEVICE)
        Xva_t = torch.tensor(Xv[va], device=DEVICE)
        net = nn.Linear(d, heads.total).to(DEVICE)
        nn.init.zeros_(net.weight)
        nn.init.zeros_(net.bias)
        valid = Yt >= 0
        for c in grid:
            opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=60, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

            def closure():
                opt.zero_grad()
                lp = heads.logp(net(Xt))
                nll = -lp.gather(2, Yt.clamp(min=0).unsqueeze(2)).squeeze(2)
                loss = (nll * valid).sum() / len(Xt) + 0.5 * c / len(Xt) * (net.weight ** 2).sum()
                loss.backward()
                return loss

            t_fit = time.time()
            opt.step(closure)
            if f == 0 and c == grid[0]:        # cost of one fit -> whole stage; give up at once if it cannot finish
                if time.time() + 1.3 * (time.time() - t_fit) * len(grid) * N_FOLDS > stage_deadline:
                    _log("ridge67 skipped (time)")
                    return None
            net.eval()
            with torch.no_grad():
                oofs[c][va] = heads.flat(heads.logp(net(Xva_t))).cpu().numpy()
                tests[c] += heads.flat(heads.logp(net(Xh_t))).cpu().numpy() / N_FOLDS
            if time.time() > stage_deadline:
                return None
        per = (time.time() - t0) / (f + 1)
        if f < N_FOLDS - 1 and time.time() + 1.3 * per > stage_deadline:
            return None
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((len(Xh), heads.total), np.float32)
    picks = []
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Y[:, j] >= 0
        best, bc = -np.inf, grid[0]
        for c in grid:
            if ok.sum() == 0:
                break
            ll = np.log(_scored(oofs[c][ok, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(ok.sum()), Y[ok, j]]).mean()
            if ll > best:
                best, bc = ll, c
        oof[:, s], test[:, s] = oofs[bc][:, s], tests[bc][:, s]
        picks.append(bc)
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        return None
    _log(f"ridge67: per-item penalty picks {dict(zip(*np.unique(picks, return_counts=True)))} ({time.time() - t0:.0f}s)")
    return oof, test



def _model(frame, rec, given, pred, opts, idx, heads, vis, held_pos, Y, Yoh, deadline):
    share = FM_RESERVE_FRAC * (deadline - time.time())
    fm_reserve = max(FM_RESERVE, min(share, FM_RESERVE_CAP) if _calls["phase"] == 2 else share)
    X = _features(frame, given, opts, idx, {g: rec[g]["values"] for g in given})
    if not NO_AGG:
        A = _agg_features(frame, given, rec)
        if A is not None:
            X = np.hstack([X, A])
            _log(f"household aggregates: +{A.shape[1]} features")
    Xv, Xh = X[vis], X[held_pos]
    prof = _profiles(frame, given)
    pv, ph = prof[vis], prof[held_pos]
    n = len(vis)
    rng = np.random.default_rng(0)
    fold = rng.permutation(np.arange(n) % N_FOLDS)

    configs = {   # name: (hidden, dropout, weight decay, lr, max epochs)
        "linear": (0, 0.0, 1e-2, 3e-3, 60),
        "mlp": (256, 0.25, 1e-4, 2e-3, 60),
    }
    oof = {c: np.zeros((n, heads.total), np.float32) for c in configs}
    test = {c: np.zeros((len(held_pos), heads.total), np.float32) for c in configs}
    oof["marginal"] = np.zeros((n, heads.total), np.float32)
    done = 0
    _distill_ft = {}
    t_start = time.time()
    for f in range(N_FOLDS):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        oof["marginal"][va] = _marginal_flat(Yoh[tr], heads)
        for c, (h, d, wd, lr, ep) in configs.items():
            net = _train(Xv[tr], Y[tr], Xv[va], Y[va], heads, h, d, wd, lr, ep, deadline, seed=f)
            oof[c][va] = _predict_flat(net, Xv[va], heads)
            pt = _predict_flat(net, Xh, heads)
            test[c] += pt
            if DISTILL_TEACHER == "blend" and not NO_DISTILL:
                _distill_ft.setdefault(f, {})[c] = pt
        done += 1
        _log(f"fold {f} done ({time.time() - t_start:.0f}s)")
        # stop adding folds if the next one would not fit; unfinished folds keep marginal OOF
        per_fold = (time.time() - t_start) / done
        if time.time() + 1.5 * per_fold > deadline:
            break
    covered = np.isin(fold, np.arange(done))
    for c in configs:
        test[c] /= done
    test["marginal"] = np.tile(_marginal_flat(Yoh, heads), (len(held_pos), 1))

    # explicit interaction columns for the linear head and the small MLP (UNICEF / World Bank only; nested self-gate)
    Xs67, Xhs67 = Xv, Xh                     # features the linear/mlp candidates ended up trained on
    if done == N_FOLDS and not NO_INTER2:
        try:
            Wi = _inter2_cols(frame, given, opts, idx, vis)
            if Wi is not None:
                if _inter2_stage((Wi[vis], Wi[held_pos]), Xv, Xh, Y, fold, heads, configs, oof, test, deadline, time.time() - t_start,
                                 INTER_UNHCR_GAIN if "wave" in given else INTER2_GAIN):
                    Xs67, Xhs67 = np.hstack([Xv, Wi[vis]]), np.hstack([Xh, Wi[held_pos]])
        except _StrictError:
            raise
        except Exception as e:
            _log("inter2 failed:", repr(e))

    # v67: extra seeds of linear/mlp on the smaller instruments (the profile estimator below is built on the averaged mlp)
    small67 = heads.J < SMALL67_MAX_ITEMS and done == N_FOLDS
    if small67 and not NO_SEEDS67:
        try:
            _seeds67_stage(Xs67, Xhs67, Y, fold, heads, configs, oof, test, deadline, time.time() - t_start)
        except _StrictError:
            raise
        except Exception as e:
            _log("seeds67 failed:", repr(e))

    # v67: per-item-CV ridge as an extra blend candidate (smaller instruments)
    extra = {}
    if small67 and not NO_RIDGE67 and time.time() < deadline - 60:
        try:
            r = _ridge67_candidate(Xs67, Y, Xhs67, fold, heads, min(deadline - 50, time.time() + RIDGE67_FRAC * (deadline - time.time())))
            if r is not None:
                extra["ridge67"] = r
        except _StrictError:
            raise
        except Exception as e:
            _log("ridge67 failed:", repr(e))

    # larger/deeper shared-trunk MLP, seed-averaged, extra candidate on large instruments only
    big = None
    big_seed_cost = 0.0
    if not NO_BIG and heads.J >= BIG_MIN_ITEMS and done == N_FOLDS:
        try:
            t_b = time.time()
            big_deadline = deadline - 150
            h, d_, wd_, lr_, ep_, dep_ = BIG_CFG
            ob = np.zeros((n, heads.total), np.float32)
            tb = np.zeros((len(held_pos), heads.total), np.float32)
            ok_big = True
            for f in range(N_FOLDS):
                tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                for sd in range(BIG_SEEDS):
                    net = _train(Xv[tr], Y[tr], Xv[va], Y[va], heads, h, d_, wd_, lr_, ep_, big_deadline, seed=100 * (sd + 1) + f, depth=dep_)
                    ob[va] += _predict_flat(net, Xv[va], heads) / BIG_SEEDS
                    tb += _predict_flat(net, Xh, heads) / (BIG_SEEDS * N_FOLDS)
                    del net
                per = (time.time() - t_b) / (f + 1)
                _log(f"mlp_big fold {f} ({time.time() - t_b:.0f}s)")
                if f < N_FOLDS - 1 and time.time() + 1.3 * per > big_deadline:
                    ok_big = False
                    break
            if ok_big and np.isfinite(ob).all() and np.isfinite(tb).all():
                big = (ob, tb)
                big_seed_cost = (time.time() - t_b) / BIG_SEEDS
        except Exception as e:
            _log("mlp_big failed:", repr(e))
            big = None

    # latent-class mixture and round-crossed trunk: cheap extra candidates, each under its own deadline
    if not NO_LCA and heads.J >= BIG_MIN_ITEMS and done == N_FOLDS and time.time() < deadline - 90:
        try:
            r = _lca_candidate(Xv, Y, Xh, Yoh, fold, heads, min(deadline - 80, time.time() + 0.25 * (deadline - time.time())))
            if r is not None:
                extra["lca"] = r
        except Exception as e:
            _log("lca failed:", repr(e))
    if not NO_XL and heads.J >= BIG_MIN_ITEMS and done == N_FOLDS and big is not None and time.time() < deadline - 200:
        try:
            r = _wide_candidate(Xv, Y, Xh, fold, heads, XL_CFG, XL_SEEDS, min(deadline - 120, time.time() + 0.3 * (deadline - time.time())), 900, "mlp_xl")
            if r is not None:
                extra["mlp_xl"] = r
        except Exception as e:
            _log("mlp_xl failed:", repr(e))
    rg = _round_columns(given, rec, {g: rec[g]["values"] for g in given})
    if not NO_ROUND and rg is not None and heads.J >= BIG_MIN_ITEMS and done == N_FOLDS and time.time() < deadline - 120:
        try:
            Xr = _round_features(frame, X, rg, idx, len(rec[rg]["values"]))
            r = _round_candidate(Xr[vis], Y, Xr[held_pos], fold, heads, min(deadline - 100, time.time() + 0.3 * (deadline - time.time())))
            if r is not None:
                extra["mlp_round"] = r
        except Exception as e:
            _log("mlp_round failed:", repr(e))
    # mask-and-recover pretrained trunk (all instruments): extra candidate under its own deadline
    if not NO_MR and not _wb_safe(given) and done == N_FOLDS and time.time() < deadline - 120:
        try:
            r = _mr_candidate(Xv, Y, Xh, Yoh, fold, heads, min(deadline - 100, time.time() + MR_SHARE * (deadline - time.time())))
            if r is not None:
                extra["mlp_mr"] = r
        except _StrictError:
            raise
        except Exception as e:
            _log("mlp_mr failed:", repr(e))
    # per-head checkpoint selection: each of the many heads stops at its own best epoch (UNHCR-sized only)
    if not NO_PH and heads.J >= BIG_MIN_ITEMS and done == N_FOLDS and time.time() < deadline - 120:
        try:
            r = _ph_candidate(Xv, Y, Xh, fold, heads, min(deadline - 110, time.time() + 0.3 * (deadline - time.time())))
            if r is not None:
                extra["ph"] = r
        except Exception as e:
            _log("ph failed:", repr(e))

    # v67: fan-out-weighted parent priority (UNHCR-sized): own seed-averaged trunk on the high fan-out gate parents, offered to the per-item blend
    fan = None
    _fanout_w.clear()
    if not NO_FANOUT and heads.J >= FANOUT_MIN_ITEMS and done == N_FOLDS:
        try:
            fw = _fanout_parents(rec, pred, idx, heads)
            _fanout_w.update(fw)
            _log("fanout parents: " + str({pred[j]: round(w, 2) for j, w in fw.items()}))
            if fw and time.time() < deadline - 200:
                fan = _fanout_candidate(Xv, Y, Xh, fold, heads, list(fw), oof["mlp"], test["mlp"],
                                        min(deadline - 150, time.time() + FANOUT_FRAC * (deadline - time.time())))
        except _StrictError:
            raise
        except Exception as e:
            _log("fanout failed:", repr(e))
            fan = None
            _fanout_w.clear()

    # v68: typed location-scale ordinal candidate on the eligible ordered-block items (non-UNHCR-sized schemas), offered to the per-item blend
    ordt = None
    if not NO_ORDTYPED and heads.J < ORDTYPED_MAX_ITEMS and done == N_FOLDS and time.time() < deadline - fm_reserve - 120:
        try:
            t_o = time.time()
            left = deadline - fm_reserve - t_o
            ordt = _ordtyped_candidate(Xv, Y, Xh, fold, heads, rec, pred, t_o + min(ORDTYPED_CAP_S, ORDTYPED_FRAC * left))
            _log(f"ordtyped stage {time.time() - t_o:.0f}s: " + ("added" if ordt is not None else "none"))
        except _StrictError:
            raise
        except Exception as e:
            _log("ordtyped failed:", repr(e)[:200])
            ordt = None

    # v69: soft copy-through (band transition) and shared tracer-response candidates (non-UNHCR-sized schemas), each self-gated
    bandt = None
    if not NO_BANDT and heads.J < V69_MAX_ITEMS and done == N_FOLDS and time.time() < deadline - fm_reserve - 60:
        try:
            t_b = time.time()
            bandt = _bandt_candidate(frame, given, rec, pred, opts, idx, heads, vis, held_pos, Y, fold, oof["mlp"], test["mlp"],
                                     t_b + min(BANDT_CAP_S, 0.05 * max(deadline - fm_reserve - t_b, 0)))
            _log(f"bandt stage {time.time() - t_b:.1f}s: " + ("added" if bandt is not None else "none"))
        except _StrictError:
            raise
        except Exception as e:
            _log("bandt failed:", repr(e)[:200])
            bandt = None
    tracer = None
    if not NO_TRACER and heads.J < V69_MAX_ITEMS and done == N_FOLDS and time.time() < deadline - fm_reserve - 60:
        try:
            t_t = time.time()
            tracer = _tracer_candidate(Xv, Xh, opts, rec, pred, heads, Y, fold, oof["mlp"], test["mlp"],
                                       t_t + min(TRACER_CAP_S, 0.05 * max(deadline - fm_reserve - t_t, 0)))
            _log(f"tracer stage {time.time() - t_t:.1f}s: " + ("added" if tracer is not None else "none"))
        except _StrictError:
            raise
        except Exception as e:
            _log("tracer failed:", repr(e)[:200])
            tracer = None

    # empirical-Bayes profile estimator shrunk toward the MLP; k chosen on OOF
    best_k, best_ll = None, -np.inf
    oof_prof = {}
    for k in PROFILE_K:
        cand = np.zeros_like(oof["mlp"])
        for f in range(done):
            tr, va = np.flatnonzero(covered & (fold != f)), np.flatnonzero(fold == f)
            cand[va] = _profile_flat(pv[tr], Yoh[tr], pv[va], oof["mlp"][va], k)
        ll = _total_ll(cand[covered], Y[covered], heads)
        oof_prof[k] = cand
        if ll > best_ll:
            best_k, best_ll = k, ll
    oof["profile"] = oof_prof[best_k]
    test["profile"] = _profile_flat(pv, Yoh, ph, test["mlp"], best_k)

    names = ["marginal", "linear", "mlp", "profile"]
    if big is not None:
        oof["mlp_big"], test["mlp_big"] = big
        names.append("mlp_big")

    for c, (o_, t_) in extra.items():
        oof[c], test[c] = o_, t_
        names.append(c)

    # hierarchical back-off cells (cheap; own try so a failure leaves the other candidates intact)
    if not NO_CELLS and not _late(120, "cells"):
        try:
            t_c = time.time()
            res = None
            if not NO_EBCELLS:
                try:
                    res = _eb_cells_candidate(frame, given, idx, vis, held_pos, Yoh, Y, fold, covered, heads)
                except Exception as e:
                    _log("eb cells failed, single-alpha cells:", repr(e))
                    res = None
            if res is None:
                res = _cells_candidate(frame, given, idx, vis, held_pos, Yoh, Y, fold, covered, heads)
            oof["cells"], test["cells"] = res
            names.append("cells")
            _log(f"cells done ({time.time() - t_c:.0f}s)")
        except Exception as e:
            _log("cells failed:", repr(e))

    # multi-output ExtraTrees (small instruments; needs every fold; off by default)
    if not NO_FOREST and done == N_FOLDS and heads.J <= FOREST_MAX_ITEMS and time.time() < deadline - 60:
        try:
            t_f = time.time()
            res = _forest_candidate(Xv, Y, Xh, fold, heads, deadline - 30)
            if res is not None:
                oof["forest"], test["forest"] = res
                names.append("forest")
            _log(f"forest {'done' if res is not None else 'skipped (time)'} ({time.time() - t_f:.0f}s)")
        except Exception as e:
            _log("forest failed:", repr(e))

    # TabPFN-3.5 on every item with remaining loss, most room first, until the instrument's deadline less a reserve;
    # TabICLv2 takes the window only when TabPFN is unavailable
    pfn, tab, kum = None, None, None
    _pfn_cov.update(k_done=0, covered=0, order=0)
    _kumo_hold["s"] = 0.0
    if not NO_KUMO and done == N_FOLDS and _kumo_present():
        _kumo_hold["s"] = KUMO_SHARE * max(deadline - fm_reserve - time.time(), 0.0)
    gp_events, gp_ev = [], None
    _gp_hold["s"] = 0.0
    if not NO_GATEPFN and done == N_FOLDS:
        try:
            gp_events, gp_dropped = _gate_events(Y, rec, pred, idx, heads, oof["mlp"])
            _log(f"gate events: {len(gp_events)} kept, {len(gp_dropped)} dropped " + str([(len(c), why, round(float(v), 3)) for c, why, v in gp_dropped][:6]))
        except Exception as e:
            _log("gate events failed:", repr(e)[:200])
            gp_events = []
        if gp_events:
            _gp_hold["s"] = GATEPFN_SHARE * max(deadline - fm_reserve - time.time(), 0.0)
    _distill_hold["s"] = 0.0
    if not NO_DISTILL and not _wb_safe(given) and done == N_FOLDS and DISTILL_TEACHER == "pfn" and not NO_TABPFN and heads.J > 0:
        _distill_hold["s"] = DISTILL_SHARE * max(deadline - fm_reserve - _gp_hold["s"] - time.time(), 0.0)
    if not NO_TABPFN and done == N_FOLDS:
        stage_deadline = deadline - fm_reserve - _gp_hold["s"] - _distill_hold["s"]
        if stage_deadline - time.time() > 20:
            t_p = time.time()
            _distill_rec["d"] = {}
            _distill_rec["on"] = not NO_DISTILL
            try:
                pfn = _tabicl_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof["mlp"], oof["marginal"],
                                        stage_deadline, n_folds=TABPFN_FOLDS, by_loss=True, kind="tabpfn",
                                        min_loss=TABPFN_MIN_LOSS,
                                        anytime_frac=ANYTIME_PFN_FRAC if big is not None and BIG_SEEDS < ANYTIME_BIG_SEEDS else 1.0)
            except _StrictError:
                raise
            except Exception as e:
                _log("tabpfn failed:", repr(e))
                pfn = None
            _distill_rec["on"] = False
            _log(f"tabpfn stage {time.time() - t_p:.0f}s")
    # Kumo-Tabular: the window held back from TabPFN's repeat passes (and whatever else is left, up to 1.5x that hold)
    if not NO_KUMO and done == N_FOLDS and _kumo_hold["s"] > 0:
        stage_deadline = min(deadline - fm_reserve - _gp_hold["s"] - _distill_hold["s"], time.time() + 1.5 * _kumo_hold["s"])
        if stage_deadline - time.time() > 15:
            t_k = time.time()
            try:
                kum = _kumo_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof["mlp"], stage_deadline,
                                      inst_deadline=deadline - fm_reserve)
            except Exception as e:
                _log("kumo failed:", repr(e)[:200])
                _kumo_state["call_fail"] = True
                _kumo_state["stage"] = "call_failed"
                _kumo_release()
                kum = None
            _log(f"kumo stage {time.time() - t_k:.0f}s")
        _kumo_hold["s"] = 0.0
    _kumo_check(kum, done)
    # gate-event model: a binary TabPFN per shared asked / not-asked event, on the window held back for it and what is left
    if gp_events:
        hold_s = _gp_hold["s"]
        _gp_hold["s"] = 0.0
        stage_deadline = min(deadline - fm_reserve, time.time() + max(1.5 * hold_s, hold_s))
        if stage_deadline - time.time() > 20:
            t_g = time.time()
            try:
                gp_ev = _gatepfn_stage(frame, given, opts, vis, held_pos, Y, fold, heads, gp_events, stage_deadline)
            except Exception as e:
                _log("gate events stage failed:", repr(e)[:200])
                gp_ev = None
            _log(f"gate events stage {time.time() - t_g:.0f}s: " + (str(getattr(_gatepfn_stage, "last", None)) if gp_ev else "none"))
        if gp_ev is not None and not covered.all():
            gp_ev = None
    # student MLP distilled from TabPFN (or, in tests, from the blend of the other candidates), with transduction
    dst = None
    if not NO_DISTILL and not _wb_safe(given) and done == N_FOLDS:
        try:
            teacher = None
            if DISTILL_TEACHER == "blend":
                teacher = _distill_blend_teacher(oof, test, _distill_ft, Y, heads, n)
                st_dl = deadline - fm_reserve
            elif pfn is not None:
                teacher = (pfn[0], pfn[1], pfn[2], pfn[3], dict(_distill_rec["d"]))
                st_dl = min(deadline - fm_reserve, time.time() + max(1.5 * _distill_hold["s"], 20.0))
            if teacher is not None and st_dl - time.time() > 20:
                t_d = time.time()
                dst = _distill_candidate(Xv, Xh, Y, fold, heads, teacher, st_dl)
                _log(f"distill stage {time.time() - t_d:.0f}s: " + ("added" if dst is not None else "none"))
        except _StrictError:
            raise
        except Exception as e:
            _log("distill failed:", repr(e)[:200])
            dst = None
    _distill_hold["s"] = 0.0
    if PROBE and not NO_TABPFN and heads.J >= BIG_MIN_ITEMS:
        if pfn is None or _pfn_cov["order"] == 0 or _pfn_cov["k_done"] < TABPFN_MAX_FOLDS or _pfn_cov["covered"] < _pfn_cov["order"]:
            raise _StrictError(f"coverage: folds {_pfn_cov['k_done']}/{TABPFN_MAX_FOLDS}, items {_pfn_cov['covered']}/{_pfn_cov['order']}")
    tab_add = (not NO_TABICL_ADD and pfn is not None and heads.J < BIG_MIN_ITEMS)
    if not NO_TABICL and done == N_FOLDS and not LEGACY and (NO_TABPFN or not _tabpfn_ready() or tab_add):
        # next to TabPFN (small instruments) TabICL only gets the time TabPFN left, inside its own cap and the reserve
        stage_deadline = deadline - fm_reserve
        if tab_add:
            stage_deadline = min(stage_deadline, time.time() + TABICL_ADD_TOTAL)
        if stage_deadline - time.time() > 30:
            t_t = time.time()
            try:
                tab = _tabicl_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof["mlp"], oof["marginal"],
                                        stage_deadline, n_folds=(TABICL_ADD_FOLDS if tab_add else TABPFN_FOLDS),
                                        by_loss=True, min_loss=TABPFN_MIN_LOSS,
                                        conservative=tab_add and not TABICL_ADD_LINEAR)
            except Exception as e:
                _log("tabicl failed:", repr(e))
                tab = None
            _log(f"tabicl stage {time.time() - t_t:.0f}s")

    # time still left before the reserve: further whole seeds of the big trunk, folded into its seed average
    if not NO_ANYTIME and big is not None and not NO_BIG:
        try:
            _more_big_seeds(Xv, Y, Xh, fold, heads, big_seed_cost, deadline - fm_reserve, oof, test)
        except Exception as e:
            _log("extra big seeds failed:", repr(e))

    # TabICLv2 candidate on the highest-gain items, inside a hard time cap (needs every fold for the OOF base)
    if not NO_TABICL and done == N_FOLDS and LEGACY:
        cap = min(TABICL_TOTAL - _tabicl_state["spent"], 0.45 * TABICL_TOTAL)
        stage_deadline = min(deadline - 45, time.time() + cap)
        if cap > 15 and stage_deadline - time.time() > 15:
            t_t = time.time()
            try:
                tab = _tabicl_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof["mlp"], oof["marginal"], stage_deadline)
            except Exception as e:
                _log("tabicl failed:", repr(e))
                tab = None
            _tabicl_state["spent"] += time.time() - t_t

    # gradient-boosted trees on the highest-nonlinearity items, after TabICL so the earlier stages keep their budgets
    gb = None
    if not NO_GBDT and done == N_FOLDS and heads.J >= GBDT_MIN_ITEMS:
        stage_deadline = min(deadline - 60, time.time() + GBDT_TOTAL)
        if stage_deadline - time.time() > 15:
            t_g = time.time()
            try:
                gb = _gbdt_candidate(frame, given, idx, vis, held_pos, Y, fold, heads, oof["mlp"], oof["linear"], oof["marginal"], stage_deadline)
            except Exception as e:
                _log("gbdt failed:", repr(e))
                gb = None
            _log(f"gbdt stage {time.time() - t_g:.0f}s")

    # broad TabICL coverage for a UNHCR-sized instrument: its own budget, one context fold per item (twice the
    # coverage per second), items ordered by the MLP's remaining loss so near-deterministic heads are skipped
    tab_big = None
    if not NO_TABICL_BIG and done == N_FOLDS and heads.J >= BIG_MIN_ITEMS:
        stage_deadline = min(deadline - 50, time.time() + TABICL_BIG_TOTAL)
        if stage_deadline - time.time() > 20:
            t_tb = time.time()
            try:
                tab_big = _tabicl_candidate(frame, given, opts, vis, held_pos, Y, fold, heads, oof["mlp"], oof["marginal"],
                                            stage_deadline, n_folds=TABICL_BIG_FOLDS, by_loss=True)
            except Exception as e:
                _log("tabicl_big failed:", repr(e))
                tab_big = None
            _log(f"tabicl_big stage {time.time() - t_tb:.0f}s")

    if VERBOSE:
        for c in names:
            _log(f"OOF skill-ish {c}: {_total_ll(oof[c][covered], Y[covered], heads):.4f}  (profile k={best_k})")

    flags = None
    if not NO_CAPS and not _late(90, "support caps"):
        try:
            flags = _cap_flags(frame, given, idx, rec, pred, vis, Y, heads)
            _log(f"support caps: {int(flags[0].any(0).sum())} items with never-strata, {int(flags[1].any(0).sum())} with always-strata")
        except Exception as e:
            _log("support caps failed:", repr(e))
    rows_oof, rows_test = vis[covered], held_pos

    def capped(M, rows):
        return _apply_caps(M, rows, flags, heads) if flags is not None else M

    given_set = set(given)
    xf = None
    if not NO_XFIT and not _late(90, "xfit rules"):
        try:
            xf = _xfit_rules(frame, given, idx, rec, pred, heads, vis, held_pos, Y, fold, covered, given_set,
                             flags is not None, not NO_BBFIT)
        except Exception as e:
            _log("cross-fitted rules failed, using all-rows rules:", repr(e))
            xf = None
    vis_frame = frame.iloc[vis[covered]]
    if xf is not None:
        parents = sorted({rec[n]["gate"]["parent"] for n in pred if rec[n].get("gate") and rec[n]["gate"]["parent"] in given_set})
        fold_frames = {f: vis_frame[parents].iloc[np.flatnonzero(xf["fo"] == f)] for f in xf["gate"]}
        test_trust, test_nside = getattr(_gate_override, "trust", {}), getattr(_gate_override, "nside", {})

        def prep_oof(M):
            M = M[covered].copy()
            for f, (trust, nside) in xf["gate"].items():
                msk = xf["fo"] == f
                M[msk] = _gate_override(M[msk], fold_frames[f], pred, rec, heads, given_set, trust, nside, xf["ab"])
            return cap_oof(M)

        def cap_oof(M):
            fl = xf["oof_caps"]
            if fl is None:
                return M
            return _apply_caps_q(M, fl, heads) if xf["q"] else _apply_caps(M, np.arange(len(M)), fl[:2], heads)

        def cap_test(M):
            fl = xf["test_caps"]
            if fl is None:
                return M
            return _apply_caps_q(M, fl, heads) if xf["q"] else _apply_caps(M, np.arange(len(M)), fl[:2], heads)

        def gate_test(P_):
            return _gate_override(P_, frame.iloc[held_pos], pred, rec, heads, given_set, test_trust, test_nside, xf["ab"])
    else:
        def prep_oof(M):
            return capped(_gate_override(M[covered].copy(), vis_frame, pred, rec, heads, given_set), rows_oof)

        def cap_oof(M):
            return capped(M, rows_oof)

        def cap_test(M):
            return capped(M, rows_test)

        def gate_test(P_):
            return _gate_override(P_, frame.iloc[held_pos], pred, rec, heads, given_set)

    # per-item blend on OOF rows
    for c in names:
        oof[c] = prep_oof(oof[c])
        test[c] = cap_test(test[c])
    Yc = Y[covered]
    partials = []                                  # (name, oof, test, have, rows covered by the candidate)
    for name, res in (("tabicl", tab), ("gbdt", gb), ("tabicl_big", tab_big), ("tabpfn", pfn), ("kumo", kum), ("distill", dst), ("fanout", fan), ("ordtyped", ordt), ("bandt", bandt), ("tracer", tracer)):
        if res is not None:
            p_oof, p_test, p_have, p_rows = res
            p_oof = prep_oof(p_oof)
            partials.append((name, p_oof, cap_test(p_test), p_have, p_rows))
    nb = len(names)
    half = fold[covered] % 2
    del LOGPOOL_STATS[:]
    P = np.zeros((len(held_pos), heads.total), np.float32)
    blended_oof = np.zeros_like(oof["mlp"])
    wsum = np.zeros(nb + len(partials))
    diag = {p[0]: [] for p in partials}
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Yc[:, j] >= 0
        act = [k for k, p in enumerate(partials) if p[3][j]]
        covered_rows = np.ones(len(Yc), bool)
        for k in act:
            covered_rows &= partials[k][4]
        rows = ok & covered_rows
        if act and rows.sum() >= 20:
            w = _fit_weights([oof[c][rows, s] for c in names] + [partials[k][1][rows, s] for k in act], Yc[rows, j], _fanout_w.get(j, 1.0))
            wsum[:nb] += w[:nb]
            for i, k in enumerate(act):
                wsum[nb + k] += w[nb + i]
            P[:, s] = sum(wi * test[c][:, s] for wi, c in zip(w[:nb], names)) + sum(w[nb + i] * partials[k][2][:, s] for i, k in enumerate(act))
            full = sum(wi * oof[c][:, s] for wi, c in zip(w[:nb], names)) + sum(w[nb + i] * partials[k][1][:, s] for i, k in enumerate(act))
            a_lp = None if (NO_LOGPOOL or _late(90)) else _logpool_choice([oof[c][rows, s] for c in names] + [partials[k][1][rows, s] for k in act],
                                                           Yc[rows, j], half[rows], w)
            if a_lp is not None:
                P[:, s] = _lp_probs(a_lp, _lp_stack([test[c][:, s] for c in names] + [partials[k][2][:, s] for k in act]))
                full = _lp_probs(a_lp, _lp_stack([oof[c][:, s] for c in names] + [partials[k][1][:, s] for k in act]))
            w_oth = w[:nb] / max(w[:nb].sum(), 1e-12)
            oth = sum(wi * oof[c][:, s] for wi, c in zip(w_oth, names))
            blended_oof[:, s] = np.where(covered_rows[:, None], full, oth)
            if VERBOSE:
                yj = Yc[rows, j]
                f_ = lambda M: np.log(_scored(M[rows][:, s].astype(np.float64) + 1e-12, heads.widths[j])[np.arange(rows.sum()), yj]).mean()
                names_act = [partials[k][0] for k in act]
                for i, k in enumerate(act):
                    dpf = f_(partials[k][1]) - f_(partials[act[names_act.index("tabpfn")]][1]) if "tabpfn" in names_act and partials[k][0] == "tabicl" else float("nan")
                    diag[partials[k][0]].append((j, w[nb + i], f_(partials[k][1]) - f_(oof["mlp"]), dpf))
            continue
        if ok.sum() < 20:
            w = np.eye(nb)[0]
        else:
            w = _fit_weights([oof[c][ok, s] for c in names], Yc[ok, j])
        wsum[:nb] += w
        P[:, s] = sum(wi * test[c][:, s] for wi, c in zip(w, names))
        blended_oof[:, s] = sum(wi * oof[c][:, s] for wi, c in zip(w, names))
        a_lp = None if (NO_LOGPOOL or ok.sum() < 20 or _late(90)) else _logpool_choice([oof[c][ok, s] for c in names], Yc[ok, j], half[ok], w)
        if a_lp is not None:
            P[:, s] = _lp_probs(a_lp, _lp_stack([test[c][:, s] for c in names]))
            blended_oof[:, s] = _lp_probs(a_lp, _lp_stack([oof[c][:, s] for c in names]))
    if VERBOSE and LOGPOOL_STATS:
        sw = [t for t, sw_ in LOGPOOL_STATS if sw_]
        _log(f"logpool: {len(sw)}/{len(LOGPOOL_STATS)} items switched, mean total weight {np.mean(sw) if sw else 0:.3f}")
    if VERBOSE:
        _log("mean blend weights: " + ", ".join(f"{c}={v / heads.J:.2f}" for c, v in zip(names + [p[0] for p in partials], wsum)))
        _log(f"OOF skill-ish blend (in-sample weights): {_total_ll(blended_oof, Yc, heads):.4f}")
        for name, dg in diag.items():
            if dg:
                d_ = np.array([x[2] for x in dg]); w_ = np.array([x[1] for x in dg])
                _log(f"{name} vs mlp on {len(dg)} items: mean log-score diff {d_.mean():+.4f} nats, better on {int((d_ > 0).sum())}, mean blend weight {w_.mean():.2f}")
                v_ = np.array([x[3] for x in dg])
                if np.isfinite(v_).any():
                    _log(f"{name} vs tabpfn on {int(np.isfinite(v_).sum())} items: mean log-score diff {np.nanmean(v_):+.4f} nats; items covered {[x[0] for x in dg]}; weights {[round(float(x[1]), 2) for x in dg]}")

    P = cap_test(gate_test(P))
    _snap["P"] = P.copy()
    if os.environ.get("SB_V70_POSTFAIL") == "1":
        raise RuntimeError("v70 post-blend failure hook")   # test hook

    Bfin = blended_oof
    # gate hybrid (direct vs composed gate probability, shrunk toward the direct gate)
    if not NO_GATEHYB and not _late(60, "gate hybrid"):
        try:
            Pc, Bc = P.copy(), blended_oof.copy()
            stats = _gate_hybrid(Bc, Pc, Yc, rec, pred, idx, heads, ev=gp_ev)
            cap_test(Pc)
            cap_oof(Bc)
            acc = [x for x in stats if x[1] >= GATE_MIN_GAIN]
            if gp_ev is not None:
                es = getattr(_gate_hybrid, "ev_stats", {})
                _log(f"gate events: {sum(1 for v in es.values() if v['accepted'])}/{len(es)} gated items took the event column")
            if VERBOSE:
                _log(f"gate hybrid: {len(stats)} PREDICT-parent gated items fitted, {len(acc)} accepted; "
                     f"total gain {sum(x[1] for x in acc):.1f} nats; OOF skill {_total_ll(blended_oof, Yc, heads):.4f} -> {_total_ll(Bc, Yc, heads):.4f}")
                for j, gain, th in sorted(acc, key=lambda x: -x[1])[:5]:
                    _log(f"    {pred[j][:30]:30s} gain {gain:6.1f} theta {np.round(th, 2)}")
            if np.isfinite(Pc).all():
                P, Bfin = Pc, Bc
        except Exception as e:
            _log("gate hybrid failed:", repr(e))

    if not NO_SIBPOOL and heads.J >= SIBPOOL_MIN_ITEMS and not _late(60, "sibpool"):
        try:
            t0 = time.time()
            Pc, Bc = P.copy(), Bfin.copy()
            info = _sibpool(Bc, Pc, Yc, rec, pred, idx, heads, fold[covered] % 2)
            cap_test(Pc)
            cap_oof(Bc)
            if np.isfinite(Pc).all() and np.isfinite(Bc).all():
                P, Bfin = Pc, Bc
            _log(f"sibling pooling: {info[0]}/{info[1]} groups accepted, nested gains {info[2]}, {time.time() - t0:.2f}s")
        except Exception as e:
            _log("sibling pooling failed:", repr(e))

    if not NO_GHOST and not _late(60, "ghost"):
        try:
            P = _ghost_component(P, Bfin, Yc, frame, vis, held_pos, covered, fold, rec, pred, given, heads, Y, idx)
        except Exception as e:
            _log("ghost component failed:", repr(e))

    if not NO_TAILCAL and not _late(45, "tailcal"):
        try:
            t0 = time.time()
            P2, B2 = _tailcal(P, Bfin, Yc, rec, pred, heads, fold[covered] % 2)
            if np.isfinite(P2).all():
                P = P2
                Bfin = B2
            _log(f"tail recalibration: {time.time() - t0:.2f}s")
        except Exception as e:
            _log("tail recalibration failed:", repr(e))

    if not NO_TEMPER and not _late(45, "temper"):
        try:
            t0 = time.time()
            P, B_t = _temper(P, Bfin, Yc, frame, vis, held_pos, covered, fold, given, heads, Y, Yoh, pred)
            if not NO_V68FIX:
                Bfin = B_t
            _log(f"tempered mixture: {time.time() - t0:.2f}s")
        except Exception as e:
            _log("tempered mixture failed:", repr(e))

    if not NO_UNMIX and heads.J >= UNMIX_MIN_ITEMS and not _late(30, "unmix"):
        try:
            P = _unmix(P, Bfin, Yc, covered, fold, heads)
        except Exception as e:
            _log("unmix failed:", repr(e))

    return P


def _waterfill(V, K):
    """argmax_q sum_k v_k log(EPS + (1 - K EPS) q_k) per row (the grader mixes a flat EPS into every vector): q_k = max(0, v_k/nu - EPS)/(1 - K EPS)."""
    lo = np.full(len(V), 1e-9)
    hi = np.full(len(V), 1e3)
    for _ in range(45):
        nu = np.sqrt(lo * hi)
        s_ = (np.clip(V / nu[:, None] - EPS, 0, None) / (1 - K * EPS)).sum(1)
        big = s_ > 1
        lo = np.where(big, nu, lo)
        hi = np.where(big, hi, nu)
    q = np.clip(V / np.sqrt(lo * hi)[:, None] - EPS, 0, None)
    return q / np.maximum(q.sum(1, keepdims=True), 1e-300)


def _unmix(P, B, Yc, covered, fold, heads):
    """Final vectors replaced by (1 - a) v + a q with q the deconvolution of the grader's flat mix; a fitted on the honest OOF vectors.
    On only when a two-way nested check (fit a on one fold-parity half, score on the other) gains on both halves."""
    if UNMIX_FAIL:
        raise RuntimeError("SB_UNMIX_FAIL")
    J = heads.J
    off = heads.offsets
    par = fold[covered] % 2
    Bn, Qo = {}, {}
    Pn, Qt = {}, {}
    for j in range(J):
        K = int(heads.widths[j])
        if K < 2:
            continue
        sl = slice(off[j], off[j + 1])
        v = np.clip(np.asarray(B[:, sl], np.float64), 0, None)
        v /= np.maximum(v.sum(1, keepdims=True), 1e-300)
        Bn[j], Qo[j] = v, _waterfill(v, K)
        pv = np.clip(np.asarray(P[:, sl], np.float64), 0, None)
        pv /= np.maximum(pv.sum(1, keepdims=True), 1e-300)
        Pn[j], Qt[j] = pv, _waterfill(pv, K)
    grid = (0.0,) + UNMIX_ALPHAS
    S = np.zeros((len(grid), 2))           # total log score per alpha per parity half
    for j, v in Bn.items():
        K = int(heads.widths[j])
        y = Yc[:, j]
        ok = y >= 0
        rows = np.flatnonzero(ok)
        for gi, a in enumerate(grid):
            q = (1 - a) * v[rows] + a * Qo[j][rows]
            lp = np.log(EPS + (1 - K * EPS) * q[np.arange(len(rows)), y[rows]])
            for h in (0, 1):
                S[gi, h] += lp[par[rows] == h].sum()
    gains = []
    for h in (0, 1):
        best = int(np.argmax(S[:, 1 - h]))          # strength fitted on the other half
        gains.append(float(S[best, h] - S[0, h]))
    best_all = int(np.argmax(S.sum(1)))
    on = bool(best_all > 0 and all(g > UNMIX_MIN_GAIN for g in gains))
    _log(f"unmix: nested gains {np.round(gains, 2)} nats, alpha {grid[best_all]} -> {'ON' if on else 'off'}")
    if not on:
        return P
    a = grid[best_all]
    Q = np.array(P, np.float32, copy=True)
    for j in Pn:
        Q[:, off[j]:off[j + 1]] = ((1 - a) * Pn[j] + a * Qt[j]).astype(np.float32)
    return Q if np.isfinite(Q).all() else P


# ----------------------------------------------------------------------------- breakoff (ghost) rows

def _ghost_codes(frame, given):
    """Group code per row from the GIVEN round column and a country-like GIVEN column (by name); wave code per group."""
    low = {g: g.lower() for g in given}
    wave = next((g for g in given if low[g] == "wave"), None)
    country = next((g for g in given if re.search(r"country|asylum", low[g]) and 1 < frame[g].nunique() <= 40), None)
    n = len(frame)
    w = pd.factorize(frame[wave].astype(str))[0] if wave else np.zeros(n, np.int64)
    key = frame[country].astype(str).to_numpy() if country else np.zeros(n, np.int64).astype(str)
    gcode, uniq = pd.factorize(pd.Series(key) + "|" + pd.Series(w.astype(str)))
    wave_of_g = np.zeros(len(uniq), np.int64)
    wave_of_g[gcode] = w
    return gcode.astype(np.int64), wave_of_g, wave, country


def _ghost_routed(frame, vis, Y, gj, rec, pred, idx):
    """(n_vis, Jg) bool: the declared gate of each gated item says 'asked' given the row's visible parent value."""
    pj = {n: j for j, n in enumerate(pred)}
    R = np.zeros((len(vis), len(gj)), bool)
    for q, j in enumerate(gj):
        g = rec[pred[j]]["gate"]
        par = g["parent"]
        if par in pj:
            codes = Y[:, pj[par]]
        elif par in idx:
            codes = frame[par].map(idx[par]).fillna(-1).astype(int).to_numpy()[vis]
        else:
            continue
        obs = [idx[par][v] for v in g["observed_if"] if v in idx[par]]
        R[:, q] = np.isin(codes, obs)
    return R


def _ghost_flag(Y, gj, sent, gcode, wave_of_g, Rt):
    """Per visible row: last substantive gated position (-1 none) and whether the row is a breakoff: every later gated item is the
    sentinel although routing asks it, and the chance of that under the group's asked rates is below GHOST_ALPHA (three items
    asked in 99% of rows give 1e-6)."""
    G = len(wave_of_g)
    A = (Y[:, gj] >= 0) & (Y[:, gj] != sent[None])
    n, Jg = A.shape
    last = np.where(A.any(1), Jg - 1 - np.argmax(A[:, ::-1], 1), -1)

    def evidence_of(rows):
        R = Rt[rows]
        rate = (A[rows] & R).sum(0) / np.maximum(R.sum(0), 1)
        return np.where(R.sum(0) >= 30, -np.log1p(-np.minimum(rate, GHOST_RATE_CAP)), 0.0)

    wev = {w: evidence_of(wave_of_g[gcode] == w) for w in np.unique(wave_of_g)}
    E = np.zeros((G, Jg))
    for g in range(G):
        rows = gcode == g
        E[g] = evidence_of(rows) if rows.sum() >= 200 else wev[wave_of_g[g]]
    cand = E[gcode] * Rt                                                        # surprise of a sentinel at each routed item
    suf = np.concatenate([np.cumsum(cand[:, ::-1], 1)[:, ::-1], np.zeros((n, 1))], 1)
    later = suf[np.arange(n), last + 1]
    return last, later >= GHOST_NATS, E


def _ghost_hazard(last, flag, gcode, wave_of_g, rows, Jg):
    """beta[g, q] = P(row of group g broke off before gated position q): cumulative breakoff share, KT-smoothed, pooled toward the wave."""
    G = len(wave_of_g)
    cnt = np.zeros((G, Jg + 1))
    sel = rows & flag
    np.add.at(cnt, (gcode[sel], last[sel] + 1), 1.0)
    cum = np.cumsum(cnt, 1)[:, :Jg]
    ng = np.bincount(gcode[rows], minlength=G).astype(float)
    W = wave_of_g.max() + 1
    wc = np.zeros((W, Jg)); wn = np.zeros(W)
    np.add.at(wc, wave_of_g, cum); np.add.at(wn, wave_of_g, ng)
    bw = (wc + 0.5) / (wn[:, None] + 1.0)
    return (cum + GHOST_PRIOR * bw[wave_of_g]) / (ng[:, None] + GHOST_PRIOR)


def _ghost_mix_slice(S, b):
    """One gated item: P(sentinel) <- b + (1-b) P(sentinel), answers <- (1-b) answers, renormalised."""
    b = b[:, None].astype(S.dtype)
    out = S * (1.0 - b)
    out[:, -1:] = b + (1.0 - b) * S[:, -1:]
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def _ghost_mix(M, beta, gj, heads):
    """Apply the mixture to every gated item of M (in place)."""
    for q, j in enumerate(gj):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        M[:, s0:s1] = _ghost_mix_slice(M[:, s0:s1], beta[:, q])
    return M


def _ghost_gains(Bo, Yc, gj, heads, rows, H, lams):
    """Summed log-score gain (grader-mixed) of the mixture on the OOF `rows` for each strength in lams; H is (len(rows), Jg)."""
    out = np.zeros(len(lams))
    for q, j in enumerate(gj):
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        y = Yc[rows, j]
        ok = y >= 0
        if not ok.any():
            continue
        S = Bo[rows[ok], s0:s1].astype(np.float64) + 1e-12
        K, yy, ii = heads.widths[j], y[ok], np.arange(ok.sum())
        base = np.log(_scored(S, K)[ii, yy]).sum()
        for li, lam in enumerate(lams):
            out[li] += np.log(_scored(_ghost_mix_slice(S, lam * H[ok, q]), K)[ii, yy]).sum() - base
    return out


def _ghost_component(P, Bo, Yc, frame, vis, held_pos, covered, fold, rec, pred, given, heads, Y, idx):
    """Mix a per-group breakoff hazard into the gated slots, scaled by a strength chosen out of fold (the learners already see part
    of the breakoff mass through the round and country features). On only when breakoff rows exist in the training data and a
    two-way nested check on the out-of-fold rows gains in both halves."""
    t0 = time.time()
    info = {"on": False, "n_flag": 0, "n_full": 0}
    _ghost_last.clear(); _ghost_last.update(info)
    gj = np.array([j for j in range(heads.J) if rec[pred[j]].get("gate")], dtype=int)
    if len(gj) < 4:
        return P
    sent = np.array([heads.widths[j] - 1 for j in gj])
    gall, wave_of_g, wave, country = _ghost_codes(frame, given)
    gv, gh = gall[vis], gall[held_pos]
    Jg = len(gj)
    Rt = _ghost_routed(frame, vis, Y, gj, rec, pred, idx)
    last, flag, E = _ghost_flag(Y, gj, sent, gv, wave_of_g, Rt)
    info["n_flag"], info["n_full"] = int(flag.sum()), int((flag & (last < 0)).sum())
    if VERBOSE:
        by = {int(g): int((flag & (gv == g)).sum()) for g in np.unique(gv[flag])}
        _log(f"ghost: wave col {wave}, country col {country}, {len(wave_of_g)} groups; breakoff rows {info['n_flag']} "
             f"(full {info['n_full']}) by group {by}")
    _ghost_last.update(info)
    if info["n_flag"] < GHOST_MIN_ROWS:
        return P
    cov_rows = np.flatnonzero(covered)
    gc, fc, lc, flc = gv[cov_rows], fold[cov_rows], last[cov_rows], flag[cov_rows]
    par = fc % 2
    lams = GHOST_LAMS
    allr = np.arange(len(cov_rows))

    def crossfit(fit_mask, rows):
        """Gains per strength on `rows`, hazards for each fold's rows from the fit rows of the other folds."""
        tot = np.zeros(len(lams))
        for f in np.unique(fc[rows]):
            r = rows[fc[rows] == f]
            H = _ghost_hazard(lc, flc, gc, wave_of_g, fit_mask & (fc != f), Jg)
            tot += _ghost_gains(Bo, Yc, gj, heads, r, H[gc[r]], lams)
        return tot

    gains = []
    for h in (0, 1):
        fit, ev = par == h, np.flatnonzero(par != h)
        cf = crossfit(fit, allr[fit])
        li = int(np.argmax(cf))
        lam = lams[li] if cf[li] > 0 else 0.0
        H = _ghost_hazard(lc, flc, gc, wave_of_g, fit, Jg)
        gains.append(float(_ghost_gains(Bo, Yc, gj, heads, ev, H[gc[ev]], [lam])[0]) if lam > 0 else 0.0)
    cf_all = crossfit(np.ones(len(cov_rows), bool), allr)
    li = int(np.argmax(cf_all))
    lam = float(lams[li]) if cf_all[li] > 0 else 0.0
    on = bool(gains[0] > 0 and gains[1] > 0 and sum(gains) > GHOST_MIN_GAIN and lam > 0)
    info.update(on=on, lam=lam, gains=gains, cf_gains=[float(x) for x in cf_all])
    if VERBOSE:
        Bn = Bo.copy()
        for f in np.unique(fc):
            H = _ghost_hazard(lc, flc, gc, wave_of_g, fc != f, Jg)
            r = np.flatnonzero(fc == f)
            Bn[r] = _ghost_mix(Bo[r].copy(), lam * H[gc[r]], gj, heads)
        info.update(oof_before=_total_ll(Bo, Yc, heads), oof_after=_total_ll(Bn, Yc, heads))
    if VERBOSE:
        Hall = _ghost_hazard(lc, flc, gc, wave_of_g, np.ones(len(cov_rows), bool), Jg)
        qs = [0, Jg // 4, Jg // 2, 3 * Jg // 4, Jg - 1]
        for g in range(len(wave_of_g)):
            if (gc == g).sum() and flc[gc == g].any():
                _log(f"ghost: group {g} hazard at gated pos {qs}: {np.round(Hall[g, qs], 4)}")
        _log(f"ghost: nested gains {np.round(gains, 1)} nats, cross-fit gains by strength {dict(zip(lams, np.round(cf_all, 1)))}, "
             f"strength {lam}, {'ON' if on else 'OFF'}; OOF skill {info['oof_before']:.4f} -> {info['oof_after']:.4f}; {time.time() - t0:.1f}s")
    _ghost_last.update(info)
    if not on:
        return P
    H = _ghost_hazard(last, flag, gv, wave_of_g, np.ones(len(vis), bool), Jg)
    Pn = _ghost_mix(P.copy(), lam * H[gh], gj, heads)
    return Pn if np.isfinite(Pn).all() else P


# ----------------------------------------------------------------------------- pooled tail recalibration

TAIL_P = 0.05          # recalibrate options predicted below this
TAIL_BINS = 24
TAIL_LO = -5.0         # log10 p floor of the bin grid
TAIL_PRIOR = 2.0       # pseudo-observations placed at each bin's own mean prediction (shrinks toward "p is right")
TAIL_MIN_GAIN = 2.0    # nats, total over both nested halves


def _pav(y, w):
    """Weighted isotonic regression (increasing) by pool-adjacent-violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww); wts.append(ww); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def _tail_fit(p, hit):
    """Return (xs, gs): bin mean log10 p and g = max(p, isotonic observed frequency); None if too few pairs."""
    if len(p) < 200:
        return None
    lp = np.clip(np.log10(np.maximum(p, 1e-12)), TAIL_LO, None)
    edges = np.linspace(TAIL_LO, np.log10(TAIL_P), TAIL_BINS + 1)
    b = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, TAIL_BINS - 1)
    n = np.bincount(b, minlength=TAIL_BINS).astype(np.float64)
    h = np.bincount(b, weights=hit, minlength=TAIL_BINS)
    sp = np.bincount(b, weights=p, minlength=TAIL_BINS)
    keep = n > 0
    n, h, sp = n[keep], h[keep], sp[keep]
    pbar = sp / n
    freq = (h + TAIL_PRIOR * pbar) / (n + TAIL_PRIOR)
    iso = _pav(freq, n)
    g = np.maximum(pbar, iso)
    return np.log10(pbar), g, (n, pbar, h / n, g)


def _tail_apply(M, cols, fit, heads, item_of):
    xs, gs = fit[0], fit[1]
    M = M.astype(np.float64)
    sub = M[:, cols]
    mask = sub < TAIL_P
    g = np.interp(np.clip(np.log10(np.maximum(sub, 1e-12)), TAIL_LO, None), xs, gs)
    M[:, cols] = np.where(mask, np.maximum(sub, g), sub)
    tot = np.add.reduceat(M, heads.offsets[:-1], axis=1)
    return (M / tot[:, item_of]).astype(np.float32)


def _tail_gain(Mold, Mnew, Yc, rows, heads, item_of, lo):
    yt = Yc[rows][:, item_of]
    ok = yt >= 0
    hit = (yt == lo[None, :]) & ok
    tot = 0.0
    for M in (Mold, Mnew):
        S = EPS + (1 - heads.widths[item_of][None, :] * EPS) * (M[rows] / np.add.reduceat(M[rows], heads.offsets[:-1], axis=1)[:, item_of])
        tot += (1 if M is Mnew else -1) * np.log(S[hit]).sum()
    return tot, int(ok.any(1).sum())


def _tailcal(P, B, Yc, rec, pred, heads, half):
    item_of = np.repeat(np.arange(heads.J), heads.widths)
    lo = np.arange(heads.total) - heads.offsets[:-1][item_of]
    gated_j = np.array([bool(rec[n].get("gate")) for n in pred])
    is_g = gated_j[item_of] & (lo == heads.widths[item_of] - 1)
    B = np.asarray(B, np.float32)
    Bn = B / np.add.reduceat(B, heads.offsets[:-1], axis=1)[:, item_of]
    yt = Yc[:, item_of]
    hit_all = (yt == lo[None, :])
    valid = yt >= 0
    before = _total_ll(B, Yc, heads)
    Pn, Bo = P, B
    for tname, cols in (("gate-slot", np.where(is_g)[0]), ("answer-slot", np.where(~is_g)[0])):
        if len(cols) == 0:
            continue

        def pairs(rows):
            sub = Bn[np.ix_(rows, cols)]; v = valid[np.ix_(rows, cols)] & (sub < TAIL_P)
            return sub[v].astype(np.float64), hit_all[np.ix_(rows, cols)][v].astype(np.float64)

        full = np.ones(len(Yc), bool)
        if VERBOSE:
            _log(f"tailcal {tname}: {len(cols)} columns, {len(pairs(np.where(full)[0])[0])} tail pairs")
        gains = []
        for a in (0, 1):
            tr, te = np.where(half == a)[0], np.where(half != a)[0]
            f = _tail_fit(*pairs(tr))
            if f is None:
                gains.append((0.0, 1)); continue
            Mnew = _tail_apply(Bn[te], cols, f, heads, item_of)
            gains.append(_tail_gain(Bn[te], Mnew, Yc[te], np.arange(len(te)), heads, item_of, lo))
        on = all(g[0] / max(g[1], 1) > 0 for g in gains) and sum(g[0] for g in gains) > TAIL_MIN_GAIN
        f = _tail_fit(*pairs(np.where(full)[0]))
        if VERBOSE:
            _log(f"tailcal {tname}: nested gain half0->1 {gains[0][0]:+.2f} nats ({gains[0][1]} rows), half1->0 {gains[1][0]:+.2f} ({gains[1][1]} rows); "
                 f"apply={bool(on and f is not None)}")
            if f is not None:
                n_, pb_, fr_, g_ = f[2]
                for i in range(len(n_)):
                    _log(f"    bin {i:2d} n {int(n_[i]):8d} mean p {pb_[i]:.2e} observed {fr_[i]:.2e} g {g_[i]:.2e}")
        if on and f is not None:
            Bo = _tail_apply(Bo, cols, f, heads, item_of)
            Pn = _tail_apply(Pn, cols, f, heads, item_of)
    if VERBOSE:
        _log(f"tailcal OOF skill {before:.4f} -> {_total_ll(Bo, Yc, heads):.4f}")
    return Pn, Bo


# ----------------------------------------------------------------------------- tempered mixture toward the conditional marginal

TEMPER_LAM = 1.0        # L2 penalty on per-item offsets
TEMPER_MIN_GAIN = 2.0   # nats, total over both nested halves
TEMPER_SAT = 0.995      # vectors whose largest slot is at least this are treated as determined and left alone
TEMPER_MIN_CELL = 20    # group-item counts below this fall back to the pooled marginal
_temper_last = {}


def _temper_sections(pred):
    tok = []
    for n in pred:
        m = re.match(r"^([A-Za-z]+\d*[a-z]?)(?=[_.])", n)
        tok.append(m.group(1) if m else re.match(r"^[A-Za-z]*", n).group(0))
    cnt = pd.Series(tok).value_counts()
    tok = [t if cnt[t] >= 3 else "_other" for t in tok]
    return pd.factorize(pd.Series(tok))[0]


def _temper_fit(py, my, K, valid, sec, S, lam=TEMPER_LAM):
    """eps_j = sigmoid(mu_section + delta_j) fitted on out-of-fold cells with the grader's flat mixing inside the loss."""
    J = py.shape[1]
    onehot = np.zeros((J, S)); onehot[np.arange(J), sec] = 1.0
    c = 1.0 - K * EPS
    d = np.where(valid, my - py, 0.0)

    def f(th):
        mu, de = th[:S], th[S:]
        e = 1.0 / (1.0 + np.exp(-(onehot @ mu + de)))
        q = EPS + c * (py + e[None, :] * d)
        q = np.where(valid, q, 1.0)
        L = -np.log(q).sum() + lam * (de ** 2).sum()
        ge = -((c * d) / q).sum(0) * e * (1 - e)
        return L, np.concatenate([onehot.T @ ge, ge + 2 * lam * de])

    th0 = np.concatenate([np.full(S, -4.0), np.zeros(J)])
    r = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=[(-14, 3)] * S + [(-8, 8)] * J, options={"maxiter": 150})
    mu, de = r.x[:S], r.x[S:]
    return 1.0 / (1.0 + np.exp(-(onehot @ mu + de)))


def _temper_gain(e, py, my, K, valid):
    c = 1.0 - K * EPS
    q1 = EPS + c * ((1 - e[None, :]) * py + e[None, :] * my)
    q0 = EPS + c * py
    return float(np.where(valid, np.log(q1) - np.log(q0), 0.0).sum())


def _temper(P, B, Yc, frame, vis, held_pos, covered, fold, given, heads, Y, Yoh, pred):
    """Pull every item toward the conditional (country, wave) marginal with a data-fitted per-item weight. The marginal of each
    out-of-fold row uses only rows outside its fold. On only when a two-way nested check gains in both halves."""
    J = heads.J
    item_of = np.repeat(np.arange(J), heads.widths)
    K = heads.widths.astype(np.float64)
    gall, _, _, _ = _ghost_codes(frame, given)
    gv, gh = gall[vis], gall[held_pos]
    G = int(gall.max()) + 1
    onehotg = np.zeros((len(vis), G)); onehotg[np.arange(len(vis)), gv] = 1.0
    off = heads.offsets[:-1]

    def marginal(cnt):                       # (G, total) counts -> KT-smoothed marginals, pooled where the cell is thin
        n_it = np.add.reduceat(cnt, off, axis=1)
        pool = cnt.sum(0, keepdims=True)
        n_pool = np.add.reduceat(pool, off, axis=1)
        kt = (cnt + 0.5) / (n_it[:, item_of] + 0.5 * K[item_of][None, :])
        kt_pool = (pool + 0.5) / (n_pool[:, item_of] + 0.5 * K[item_of][None, :])
        return np.where(n_it[:, item_of] >= TEMPER_MIN_CELL, kt, kt_pool)

    Yf = Yoh.astype(np.float64)
    cnt_all = onehotg.T @ Yf
    cov = np.flatnonzero(covered)
    fc = fold[cov]
    Mo = np.zeros((len(cov), heads.total))
    for f in np.unique(fc):
        r = np.flatnonzero(fc == f)
        cnt_f = onehotg[cov[r]].T @ Yf[cov[r]]
        Mo[r] = marginal(cnt_all - cnt_f)[gv[cov[r]]]
    Mt = marginal(cnt_all)[gh]

    B = np.asarray(B, np.float64)
    Bn = B / np.add.reduceat(B, off, axis=1)[:, item_of]
    sat = np.maximum.reduceat(Bn, off, axis=1) >= TEMPER_SAT
    valid = (Yc >= 0) & ~sat
    ycol = off[None, :] + np.maximum(Yc, 0)
    py = np.take_along_axis(Bn, ycol, 1)
    my = np.take_along_axis(Mo, ycol, 1)
    sec = _temper_sections(pred)
    S = int(sec.max()) + 1
    par = fc % 2
    gains = []
    for a in (0, 1):
        tr, te = par == a, par != a
        e = _temper_fit(py[tr], my[tr], K, valid[tr], sec, S)
        gains.append(_temper_gain(e, py[te], my[te], K, valid[te]))
    on = bool(all(g > 0 for g in gains) and sum(gains) > TEMPER_MIN_GAIN)
    e = _temper_fit(py, my, K, valid, sec, S) if on else None
    _temper_last.clear(); _temper_last.update(on=on, gains=gains, sections=S, eps=None if e is None else e.copy())
    if VERBOSE:
        _log(f"temper: {S} sections, nested gains {np.round(gains, 2)} nats, {'ON' if on else 'OFF'}"
             + (f", eps min/median/max {e.min():.4f}/{np.median(e):.4f}/{e.max():.4f}, OOF gain {_temper_gain(e, py, my, K, valid):+.1f} nats" if on else ""))
    if not on:
        return P, B
    Pn = np.asarray(P, np.float64)
    Pn = Pn / np.add.reduceat(Pn, off, axis=1)[:, item_of]
    sat_t = np.maximum.reduceat(Pn, off, axis=1) >= TEMPER_SAT
    ee = np.where(sat_t, 0.0, e[None, :])[:, item_of]
    Q = ((1 - ee) * Pn + ee * Mt).astype(np.float32)
    # the same tempering on the out-of-fold vectors, so a later stage (unmix) is fitted on what it will be applied to
    sat_i = np.repeat(sat, K.astype(int), axis=1)          # B, Bn and Mo are row-aligned (covered rows only)
    eo = np.where(sat_i, 0.0, e[item_of][None, :])
    Bt = (1 - eo) * Bn + eo * Mo
    return (Q, Bt.astype(np.float32)) if (np.isfinite(Q).all() and np.isfinite(Bt).all()) else (P, B)


def _total_ll(Pflat, Y, heads):
    """Skill on these rows, as the grader computes it (1 + mean log score / uniform reference)."""
    tot, cnt = 0.0, 0
    for j in range(heads.J):
        s = slice(heads.offsets[j], heads.offsets[j + 1])
        ok = Y[:, j] >= 0
        if not ok.any():
            continue
        K = heads.widths[j]
        p = _scored(Pflat[ok, s].astype(np.float64) + 1e-12, K)
        tot += np.log(p[np.arange(ok.sum()), Y[ok, j]]).sum()
        cnt += ok.sum()
    U = np.mean([np.log(w) for w in heads.widths])
    return 1 + (tot / max(cnt, 1)) / U


if STRICT and not NO_TABPFN and not _tabpfn_ready():
    raise RuntimeError("TabPFN is required (SB_STRICT=1) but unavailable: vendored package, dependencies or checkpoint")


import math
import torch.nn.functional as F
NO_ORDTYPED = os.environ.get("SB_NO_ORDTYPED") == "1"           # v68: on by default; 1 = v67 exactly
ORDTYPED_FAIL = os.environ.get("SB_ORDTYPED_FAIL") == "1"       # test only: raise inside the candidate
ORDTYPED_MAX_ITEMS = 150          # non-UNHCR-sized schemas only
ORDTYPED_CAP_S = float(os.environ.get("SB_ORDTYPED_CAP", 100.0))            # hard cap on the stage
ORDTYPED_FRAC = float(os.environ.get("SB_ORDTYPED_FRAC", 0.12))              # or this share of the time left before the foundation-model reserve, whichever is smaller
_ORD_NUM_RE = r"[-+]?\d+(?:[.,]\d+)?"
_ORD_SPECIAL_RE = re.compile(r"not answered|don'?t know|refus|prefer not|not sure|unknown|no answer|n/?a\b|not applicable|other|none of|missing", re.I)


def _ordx_level(v):
    s = str(v).strip()
    m = re.match(r"^(\d+)\.\s+[A-Za-z]", s)
    if m:
        if _ORD_SPECIAL_RE.search(s) and not re.match(r"^\d+\.\s+(Never|Rarely|Sometimes|Often|Always|None)", s, re.I):
            return None
        n = float(m.group(1))
        return (n, n) if n < 90 else None
    if _ORD_SPECIAL_RE.search(s):
        return None
    t = s.replace(",", "")
    f = float
    m = re.fullmatch(rf"({_ORD_NUM_RE})", t)
    if m:
        return (f(m.group(1)),) * 2
    m = re.fullmatch(rf"({_ORD_NUM_RE})\s*(?:-|to)\s*({_ORD_NUM_RE})", t)
    if m:
        return (f(m.group(1)), f(m.group(2)))
    m = re.fullmatch(rf"(?:<=|<|under|less than|below)\s*({_ORD_NUM_RE})", t, re.I)
    if m:
        return (f(m.group(1)) - 1e-6,) * 2
    m = re.fullmatch(rf"({_ORD_NUM_RE})\s*(?:\+|or more|and over|and above)", t, re.I)
    if m:
        return (f(m.group(1)), f(m.group(1)) + 1e-6)
    m = re.fullmatch(rf"(?:>=|>|over|more than|above|at least)\s*({_ORD_NUM_RE})", t, re.I)
    if m:
        return (f(m.group(1)) + 1e-6,) * 2
    return None


def _ordx_block(values, min_levels=4):
    """(start, end) of the contiguous, strictly monotone ordered block of an item's levels, or None (v32 detector)."""
    par = [(i, _ordx_level(v)) for i, v in enumerate(values)]
    ok = [(i, p) for i, p in par if p is not None]
    if len(ok) < min_levels:
        return None
    mids = [(p[0] + p[1]) / 2 for _, p in ok]
    if not all(a < b for a, b in zip(mids, mids[1:])):
        return None
    pos = [i for i, _ in ok]
    if pos != list(range(pos[0], pos[-1] + 1)):
        return None
    if str(values[pos[0]]).strip().lower().startswith(("1. yes", "0. no")):
        return None
    return pos[0], pos[-1] + 1


def _ordx_items(heads, rec, pred):
    out = []
    for j, name in enumerate(pred):
        blk = _ordx_block(rec[name]["values"])
        if blk is None:
            continue
        K = int(heads.widths[j])
        a, b = blk
        out.append((j, a, b, [c for c in range(K) if not (a <= c < b)], K))
    return out


def _isp(x):                                    # inverse softplus
    x = np.maximum(x, 1e-4)
    return np.where(x > 20, x, np.log(np.expm1(x)))


class _OrdNet(nn.Module):
    """mode 'loc': typed location-scale ordinal head. mode 'cat': categorical head (D2 penalty applied in the trainer)."""

    def __init__(self, d, items, hidden, drop, mode, marg_flat=None, offs=None):
        super().__init__()
        self.mode, self.items = mode, items
        E = len(items)
        self.E = E
        self.Kmax = max(it[4] for it in items)
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        K = np.array([it[4] for it in items])
        self.register_buffer("Kvec", torch.tensor(K, dtype=torch.float32))
        kmask = np.zeros((E, self.Kmax), bool)
        for e, it in enumerate(items):
            kmask[e, :it[4]] = True
        self.register_buffer("kmask", torch.tensor(kmask))
        if mode == "cat":
            self.out = nn.Linear(hidden, int(K.sum()))
            self.offs = np.concatenate([[0], np.cumsum(K)])
            g = np.zeros((E, self.Kmax), np.int64)
            for e in range(E):
                g[e, :K[e]] = self.offs[e] + np.arange(K[e])
            self.register_buffer("gather", torch.tensor(g))
            return
        B = np.array([it[2] - it[1] for it in items])
        S = np.array([len(it[3]) for it in items])
        self.Bmax, self.Smax = int(B.max()), int(max(S.max(), 1))
        self.mu = nn.Linear(hidden, E)
        self.sc = nn.Linear(hidden, E)
        self.ty = nn.Linear(hidden, E * (1 + self.Smax))
        self.raw_steps = nn.Parameter(torch.zeros(E, self.Bmax - 2))
        cutmask = np.zeros((E, self.Bmax - 1), bool)
        stepmask = np.zeros((E, self.Bmax - 2), bool)
        tmask = np.zeros((E, 1 + self.Smax), bool)
        G = np.zeros((E, self.Kmax), np.int64)
        for e, (j, a, b, spec, Kj) in enumerate(items):
            cutmask[e, :B[e] - 1] = True
            stepmask[e, :B[e] - 2] = True
            tmask[e, :1 + len(spec)] = True
            for c in range(Kj):
                G[e, c] = (c - a) if a <= c < b else self.Bmax + spec.index(c)
        self.register_buffer("cutmask", torch.tensor(cutmask))
        self.register_buffer("stepmask", torch.tensor(stepmask))
        self.register_buffer("tmask", torch.tensor(tmask))
        self.register_buffer("G", torch.tensor(G))
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.sc.weight)
        self.sc.bias.data.fill_(0.5413)                       # softplus -> 1
        if marg_flat is not None:                              # initialise thresholds / location / type bias from the marginal
            tb = np.zeros((E, 1 + self.Smax), np.float32)
            rs = np.zeros((E, self.Bmax - 2), np.float32)
            mub = np.zeros(E, np.float32)
            for e, (j, a, b, spec, Kj) in enumerate(items):
                p = np.maximum(marg_flat[offs[j]:offs[j + 1]].astype(np.float64), 1e-4)
                ps = p[a:b] / p[a:b].sum()
                F = np.clip(np.cumsum(ps)[:-1], 1e-3, 1 - 1e-3)
                lg = np.log(F / (1 - F))
                lg = np.maximum.accumulate(lg + 1e-3 * np.arange(len(lg)))
                st = np.maximum(np.diff(lg), 0.1)
                rs[e, :len(st)] = _isp(st)
                mub[e] = -lg[0]
                tb[e, 0] = np.log(p[a:b].sum())
                for i, c in enumerate(spec):
                    tb[e, 1 + i] = np.log(p[c])
                tb[e, ~self.tmask[e].cpu().numpy()] = 0
            self.raw_steps.data = torch.tensor(rs)
            self.mu.bias.data = torch.tensor(mub)
            self.ty.bias.data = torch.tensor(tb.reshape(-1))
        nn.init.zeros_(self.ty.weight)

    def probs(self, h):                                       # (B, E, Kmax) probabilities (loc mode)
        Bn, E = h.shape[0], self.E
        mu = self.mu(h)
        scale = F.softplus(self.sc(h)) + 1e-3
        steps = (F.softplus(self.raw_steps) + 1e-3) * self.stepmask
        cuts = torch.cat([torch.zeros(E, 1, device=h.device), steps.cumsum(1)], 1)          # (E, Bmax-1), first cut fixed at 0
        cdf = torch.sigmoid((cuts[None] - mu[:, :, None]) / scale[:, :, None])
        cdf = torch.where(self.cutmask[None], cdf, torch.ones_like(cdf))
        bnd = torch.cat([torch.zeros_like(cdf[:, :, :1]), cdf, torch.ones_like(cdf[:, :, :1])], 2)
        ordered = (bnd[:, :, 1:] - bnd[:, :, :-1]).clamp_min(0.0)                           # (B, E, Bmax)
        ordered = ordered / ordered.sum(2, keepdim=True).clamp_min(1e-9)
        tl = self.ty(h).reshape(Bn, E, 1 + self.Smax).masked_fill(~self.tmask[None], -1e9)
        ty = torch.softmax(tl, 2)
        V = torch.cat([ty[:, :, :1] * ordered, ty[:, :, 1:]], 2)                           # (B, E, Bmax+Smax)
        q = torch.gather(V, 2, self.G[None].expand(Bn, -1, -1))
        return q * self.kmask[None]

    def forward(self, X):
        """-> (log mixed prob (B,E,Kmax), unmixed prob (B,E,Kmax))"""
        h = self.trunk(X)
        if self.mode == "cat":
            z = self.out(h)[:, self.gather].masked_fill(~self.kmask[None], -1e9)
            lp = torch.log_softmax(z, 2)
            q = lp.exp()
        else:
            q = self.probs(h)
            lp = None
        K = self.Kvec[None, :, None]
        if lp is None:
            lmix = torch.log(EPS + (1 - K * EPS) * q.clamp_min(0.0) + 1e-12)
        else:
            lmix = torch.logaddexp(torch.full_like(lp, math.log(EPS)), torch.log(1 - K * EPS) + lp)
        return lmix, q

    def d2_penalty(self):
        W = self.out.weight                                    # (total, H)
        tot = 0.0
        for e, (j, a, b, spec, Kj) in enumerate(self.items):
            blk = W[self.offs[e] + a:self.offs[e] + b]
            bb = self.out.bias[self.offs[e] + a:self.offs[e] + b]
            d2 = blk[2:] - 2 * blk[1:-1] + blk[:-2]
            tot = tot + (d2 ** 2).sum() / W.shape[1] + ((bb[2:] - 2 * bb[1:-1] + bb[:-2]) ** 2).sum()
        return tot / max(self.E, 1)


def _ordnet_fit(Xtr, Ytr, Xva, Yva, items, mode, lam, hidden, drop, wd, lr, max_epochs, deadline, seed, marg_flat, offs):
    """Ytr: (n, E) class codes of the eligible items (-1 missing)."""
    torch.manual_seed(seed)
    net = _OrdNet(Xtr.shape[1], items, hidden, drop, mode, marg_flat, offs).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt, Yt = torch.tensor(Xtr, device=DEVICE), torch.tensor(Ytr, device=DEVICE)
    Xv_, Yv_ = torch.tensor(Xva, device=DEVICE), torch.tensor(Yva, device=DEVICE)
    n, bs = len(Xtr), 256
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def nll(lmix, Y):
        valid = Y >= 0
        v = -lmix.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (v * valid).sum() / valid.sum().clamp(min=1)

    best, state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = nll(net(Xt[b])[0], Yt[b])
            if mode == "cat" and lam > 0:
                loss = loss + lam * net.d2_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nll(net(Xv_)[0], Yv_).item()
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
        if time.time() > deadline:
            break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net


def _ordnet_predict(net, X, items, heads):
    out = np.zeros((len(X), heads.total), np.float32)
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEVICE)
        for i in range(0, len(Xt), 4096):
            q = net(Xt[i:i + 4096])[1].float().cpu().numpy()
            for e, (j, a, b, spec, K) in enumerate(items):
                out[i:i + len(q), heads.offsets[j]:heads.offsets[j + 1]] = q[:, e, :K]
    return out


def _ordtyped_candidate(Xv, Y, Xh, fold, heads, rec, pred, stage_deadline, mode="loc", lam=0.0, seeds=2,
                        hidden=256, drop=0.25, wd=1e-4, lr=2e-3, max_epochs=60, marg_given=None):
    """Five-fold direct candidate. Returns (oof, test, have, rows) in the partial-candidate format, or None."""
    if ORDTYPED_FAIL:
        raise RuntimeError("SB_ORDTYPED_FAIL")
    items = _ordx_items(heads, rec, pred)
    if not items:
        return None
    n = len(Xv)
    jj = [it[0] for it in items]
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((len(Xh), heads.total), np.float32)
    nf = int(fold.max()) + 1
    t0 = time.time()
    Yoh_all = _onehot_flat(Y, heads)
    stats = {"fold_s": []}
    seeds_eff = seeds
    for f in range(nf):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        rng = np.random.default_rng(1000 + f)
        perm = rng.permutation(tr)
        k_in = max(int(0.1 * len(tr)), 50)
        inner_va, fit = perm[:k_in], perm[k_in:]
        marg = _marginal_flat(Yoh_all[fit], heads)
        t_f = time.time()
        if f == 1:                                     # fold 0 timed with `seeds` seeds: drop to one seed if the rest cannot fit
            per_seed = stats["fold_s"][0] / seeds
            left_f = stage_deadline - time.time()
            if per_seed * (nf - 1) * seeds > left_f:
                if per_seed * (nf - 1) > left_f:
                    return None
                seeds_eff = 1
        for sd in range(seeds_eff):
            net = _ordnet_fit(Xv[fit], Y[fit][:, jj], Xv[inner_va], Y[inner_va][:, jj], items, mode, lam, hidden, drop, wd, lr,
                              max_epochs, stage_deadline, 17 * (sd + 1) + f, marg, heads.offsets)
            oof[va] += _ordnet_predict(net, Xv[va], items, heads) / seeds_eff
            test += _ordnet_predict(net, Xh, items, heads) / (seeds_eff * nf)
        stats["fold_s"].append(time.time() - t_f)
        if time.time() > stage_deadline:
            return None
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        return None
    have = np.zeros(heads.J, bool)
    have[jj] = True
    _ordtyped_candidate.stats = stats
    return oof, test, have, np.ones(n, bool)


# ----------------------------------------------------------------------------- v69: World Bank structure candidates

NO_BANDT = os.environ.get("SB_NO_BANDT") == "1"            # v69: soft copy-through / band-transition candidate; 1 = v68 exactly
BANDT_FAIL = os.environ.get("SB_BANDT_FAIL") == "1"        # test only: raise inside the candidate
NO_TRACER = os.environ.get("SB_NO_TRACER") == "1"          # v69: shared tracer-response event candidate; 1 = v68 exactly
TRACER_FAIL = os.environ.get("SB_TRACER_FAIL") == "1"      # test only: raise inside the candidate
V69_MAX_ITEMS = 150                                        # non-UNHCR-sized schemas only
BANDT_CAP_S = float(os.environ.get("SB_BANDT_CAP", 30.0))
TRACER_CAP_S = float(os.environ.get("SB_TRACER_CAP", 30.0))
BANDT_MIN_NATS = 1.0               # both parity halves must gain more than this (summed over an item's rows)
TRACER_MIN_NATS = 1.0
TRACER_AGREE = 0.9                 # share of non-response rows in which the group's known items agree
_BT_STOP = {"tr", "user", "after", "before", "full", "next", "of", "the", "and", "per", "last", "in", "to", "sc", "nstd"}


def _bt_special(v):
    return bool(_ORD_SPECIAL_RE.search(str(v))) or "remember" in str(v).lower()


def _bt_tokens(name):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(t) >= 2 and t not in _BT_STOP]


def _bt_nameaff(a, b):
    ta, tb = _bt_tokens(a), _bt_tokens(b)
    hit = 0
    for t in ta:
        if any(t == u or (min(len(t), len(u)) >= 3 and (u.startswith(t) or t.startswith(u))) for u in tb):
            hit += 1
    return hit


def _bt_pairs(rec, given, pred):
    """Generic re-measurement pairs (j, source GIVEN name): the target's non-special labels (almost) all occur among the source's, and the
    names share tokens (X / XAfter, tr_*); a value-compatible source without name evidence is used only if it is the single candidate."""
    out = []
    for j, n in enumerate(pred):
        ns = [v for v in rec[n]["values"] if not _bt_special(v)]
        if len(ns) < 3:
            continue
        cands = []
        for g in given:
            gs = [v for v in rec[g]["values"] if not _bt_special(v)]
            common = len(set(ns) & set(gs))
            if len(gs) < 3 or common < max(3, 0.6 * len(ns)) or common < 0.6 * len(gs):
                continue
            cands.append((_bt_nameaff(n, g), g))
        if not cands:
            continue
        cands.sort(key=lambda t: -t[0])
        if len(cands) > 1 and (cands[0][0] == 0 or cands[0][0] == cands[1][0]):
            continue
        out.append((j, cands[0][1]))
    return out


def _bt_kernel(C, pos_t, typ, g, s):
    nS, nT = C.shape
    n_b = C.sum(1)
    tot = max(C.sum(), 1.0)
    bg = (C.sum(0) + 1.0) / (tot + nT)
    if typ == "offset":
        q = np.zeros(2 * nS + 2 * int(max(pos_t.max(), 0)) + 3)
        o = len(q) // 2
        for a in range(nT):
            if pos_t[a] >= 0:
                for b in range(nS):
                    q[pos_t[a] - b + o] += C[b, a]
        pr = np.zeros((nS, nT))
        for a in range(nT):
            if pos_t[a] >= 0:
                pr[:, a] = q[pos_t[a] - np.arange(nS) + o] + 0.5
    else:
        q0 = sum(C[pos_t[a], a] for a in range(nT) if 0 <= pos_t[a] < nS) / tot
        pr = np.zeros((nS, nT))
        for a in range(nT):
            if 0 <= pos_t[a] < nS:
                pr[pos_t[a], a] += q0
        pr = pr + (1 - q0) * bg[None, :]
    rs = pr.sum(1, keepdims=True)
    pr = np.where(rs > 0, pr / np.maximum(rs, 1e-12), bg[None, :])
    prior = g * pr + (1 - g) * bg[None, :]
    T = (C + s * prior) / (n_b[:, None] + s)
    return T / T.sum(1, keepdims=True)


def _bt_apply(base, bsrc, T, lam, ns_cols):
    """base (n, K) probabilities; bsrc (n,) source band index or -1. Non-special block of the rows with a source band becomes
    lam * T(.|b) + (1 - lam) * base-shape, rescaled to the base non-special mass; specials / gated slot untouched."""
    out = base.astype(np.float64).copy()
    r = np.flatnonzero(bsrc >= 0)
    if not len(r):
        return out
    blk = np.clip(out[np.ix_(r, ns_cols)], 0, None)
    m = blk.sum(1, keepdims=True)
    shape = blk / np.maximum(m, 1e-12)
    out[np.ix_(r, ns_cols)] = m * (lam * T[bsrc[r]] + (1 - lam) * shape)
    return out


def _bt_ll(P, y, K):
    return np.log(EPS + (1 - K * EPS) * P[np.arange(len(y)), y] / P.sum(1))


def _bt_tune(bs_all, ya_all, y_all, base_all, tr_rows, ns_cols, pos_t, types, K, nS, seed=0):
    """Pick (type, g, s, lam) by 2-fold log score inside the fit rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr_rows)
    halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
    nT = len(ns_cols)
    best, best_v = None, -np.inf
    Cs = []
    for h in (0, 1):
        fit = halves[h]
        r = fit[(bs_all[fit] >= 0) & (ya_all[fit] >= 0)]
        C = np.zeros((nS, nT))
        np.add.at(C, (bs_all[r], ya_all[r]), 1.0)
        Cs.append(C)
    for typ in types:
        for g in (0.5, 0.9):
            for s in (2.0, 10.0, 50.0):
                Ts = [_bt_kernel(Cs[h], pos_t, typ, g, s) for h in (0, 1)]
                for lam in (0.5, 1.0):
                    v = 0.0
                    for h in (0, 1):
                        ev = halves[1 - h]
                        ev = ev[(bs_all[ev] >= 0) & (y_all[ev] >= 0)]
                        v += _bt_ll(_bt_apply(base_all[ev], bs_all[ev], Ts[h], lam, ns_cols), y_all[ev], K).sum()
                    if v > best_v:
                        best, best_v = (typ, g, s, lam), v
    return best


def _bandt_candidate(frame, given, rec, pred, opts, idx, heads, vis, held_pos, Y, fold, oof_mlp, test_mlp, stage_deadline):
    """Soft copy-through of a GIVEN re-measured quantity into its PREDICT counterpart. Returns (oof, test, have, rows) or None."""
    if BANDT_FAIL:
        raise RuntimeError("SB_BANDT_FAIL")
    pairs = _bt_pairs(rec, given, pred)
    if not pairs:
        return None
    n, nh = len(vis), len(held_pos)
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((nh, heads.total), np.float32)
    have = np.zeros(heads.J, bool)
    info = []
    for j, g in pairs:
        if time.time() > stage_deadline:
            return None
        s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
        K = s1 - s0
        vals = list(rec[pred[j]]["values"])
        ns_cols = np.array([c for c, v in enumerate(vals) if not _bt_special(v)], int)
        sl = [v for v in rec[g]["values"] if not _bt_special(v)]
        pos_t = np.array([sl.index(vals[c]) if vals[c] in sl else -1 for c in ns_cols], int)
        code2b = np.full(len(opts[g]), -1, int)
        for b, v in enumerate(sl):
            code2b[idx[g][v]] = b
        cs = frame[g].map(idx[g]).fillna(-1).astype(int).to_numpy()
        bs_v, bs_h = np.where(cs[vis] >= 0, code2b[np.maximum(cs[vis], 0)], -1), np.where(cs[held_pos] >= 0, code2b[np.maximum(cs[held_pos], 0)], -1)
        if (bs_v >= 0).sum() < 200:
            continue
        col2a = np.full(K, -1, int)
        col2a[ns_cols] = np.arange(len(ns_cols))
        yj = Y[:, j]
        ya = np.where(yj >= 0, col2a[np.clip(yj, 0, K - 1)], -1)
        base_v = np.clip(oof_mlp[:, s0:s1].astype(np.float64), 0, None)
        base_v /= np.maximum(base_v.sum(1, keepdims=True), 1e-12)
        base_h = np.clip(test_mlp[:, s0:s1].astype(np.float64), 0, None)
        base_h /= np.maximum(base_h.sum(1, keepdims=True), 1e-12)
        ordered = _ordx_block(sl) is not None
        types = ("offset", "diag") if ordered else ("diag",)
        nS = len(sl)
        cand = np.zeros((n, K))
        for f in range(N_FOLDS):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            typ, gg, s, lam = _bt_tune(bs_v, ya, yj, base_v, tr, ns_cols, pos_t, types, K, len(sl), seed=f)
            r = tr[(bs_v[tr] >= 0) & (ya[tr] >= 0)]
            C = np.zeros((nS, len(ns_cols)))
            np.add.at(C, (bs_v[r], ya[r]), 1.0)
            cand[va] = _bt_apply(base_v[va], bs_v[va], _bt_kernel(C, pos_t, typ, gg, s), lam, ns_cols)
        allr = np.arange(n)
        typ, gg, s, lam = _bt_tune(bs_v, ya, yj, base_v, allr, ns_cols, pos_t, types, K, len(sl), seed=99)
        r = allr[(bs_v >= 0) & (ya >= 0)]
        C = np.zeros((nS, len(ns_cols)))
        np.add.at(C, (bs_v[r], ya[r]), 1.0)
        cand_h = _bt_apply(base_h, bs_h, _bt_kernel(C, pos_t, typ, gg, s), lam, ns_cols)
        ev = np.flatnonzero((bs_v >= 0) & (yj >= 0))
        dl = _bt_ll(cand[ev], yj[ev], K) - _bt_ll(base_v[ev], yj[ev], K)
        gains = [float(dl[fold[ev] % 2 == h].sum()) for h in (0, 1)]
        info.append((pred[j], round(min(gains), 1), typ, gg, s, lam))
        if min(gains) > BANDT_MIN_NATS and np.isfinite(cand).all() and np.isfinite(cand_h).all():
            oof[:, s0:s1] = cand
            test[:, s0:s1] = cand_h
            have[j] = True
    _bandt_candidate.stats = info
    _log("bandt pairs (item, min half gain nats, type, g, s, lam): " + str(info))
    if not have.any():
        return None
    return oof, test, have, np.ones(n, bool)


def _tr_na_col(vals):
    for c, v in enumerate(vals):
        if re.fullmatch(r"\s*(not answered|no answer)\s*", str(v), re.I):
            return c
    return -1


def _tracer_groups(rec, pred, opts, heads, Y):
    """Groups of gated sibling items (same parent and observed set) whose 'Not answered' answers co-occur: among eligible rows where
    >= 2 items are known and one is Not answered, the known items agree in >= TRACER_AGREE of rows. Items that fail the test are dropped.
    Returns [(items, na columns, gated col)]."""
    groups = {}
    for j, n in enumerate(pred):
        gt = rec[n].get("gate")
        if not gt:
            continue
        nc = _tr_na_col(opts[n])
        if nc < 0:
            continue
        groups.setdefault((gt["parent"], tuple(gt["observed_if"])), []).append(j)
    out = []
    for kids in groups.values():
        if len(kids) < 3:
            continue
        for _ in range(3):
            A = np.full((len(Y), len(kids)), -1, np.int8)
            for c, j in enumerate(kids):
                K = int(heads.widths[j])
                kn = (Y[:, j] >= 0) & (Y[:, j] != K - 1)
                A[kn, c] = (Y[kn, j] == _tr_na_col(opts[pred[j]])).astype(np.int8)
            nk = (A >= 0).sum(1)
            Am = np.where(A >= 0, A, 0)
            keep = []
            for c in range(len(kids)):
                rows = np.flatnonzero((A[:, c] == 1) & (nk >= 2))
                if len(rows) < 30:
                    continue
                oth = (Am[rows].sum(1) - 1) / np.maximum(nk[rows] - 1, 1)
                if (oth >= 0.5).mean() >= TRACER_AGREE:
                    keep.append(kids[c])
            if len(keep) == len(kids):
                break
            kids = keep
            if len(kids) < 3:
                break
        if len(kids) >= 3:
            out.append(kids)
    return out


def _tracer_event(Y, kids, pred, opts, heads):
    A = np.full((len(Y), len(kids)), -1, np.int8)
    for c, j in enumerate(kids):
        K = int(heads.widths[j])
        kn = (Y[:, j] >= 0) & (Y[:, j] != K - 1)
        A[kn, c] = (Y[kn, j] == _tr_na_col(opts[pred[j]])).astype(np.int8)
    nk = (A >= 0).sum(1)
    E = (np.where(A >= 0, A, 0).sum(1) >= 0.5 * np.maximum(nk, 1)).astype(int)
    return A, nk, E


def _tr_fit_r(Xtr, ytr, Xq, seed):
    """P(respond | X) by L2 logistic regression; C by 2-fold log loss on the fit rows."""
    from sklearn.linear_model import LogisticRegression
    if ytr.min() == ytr.max():
        return np.full(len(Xq), float(ytr.mean()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    h = (perm[: len(perm) // 2], perm[len(perm) // 2:])
    bestC, bv = 0.1, np.inf
    for C in (0.01, 0.1, 1.0):
        v = 0.0
        for a in (0, 1):
            fit, ev = h[a], h[1 - a]
            if ytr[fit].min() == ytr[fit].max():
                continue
            m = LogisticRegression(C=C, max_iter=200).fit(Xtr[fit], ytr[fit])
            p = np.clip(m.predict_proba(Xtr[ev])[:, 1], 1e-4, 1 - 1e-4)
            v += -(ytr[ev] * np.log(p) + (1 - ytr[ev]) * np.log(1 - p)).sum()
        if v < bv:
            bestC, bv = C, v
    m = LogisticRegression(C=bestC, max_iter=300).fit(Xtr, ytr)
    return np.clip(m.predict_proba(Xq)[:, 1], 1e-4, 1 - 1e-4)


def _tr_vec(base, r, u, w, na_col, gcol):
    """Rows of one item: P(NA) = m * ((1 - r) w + r u) with m the non-gated mass (w = P(NA | event), shrunk to 1 so the model is 1 - r + r u by default); the other answers keep the base shape."""
    out = np.clip(base.astype(np.float64), 0, None)
    K = out.shape[1]
    m = 1.0 - (out[:, gcol] if gcol >= 0 else 0.0)
    rest_c = [c for c in range(K) if c != na_col and c != gcol]
    rest = out[:, rest_c]
    rs = rest.sum(1, keepdims=True)
    pna = np.clip(m * ((1 - r) * w + r * u), 1e-5, np.maximum(m - 1e-5, 1e-5))
    out[:, rest_c] = rest / np.maximum(rs, 1e-12) * np.maximum(m - pna, 0)[:, None]
    out[:, na_col] = pna
    return out


def _tracer_candidate(Xv, Xh, opts, rec, pred, heads, Y, fold, oof_mlp, test_mlp, stage_deadline):
    """Shared tracer-response event: P(Not answered) = 1 - r(X) + r(X) u_j for the co-occurring gated siblings."""
    if TRACER_FAIL:
        raise RuntimeError("SB_TRACER_FAIL")
    groups = _tracer_groups(rec, pred, opts, heads, Y)
    if not groups:
        return None
    n = len(Y)
    oof = np.zeros((n, heads.total), np.float32)
    test = np.zeros((len(Xh), heads.total), np.float32)
    have = np.zeros(heads.J, bool)
    info = []
    for kids in groups:
        A, nk, E = _tracer_event(Y, kids, pred, opts, heads)
        elig = nk >= 1
        resp = (1 - E).astype(int)
        r_oof = np.zeros(n)
        for f in range(N_FOLDS):
            if time.time() > stage_deadline:
                return None
            tr = np.flatnonzero((fold != f) & elig)
            va = np.flatnonzero(fold == f)
            r_oof[va] = _tr_fit_r(Xv[tr], resp[tr], Xv[va], f)
        r_h = _tr_fit_r(Xv[elig], resp[elig], Xh, 99)
        cand = {}
        cand_h = {}
        dl = np.zeros(n)
        ok_any = np.zeros(n, bool)
        for c, j in enumerate(kids):
            s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
            K = s1 - s0
            na = _tr_na_col(opts[pred[j]])
            gcol = K - 1 if rec[pred[j]].get("gate") else -1
            u_oof, w_oof = np.zeros(n), np.zeros(n)
            for f in range(N_FOLDS):
                tr = (fold != f) & (E == 0) & (A[:, c] >= 0)
                u_oof[fold == f] = (A[tr, c].sum() + 0.5) / (tr.sum() + 1.0)
                tr = (fold != f) & (E == 1) & (A[:, c] >= 0)
                w_oof[fold == f] = (A[tr, c].sum() + 2.0) / (tr.sum() + 2.0)
            tr = (E == 0) & (A[:, c] >= 0)
            u_all = (A[tr, c].sum() + 0.5) / (tr.sum() + 1.0)
            tr = (E == 1) & (A[:, c] >= 0)
            w_all = (A[tr, c].sum() + 2.0) / (tr.sum() + 2.0)
            cand[j] = _tr_vec(oof_mlp[:, s0:s1], r_oof, u_oof, w_oof, na, gcol)
            cand_h[j] = _tr_vec(test_mlp[:, s0:s1], r_h, u_all, w_all, na, gcol)
            ev = (A[:, c] >= 0)
            base_n = np.clip(oof_mlp[:, s0:s1].astype(np.float64), 0, None)
            base_n /= np.maximum(base_n.sum(1, keepdims=True), 1e-12)
            d = np.zeros(n)
            d[ev] = _bt_ll(cand[j][ev], Y[ev, j], K) - _bt_ll(base_n[ev], Y[ev, j], K)
            dl += d
        gains = [float(dl[fold % 2 == h].sum()) for h in (0, 1)]
        info.append((len(kids), round(min(gains), 1)))
        if min(gains) > TRACER_MIN_NATS:
            for j in kids:
                s0, s1 = int(heads.offsets[j]), int(heads.offsets[j + 1])
                oof[:, s0:s1] = cand[j]
                test[:, s0:s1] = cand_h[j]
                have[j] = True
    _tracer_candidate.stats = info
    _log("tracer groups (n items, min half gain nats): " + str(info))
    if not have.any() or not (np.isfinite(oof).all() and np.isfinite(test).all()):
        return None
    return oof, test, have, np.ones(n, bool)
