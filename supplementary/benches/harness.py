"""Local harness: load main.py by path (submission dir NOT on sys.path), run a phase frame, score like the grader.

usage: harness.py <submission_dir> [--phase 1|2] <sandbox names...>
phase 1 (default): rows TRAIN+DEV in file order, DEV held out and scored.
phase 2: visible = TRAIN + DEV (complete), then TEST rows blanked on PREDICT items; TEST is scored (mirrors the organisers' score.py sample_rows).
"""
import os, sys, json, importlib.util, time, resource
import numpy as np, pandas as pd
args = sys.argv[1:]
phase = 1
if '--phase' in args:
    k = args.index('--phase'); phase = int(args[k + 1]); del args[k:k + 2]
assert phase in (1, 2)
tail = '--tail' in args
if tail: args.remove('--tail')
sub = args[0]; names = args[1:]
root = os.environ.get('SB_SANDBOX_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_sandbox'))
spec = importlib.util.spec_from_file_location('main', sub + '/main.py'); m = importlib.util.module_from_spec(spec)
sys.modules['main'] = m; spec.loader.exec_module(m)
t_all = time.time(); res = {}
for name in names:
    d = pd.read_parquet(f'{root}/{name}/respondents.parquet')
    schema = json.load(open(f'{root}/{name}/schema.json')); schema.setdefault('gated_value', 'NA_GATED')
    rec = schema['items']
    items = [n for n, r in rec.items() if r['class'] in ('GIVEN', 'PREDICT')]
    pred = [n for n, r in rec.items() if r['class'] == 'PREDICT']
    if phase == 1:
        d = d[d.role.isin(['TRAIN', 'DEV'])].reset_index(drop=True)
        truth = d.copy()
        frame = d[['respondent_id'] + items].copy()
        dev = (d.role == 'DEV').to_numpy()
        frame.loc[dev, pred] = np.nan
    else:
        vis = d[d.role.isin(['TRAIN', 'DEV'])]
        hid = d[d.role == 'TEST']
        assert len(vis) and len(hid)
        vis = vis[['respondent_id'] + items].reset_index(drop=True)
        hid = hid[['respondent_id'] + items].reset_index(drop=True)
        blanked = hid.copy(); blanked[pred] = np.nan
        frame = pd.concat([vis, blanked], ignore_index=True)
        truth = pd.concat([vis, hid], ignore_index=True)
    t = time.time(); out = m.predict(frame, schema); dt = time.time() - t
    opts = {n: list(rec[n]['values']) + ([schema['gated_value']] if rec[n].get('gate') else []) for n in items}
    ll, i, bad = [], 0, 0
    tp, tg = [], []
    for r in range(len(frame)):
        for n in items:
            if pd.isna(frame.at[r, n]):
                q = np.asarray(out[i], float); i += 1; K = len(opts[n])
                if q.shape != (K,) or not np.isfinite(q).all() or (q < 0).any() or q.sum() <= 0: bad += 1; q = np.ones(K)
                p = 1e-3 + (1 - K * 1e-3) * q / q.sum()
                ti = opts[n].index(truth.at[r, n])
                ll.append(np.log(p[ti]))
                if tail: tp.append(p[ti]); tg.append(truth.at[r, n] == schema['gated_value'])
    assert i == len(out), (i, len(out))
    U = np.mean([np.log(len(opts[n])) for n in pred])
    res[name] = 1 + np.mean(ll) / U
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)  # macOS: bytes
    print(f'{name}: phase {phase} skill {res[name]:.4f}  predict {dt:.1f}s  cells {len(out)}  invalid {bad}  maxrss_so_far {rss:.2f}GB', flush=True)
    if tail:
        tp = np.array(tp); tg = np.array(tg); L = -np.log(tp); N = len(tp)
        print(f'  tail[{name}] cells {N}  loss/cell {L.sum() / N:.4f}  (band: share of cells | loss nats/cell)')
        for lab, lo, hi in (('p<1e-3', 0, 1e-3), ('1e-3<=p<1e-2', 1e-3, 1e-2), ('1e-2<=p<0.05', 1e-2, 0.05), ('p>=0.05', 0.05, 2)):
            mk = (tp >= lo) & (tp < hi)
            print(f'    {lab:14s} share {mk.mean():.5f}  loss {L[mk].sum() / N:.5f}')
        for lab, mk in (('true=gated', tg), ('true=answer', ~tg)):
            mb = mk & (tp < 1e-2)
            print(f'    {lab:14s} share {mk.mean():.5f}  loss {L[mk].sum() / N:.5f}  of which p<1e-2: share {mb.mean():.5f} loss {L[mb].sum() / N:.5f}')
print('total wall', round(time.time() - t_all, 1), 's  mean', round(np.mean(list(res.values())), 4))
