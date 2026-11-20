# Supplementary material (added after submission)

Nothing in this folder is part of the submitted archive. The submitted code is the tag `v70-submission`, also
`main.py` and `vendor/` at the repository root. This folder supports the development history and local evaluation
described in the technical report (`paper.pdf`, Appendices A, C, D and G).

## `ledger.csv`: development history
One row per version or group of versions (v0 to v70): the change, then the development-leaderboard readout and our
verdict. It is the table in Appendix G of the report in machine-readable form.

- Readouts are the rounded values the leaderboard displayed. Up to v16 they are on the uniform-reference scale
  (U/W/H = UNICEF / World Bank / UNHCR). From v17 they are on the crowd-marginal scale. "Same" means UNICEF 0.34993 and
  UNHCR 0.86501. W-H / W-L means World Bank read 0.24856 / 0.23353.
- Development scores were noised and rounded. "null" means no readout change. A single World Bank change cannot
  distinguish a model effect from reporting noise. Rows are observational, not controlled ablations.

## `benches/`: local benchmarks and runner
None of these touch competition microdata, which were never available. Each generator writes
`<SB_SANDBOX_ROOT>/<name>/{schema.json,respondents.parquet}`. The default `SB_SANDBOX_ROOT` is `supplementary/_sandbox`.

| Script | What it builds |
|---|---|
| `harness.py` | Loads a submission's `main.py` the way the grader does and scores it like the official scorer. Phase 1 (DEV held out) or `--phase 2` (TEST held out). |
| `make_proxy.py` | Real-signal proxy from US Census ACS 2022 1-year PUMS, Rhode Island persons (public domain). Set `SB_ACS_CSV` to `psam_p44.csv` from https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_pri.zip. |
| `make_unhcr_like.py` | Synthetic UNHCR-shaped bench on the public UNHCR schema, with planted latent types, routing and noisy gates. Set `SB_DATA_DIR` to the starter kit's `data/`. |
| `make_messy.py` | Proxy plus documented routing irregularities: GIVEN/PREDICT gate violations and a wrong declared parent. |
| `make_violation.py` | UNHCR-shaped bench with gate violations, rare violations under support caps and breakoff rows. |
| `make_ordinal.py` (+ `ord_detect.py`) | Planted latent ordinal structure on a World Bank sandbox. |
| `make_inter_bench.py`, `make_wave_bench.py` | Planted GIVEN x GIVEN and wave x country interactions, each with a zero-interaction control. |

Example, after generating the official sandboxes with the organisers' starter kit
(https://github.com/SituatedEvals/public, `make_sandbox.py`) into `supplementary/_sandbox/`:
```bash
cd supplementary
SB_DATA_DIR=/path/to/starter-kit/data python benches/make_unhcr_like.py --out unhcr_like
python benches/make_violation.py --seed 0 --out viol_s0 --given --pred --branch --tail --ghost
python -I benches/harness.py .. unicef viol_s0          # score the submission at the repository root
```
The benches are synthetic or come from a different population. They test mechanics and rank ideas. They do not
estimate scores on the hidden competition data.

Licence: MIT (the repository `LICENSE`).
