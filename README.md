# SimulacraBench 2026: final submission

Source code of our final-phase submission to the NeurIPS 2026 SimulacraBench competition (probabilistic survey
completion). The files are exactly those of the submitted archive, except for the TabPFN-3.5 weight file, which is not
redistributed (see below).

## Contents
- `main.py`: the submission. The platform calls `predict(frame, schema)` once per instrument.
- `requirements.txt`: nothing to install beyond the hosted image (numpy, pandas, scipy, scikit-learn, torch).
- `vendor/`, `vendor_pd/`: unmodified third-party inference code bundled with the submission (tabpfn 9.2.0, tabicl 2.2.0,
  einops, sdm, pydantic stack). Licences are in `vendor/README.txt` and `vendor/LICENSES/`.
- `weights/`: licence and notice for TabPFN-3.5.

## Rebuild the submitted archive
1. Download `tabpfn-v3.5-fast-20260909.safetensors` from https://huggingface.co/Prior-Labs/tabpfn_3_5 (accepting the
   TabPFN-3.5 licence) and place it in `weights/`.
2. From this directory: `zip -qr submission.zip main.py requirements.txt vendor vendor_pd weights`
   (`main.py` must sit at the archive root).

Without the weight file the TabPFN candidate is skipped and the rest of the pipeline runs unchanged.
The method is described in the accompanying technical report.

## Licence
MIT for `main.py` (see `LICENSE`). Bundled third-party components keep their own licences. The TabPFN-3.5 weights are
licensed by Prior Labs GmbH under a non-commercial licence.
