# SimulacraBench 2026: final submission

Source code of our final-phase submission to the NeurIPS 2026 SimulacraBench competition (probabilistic survey
completion), with the technical report.

- **Technical report:** [`paper.pdf`](paper.pdf), "Data-Verified Routing and Anytime Ensembling for Probabilistic Survey
  Completion".
- **Submitted code:** the tag [`v70-submission`](../../tree/v70-submission) holds exactly the files of the submitted
  archive, except for the TabPFN-3.5 weight file, which is not redistributed (see below). `paper.pdf` was added to
  `main` after the submission and is not part of the submitted archive.
- **Supplementary material:** [`supplementary/`](supplementary/) holds a development ledger (v0 to v70) and the local
  benchmark generators and runner. It was added after the submission and is not part of the submitted archive.

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

## Licence
MIT for `main.py` (see `LICENSE`). Bundled third-party components keep their own licences. The TabPFN-3.5 weights are
licensed by Prior Labs GmbH under a non-commercial licence.
