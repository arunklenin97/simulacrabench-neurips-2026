Bundled third-party code (unmodified; licence texts in vendor/LICENSES/):
- tabicl 2.2.0 (BSD-3-Clause, Soda team @ Inria), inference modules only: https://github.com/soda-inria/tabicl
- einops 0.8.2 (MIT): https://github.com/arogozhnikov/einops
- tabpfn 9.2.0 (Apache-2.0, Prior Labs): https://github.com/PriorLabs/TabPFN
- vendor_pd/ (appended to the end of sys.path, so installed copies take precedence): pydantic 2.14.1 and pydantic_core
  (MIT, Linux cp313 build), pydantic-settings (MIT), python-dotenv (BSD-3-Clause), typing-inspection (MIT),
  annotated-types (MIT)
- skrub is not bundled; main.py installs a stub module because it is only used for datetime/text columns.

Weights, bundled as files (nothing is downloaded; there is no models.txt):
- weights/tabicl-classifier-v2-20260212.ckpt: TabICLv2 classifier (BSD-3-Clause), https://huggingface.co/jingang/TabICL
- weights/tabpfn-v3.5-fast-20260909.safetensors: TabPFN-3.5 fast checkpoint, https://huggingface.co/Prior-Labs/tabpfn_3_5,
  under the TabPFN-3.5 licence (weights/TABPFN-3.5-LICENSE.txt) with the required attribution (weights/TABPFN-3.5-NOTICE.txt).
- sdm (structured-data-models, Apache-2.0, NVIDIA; third-party notices in vendor/LICENSES/sdm-THIRD_PARTY_LICENSES.md.txt, the
  TabICLv2-derived parts keep their BSD-3-Clause licence in sdm/models/*/LICENSE and NOTICE), unmodified: https://github.com/NVIDIA/structured-data-models
- weights/kumo/small/classifier.pt: Kumo-Tabular small classifier, OpenMDW-1.1 (weights/KUMO-TABULAR-OPENMDW-1.1-LICENSE.txt),
  attribution in weights/KUMO-TABULAR-NOTICE.txt.
