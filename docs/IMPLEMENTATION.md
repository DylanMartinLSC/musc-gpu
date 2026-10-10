# Implementation and release verification

[Back to the README](../README.md)

## Acceleration mechanisms

- **GPU mutual scoring:** symmetric tiled dot products reuse work in both
  image directions. Centered distance certificates screen candidates, and
  original FP32 features refine distances. Per-reference-image Euclidean
  minima, self exclusion, and trimmed ranks are preserved; scores are FP64.
- **Direct spatial pooling:** replaces expanded Unfold intermediates while
  preserving normalization and padding behavior.
- **Ordered threaded loading:** four workers preserve image order and batch4
  assembly while overlapping decoding and preprocessing.
- **Normalization reuse:** reuses radius-independent spatial normalization
  only within one inference division. Features are recomputed on every call;
  no persistent image-feature cache is used. This mechanism adds about
  1.76 GiB peak allocation in the accepted bottle83 comparison.
- **Attention:** alignment-aware FP32 softmax retains 1024 logical lanes on
  256 physical threads and exact FP16 probabilities. Block12 retains both
  probability stores; the other 23 omit discarded FP32 stores.

The trained weights, resolution, requested layers, radii, reference pool and
scoring contract remain intact. Full-image agreement tolerance is 2e-5;
the protected scoring gate uses 3e-5 absolute / 1e-5 relative tolerance.
Numerical equivalence means agreement within these tolerances, not bitwise
identity with the original pipeline. No labels or masks enter scoring.

## Evidence and release verification

The release includes a reproducible [benchmark suite](../docs/BENCHMARK.md) for
complete MVTec AD categories and explicitly labeled pool-size scaling tests.
It measures original-vs-GPU latency, image/pixel AUROC, AP, F1 and optional
AUPRO, numerical agreement, allocated/reserved memory, model loading and
metric costs. Fresh-process trials include first-use compilation on every
pass; failed agreement rejects speed claims.

### Backbone scope

The benchmark harness currently supports **only CLIP ViT-L-14-336 with
OpenAI pretrained weights at 518px**, batch size 4, feature indices
5/11/17/23 and radii 1/3/5. It sets these values explicitly; changing
`configs/musc.yaml` does not change the benchmark model, and there is no
backbone-selection CLI option. `--categories all` means all 15 MVTec AD
categories with this same backbone.

The upstream inference code retains other backbone configuration paths,
but they are not covered by this harness or the published speedup and
accuracy results. Use `--inference_backend original` for those unvalidated
configurations. Supporting another backbone in the comparison harness
requires adapting its model/layer configuration, output-shape checks,
memory estimates and accelerated-dispatch validation, then qualifying
numerical agreement and measuring it separately.

Run these commands from the repository root after the README setup.

```powershell
.venv/Scripts/python.exe tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories bottle grid cable --pairs 2 --include-pro --output benchmarks/runs/representative
.venv/Scripts/python.exe tools/report_benchmark.py benchmarks/runs/representative/results.json --output benchmarks/runs/representative/figures
```

```powershell
.venv/Scripts/python.exe tools/verify_release.py
.venv/Scripts/python.exe tools/test_benchmark.py
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe tools/plot_benchmarks.py
```

Verification checks source integrity, Python syntax, backend selection and
preservation of the deployed kernel source and aggregation factory. It runs
on CPU without downloading weights or launching CUDA. Graphs regenerate
from the retained raw JSON; SVG versions are included for editing.

`SOURCE_MANIFEST.json` records original and packaged source hashes. Packaging
changes only runtime import locations, platform fallback selection, and
portable configuration defaults in inference sources. The qualified CUDA
source and encoder aggregation hooks are preserved. The bottle83 qualification
is historical evidence; the three-category
benchmark records fresh-process runs of this packaged inference implementation.

The package contains runtime models, upstream dataset loaders/utilities,
qualified GPU modules, accepted benchmark evidence and verification tools.
Datasets and model checkpoints must be obtained separately.

## Historical qualification

An earlier [bottle83 qualification](../benchmarks/bottle83/results.json) measured
**3.836766×**, 37.13 s → 9.68 s, under the historical shared-process protocol.
Its [report](../benchmarks/bottle83/qualification_report.md) and
[deployment receipt](../benchmarks/bottle83/deployment_checks.json) preserve the
accepted implementation lineage. The [three-category benchmark](../benchmarks/representative/REPORT.md) uses a
distinct fresh-process protocol; cross-run differences do not establish a new optimization.
Historical workspace paths are provenance, not required release paths.

