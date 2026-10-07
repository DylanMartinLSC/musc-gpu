# musc-gpu

An **unofficial GPU-optimized implementation of MuSc**, with **3.74×–4.35×
measured inference speedup** across complete MVTec AD bottle, grid and cable
pools on an NVIDIA RTX 3090. The original backend is included and selectable.
This release contains the latest **qualified and deployed** acceleration code,
its reproducible benchmark, and measured accuracy and resource comparisons.

MuSc is *Zero-Shot Industrial Anomaly Classification and Segmentation with
Mutual Scoring of the Unlabeled Images* (ICLR 2024), by **Xurui Li, Ziming
Huang, Feng Xue, and Yu Zhou**. See the [original repository](https://github.com/xrli-U/MuSc),
[paper](https://arxiv.org/abs/2401.16753), and [attribution notices](THIRD_PARTY_NOTICES.md).
This release is independently maintained and is not the official MuSc repo.

## Performance and capabilities

Each category uses its **entire test pool**, batch4, one reference division,
518px CLIP ViT-L-14-336/OpenAI, feature indices 5/11/17/23 and radii 1/3/5.
Two alternating pairs start each backend/pass in a fresh process, with
first-use compilation included and no discarded image warmup. OS filesystem
caches are not flushed. Model loading and quality evaluation are measured
separately, outside inference.

| Complete category | Images | Original median | musc-gpu median | Speedup |
|---|---:|---:|---:|---:|
| bottle | 83 | 35.01 s | 9.36 s | **3.74×** |
| grid | 78 | 31.71 s | 8.05 s | **3.94×** |
| cable | 150 | 102.34 s | 23.55 s | **4.35×** |

The ratio of summed category medians is **4.13×** across these three
validated categories. This is not a combined reference pool or a claim for
all MVTec AD categories, backbones, or GPUs. Pool inference throughput is
not independent single-image latency. No verified 10× result exists.

![Complete pool inference latency](benchmarks/representative/latency.png)
![Paired speedup](benchmarks/representative/speedup.png)

Image AUROC/AP/F1 agree with the original results in these runs. Pixel
AUROC/AP/F1/AUPRO remain close; exact deltas are retained in raw JSON. All
full-resolution maps and image scores pass the **2e-5** agreement gate.
Maximum map errors are 4.315e-6 (bottle), 1.636e-6 (grid), and 1.738e-6
(cable). The measurements establish agreement on these workloads, rather
than bitwise identity with the original or accuracy guarantees elsewhere.

![Classification and segmentation quality](benchmarks/representative/quality.png)
![Numerical agreement](benchmarks/representative/agreement.png)

Acceleration costs memory: peak allocated GPU memory is **7.14 GiB** for
bottle, **6.80 GiB** for grid, and **11.64 GiB** for cable; original peaks
are 6.29, 6.26 and 6.64 GiB respectively. These are allocated tensor peaks,
not total device usage. The benchmark separately records reserved memory
and assesses capacity before larger pools.

![Measured GPU memory](benchmarks/representative/memory.png)

See the [complete benchmark report](benchmarks/representative/REPORT.md),
[authoritative raw JSON](benchmarks/representative/results.json),
[CSV](benchmarks/representative/summary.csv), and
[measured package versions](benchmarks/representative/environment_packages.json).
Image/pixel AP and F1 are included alongside AUROC; optional AUPRO was enabled
for this run. These are newly measured metrics, not capture placeholders.

An earlier [bottle83 qualification](benchmarks/bottle83/results.json) measured
**3.836766×**, 37.13 s → 9.68 s, under the historical shared-process protocol.
Its [report](benchmarks/bottle83/qualification_report.md) and
[deployment receipt](benchmarks/bottle83/deployment_checks.json) preserve the
accepted implementation lineage. The fresh-process benchmark above is a
distinct protocol; cross-run differences do not establish a new optimization.
Historical workspace paths are provenance, not required release paths.

## Changes from the original implementation

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

## Installation and use

Run commands from the release root. Python 3.10 is a conservative environment
choice; the supplied measurements establish the PyTorch/CUDA combination
above, not a compatibility matrix for every Python release.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python examples/musc_main.py --data_path C:/path/to/mvtec_anomaly_detection --class_name bottle --inference_backend gpu
```

Obtain MVTec AD separately, with its existing category/test/ground_truth
layout. The first model load downloads OpenAI weights if they are absent.
Change the dataset path in the command or `configs/musc.yaml`. Outputs go
to `output/`; optional upstream metrics and heatmaps run after inference.

### Original backend and fallback

```powershell
python examples/musc_main.py --data_path C:/path/to/mvtec_anomaly_detection --class_name bottle --inference_backend original
```

The original path uses the original encoder, LNAMD aggregation, MSM scoring,
and sequential loader. GPU kernels are imported only for the accelerated
path. When CUDA is unavailable or the OS is not native Windows, a GPU request
selects the original backend and prints the reason. CPU execution is an
upstream fallback, not a newly performance-qualified configuration.

The custom runtime currently loads NVRTC DLLs shipped with Windows PyTorch
and uses the caller's CUDA stream. Linux and WSL acceleration are not
implemented in this release. Missing NVRTC or a CUDA runtime failure on
Windows remains an explicit error: rerun with `--inference_backend original`.
The attention wrapper uses native PyTorch outside its supported dispatch.
Use the original backend for unvalidated model/configuration comparisons.

## Evidence and release verification

The release includes a reproducible [benchmark suite](docs/BENCHMARK.md) for
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

```powershell
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories bottle grid cable --pairs 2 --include-pro --output benchmarks/runs/representative
python tools/report_benchmark.py benchmarks/runs/representative/results.json --output benchmarks/runs/representative/figures
```

```powershell
python tools/verify_release.py
python tools/test_benchmark.py
python -m pip install -r requirements-dev.txt
python tools/plot_benchmarks.py
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
benchmark is a new run of this packaged inference implementation.

The release contains runtime models, upstream datasets/utilities, the latest
qualified GPU modules, raw accepted evidence, and graph/verification tools.
Old candidates, experiment supervisors, logs, caches, seeds, checkpoints,
datasets and Git history are excluded. The development workspace remains
available with its existing sources and evidence. Research can continue there
independently of this release.

## Citation and license

Please cite the original MuSc paper when using its method:

```bibtex
@inproceedings{li2024musc,
  title={MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images},
  author={Li, Xurui and Huang, Ziming and Xue, Feng and Zhou, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

Preserved MuSc [MIT license](LICENSE); bundled third-party sources retain
their applicable licenses and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
