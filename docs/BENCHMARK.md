# musc-gpu benchmark protocol

`tools/benchmark.py` evaluates both the original and accelerated implementations
through the actual MuSc caller. The default suite covers **bottle, grid and cable**:
an object, a texture, and a larger object pool. `--categories all` plans all 15
MVTec AD categories. Category claims are earned only by successful full-pool runs.

```powershell
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --plan-only
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories bottle grid cable --pairs 2 --include-pro --output benchmarks/runs/representative
python tools/report_benchmark.py benchmarks/runs/representative/results.json --output benchmarks/runs/representative/figures
```

Run from the musc-gpu root with the environment described in the main README.
Output directories must be new. Trials are bounded to 900 seconds each by default
(`--timeout-seconds` changes the bound). Complete original-vs-GPU suites can take
several minutes. The parent imports no torch before acquiring the GPU lock.

## Measurements

- **Inference latency:** complete pipeline entry through synchronized predictions
  immediately before metrics, including loading, decoding, preprocessing,
  transfers, encoder, normalization, aggregation, mutual scoring, interpolation,
  RsCIN and first-use compilation. Upstream ground-truth loading and array
  bookkeeping are included; labels and masks are never inputs to anomaly scoring.
- **Cold process:** each backend/pass starts in a fresh process, loads the same
  weights and computes all image features again. Both directions of paired order
  are measured, with no discarded image warmup. OS disk caches are not flushed,
  so this is not a claim of cold-storage latency.
- **Separate costs:** model loading, metric evaluation and artifact serialization
  are outside inference. Model loading, CUDA initialization and metrics receive
  their own timings; total worker wall time is also recorded.
  First model download, when necessary, is part of model loading.
- **Quality:** image/pixel AUROC, average precision and threshold-maximized F1.
  `--include-pro` adds the slower upstream AUPRO at FPR<0.3. Metrics are evaluated
  after inference for both backends; undefined values are null with a reason.
  Threshold-maximized F1 is an evaluation statistic, not a deployment threshold.
- **Numerics:** every full-resolution map and every image score are compared
  using the 2e-5 absolute agreement gate. Reports also contain relative error
  (denominator floor 1e-12), RMSE, dtype, path order and label equality. Nonfinite
  predictions, mismatched shapes or failed agreement reject the speed claim.
- **Memory:** peak allocated and reserved GPU memory, plus baseline allocation.
  A conservative capacity estimate precedes each pool. Insufficient capacity
  skips the complete pool and records the reason; it never caps survivors or
  silently shrinks the reference pool. The estimate is not a measured peak.
- **Provenance:** runtime/tool hashes, Python/PyTorch/CUDA/GPU/driver, TF32 settings,
  effective model configuration, source checks per trial, ordered relative
  image paths and byte sizes, ground-truth array hashes, actual attention dispatch, and raw logs. Image
  manifest hashes identify paths/sizes, not file contents; datasets stay read-only.

The fixed model contract is batch4, one reference pool, 518px CLIP
ViT-L-14-336/OpenAI, feature indices 5/11/17/23 and radii 1/3/5. FP32 scoring
inputs, FP64 scoring outputs, complete reference minima, self exclusion and
trimmed ranks remain unchanged. No approximations or feature cache across calls.

## Scaling and broader evaluation

```powershell
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories bottle --pool-sizes 8 32 full --pairs 2 --output benchmarks/runs/bottle_scaling
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories all --plan-only
python tools/benchmark.py --data-path C:/path/to/mvtec_anomaly_detection --categories all --include-pro --output benchmarks/runs/all_categories
```

Explicit smaller pools use deterministic path-hash selection independent of
labels and masks. Each subset is its own complete mutual-scoring pool. Subset
results are labeled and excluded from the full-category aggregate: changing
reference images changes the task, so subset accuracy/speed cannot substitute
for complete-category validation. Invalid or duplicated pool sizes are rejected.

At least two paired passes are required. More pairs can improve repeatability
assessment when justified, but two pairs do not establish statistical confidence.
The report lists per-pass ratios, range and ratio of medians. Aggregate speed is
the ratio of summed category medians for successful **full** categories, with
their names listed; skipped categories cannot support a whole-dataset claim.
Pool throughput is not independent-image latency.

## Coordination and failure handling

Native Windows CUDA is required for the accelerated suite. A backend fallback
cannot be reported as GPU acceleration. The coordinator owns
`Local\MuScGpuBenchmark`, refuses an active research supervisor, checks local
model-unload APIs and baseline GPU use before every worker, and verifies source
hashes. It does not stop/unload other processes or alter OS settings. Run only
after coordinating an exclusive GPU slot with other research.

Worker timeout, out-of-memory, metrics errors and numerical failures retain
logs, job parameters, trial JSON and partial results. A failed numerical trial
ends that workload. Successfully measured categories remain visible in a
partial report; failed/incomplete categories have no accepted speedup. Overall
exit status is nonzero for partial/failed runs. `report_benchmark.py` can still
produce an honest report from them.

Before torch import or CUDA initialization, the actual worker interpreter
waits for the coordinator to assign it to a Windows **Job Object** with
kill-on-close enabled. This handles Python virtual-environment launchers
that spawn another interpreter. Timeout or cancellation closes that owned
job and terminates its workers and descendants. It does not kill processes
by name. The go signal is issued only after assignment succeeds.

Large prediction maps are stored as uncompressed `.npy` arrays so comparison
can use memory mapping. Allow several GiB of disk for a multi-category run.
Raw arrays are research artifacts, excluded from the distribution archive;
release figures and summary JSON contain measured results only.

## Included measurement receipts

The [three-category report](../benchmarks/representative/REPORT.md) contains
12 full-pool trials: two paired orders for each of bottle, grid and cable.
Those measurements preceded the added Job Object guard and extra metadata
checks. The inference implementation is unchanged; benchmark source hashes
retain the exact measured runner revision in the development evidence.

The final runner passed a separate [8-image integration test](../benchmarks/harness_validation/small_pool_results.json)
in both orders, including process ownership, dispatch, full-resolution
predictions, ground-truth hashes and numerical gates. It measured **1.61×**
on this explicit small pool; that result is excluded from the full-category
aggregate and headline claims. Small pools can have lower speedup because
setup and encoding form more of their latency.

[CPU process-tree tests](../benchmarks/harness_validation/process_tree_tests.json)
verify that normal completion and timeout leave neither the actual worker
nor its descendant alive. Run `python tools/test_worker_process.py` on
Windows to repeat these tests without importing torch or running CUDA.
