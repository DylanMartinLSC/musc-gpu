# Trimmed-rank deferred scoring: measured results

[Back to the README](../../README.md)

This update replaces `models/modules/_MSM_gpu.py` with trimmed-rank deferred
scoring (see [implementation](../../docs/IMPLEMENTATION.md)). Every other module is
unchanged. RTX 3090, PyTorch 2.7.1 + CUDA 11.8, CLIP ViT-L-14-336/OpenAI at 518 px.

## Original versus default GPU (bottle, grid)

`tools/benchmark.py` with two alternating fresh-process pairs per category, complete
pools, first-use compilation included, model loading and metrics separate
([raw JSON](results.json)).

| Category | Images | Original median | Default GPU median | Speedup | Paired speedups | Max map / score error |
|---|---:|---:|---:|---:|---|---|
| bottle | 83 | 37.48 s | 8.88 s | **4.22×** | 4.07× / 4.38× | 4.315e-6 / 2.086e-7 |
| grid | 78 | 31.84 s | 7.45 s | **4.27×** | 4.29× / 4.26× | 1.636e-6 / 3.800e-7 |

Peak allocated tensor memory: bottle 7.49 GiB (original 6.29 GiB), grid 7.11 GiB
(original 6.26 GiB). The previous release measured 3.74× (bottle) and 3.94× (grid).

## Cable: previous versus new default GPU scoring

The cable original-backend worker exceeded the capped experiment job used for these
runs (16 GiB process-tree commit limit on the measurement machine) and crashed before
producing predictions; the receipt above records that failure. Cable was therefore
measured GPU-only: one process, model loaded once, two alternating pairs through
`MuSc.make_category_data` with only the scoring module swapped
([raw JSON](cable_gpu_check.json)).

| Scoring | Median | Peak allocated |
|---|---:|---:|
| previous release | 25.98 s | 11.64 GiB |
| trimmed-rank deferred | 22.38 s | 12.92 GiB |

New scoring is **1.16× faster** (pairs 1.165× / 1.157×) and produces **bitwise
identical maps, scores and labels**, so the previous release's cable agreement with
original (max map error 1.738e-6) still applies. This in-process timing is not
comparable to the fresh-process original/GPU figures; the published cable 4.35×
original-relative speedup was measured with the previous scoring.

## Super fast with the current scoring

`tools/benchmark_super_fast.py`, two alternating fresh-process original/super fast
pairs per complete pool ([raw JSON](super_fast_results.json)). All seven accuracy
metric medians, for both original and super fast, are identical to the previous
release's [category results](../super_fast/category_results.json); repeated
predictions agree.

| Category | Images | Original median | Super fast median | Speedup | Previous release | Super fast peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 83 | 36.09 s | 3.99 s | **9.05×** | 8.50× | 3.47 GiB (previously 3.17) |
| grid | 78 | 31.84 s | 3.81 s | **8.37×** | 8.43× | 3.35 GiB (previously 3.07) |
| cable | 150 | 103.09 s | 6.84 s | **15.08×** | 15.00× | 5.27 GiB (previously 4.45) |

Grid's change is within run-to-run variation; super fast spends little time in
scoring at its 28×28 grid. All three categories keep the accuracy outcomes
reported previously (only bottle meets the provisional 1 pp tolerance).

## Memory

Deferred refinement stores per-pair bounds (~16 bytes per query patch × reference
image) and up to four candidate records (12 bytes each) per pair. This is small for
pools near 80 images but grows with images² × patches: about +1.3 GiB peak on the
150-image cable pool. Larger pools need their own capacity assessment.

## Qualification history

Development qualification used the research harness on bottle83 (bitwise against the
previous incumbent on every check, protected scoring gate, representative early/final
layer screens, bottle32, then two alternating full passes). An identical-code A/A
control on that harness measured 1.040× / 1.008×, so its first-pair ratios favor the
candidate; second-pair gains for the two accepted steps were 1.063× and 1.079×.
The fresh-process figures above are the release evidence.
