# musc-gpu benchmark

Run status: **pass**.

[Authoritative raw JSON](results.json)

MuSc.make_category_data entry through synchronized pre-metric predictions; includes loading, decoding, preprocessing, transfers, encoding, aggregation, scoring, interpolation, RsCIN and first-use compilation; excludes imports, model loading, metric evaluation and artifact serialization.

Fresh processes include first-use compilation on every trial. No discarded image warmup. Operating-system file cache is not flushed. Metrics and model loading are separate.

| Pool | Images | Scope | Status | Original (s) | musc-gpu (s) | Speedup | Worst pair |
|---|---:|---|---|---:|---:|---:|---:|
| bottle-full | 83 | complete category | pass | 35.014 | 9.356 | 3.742× | 3.742× |
| grid-full | 78 | complete category | pass | 31.707 | 8.052 | 3.938× | 3.912× |
| cable-full | 150 | complete category | pass | 102.345 | 23.551 | 4.346× | 4.313× |

Aggregate speedup for validated full categories (bottle, grid, cable): **4.128×**.

Ratio of summed per-category medians; excludes skipped/failed categories; not one combined reference pool.

## Quality and resource measurements

Metric values below are medians across the measured trials (0–1 scale). Undefined/not requested metrics remain blank. F1 is maximized over evaluation thresholds; this does not establish a production threshold.

| Pool / backend | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Pixel AUPRO | Peak allocated GiB | Model load (s) | Metrics (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle-full / original | 0.9992 | 0.9998 | 0.9848 | 0.8304 | 0.9610 | 6.29 | 4.64 | 43.98 |
| bottle-full / gpu | 0.9992 | 0.9998 | 0.9848 | 0.8304 | 0.9610 | 7.14 | 4.39 | 43.21 |
| grid-full / original | 0.9875 | 0.9957 | 0.9816 | 0.3824 | 0.9391 | 6.26 | 4.40 | 39.67 |
| grid-full / gpu | 0.9875 | 0.9957 | 0.9816 | 0.3824 | 0.9391 | 6.80 | 4.42 | 37.60 |
| cable-full / original | 0.9908 | 0.9948 | 0.9576 | 0.5770 | 0.8962 | 6.64 | 4.41 | 75.02 |
| cable-full / gpu | 0.9908 | 0.9948 | 0.9576 | 0.5770 | 0.8962 | 11.64 | 4.43 | 76.62 |

## Numerical agreement

| Pool | Max map error | Max score error | Tolerance |
|---|---:|---:|---:|
| bottle-full | 4.314954e-06 | 2.086163e-07 | 2e-5 |
| grid-full | 1.636432e-06 | 3.799796e-07 | 2e-5 |
| cable-full | 1.738490e-06 | 2.980232e-07 | 2e-5 |

No speed claim is made for failed, incomplete or skipped pools. A subset changes the reference pool and cannot establish full-category quality or the bottle83 objective.

Full per-pass timings, dispatch diagnostics, quality values, paths, hashes and errors are in the input JSON. Raw prediction arrays and logs remain in the original benchmark run directory.

![Pool latency](latency.png)

![Paired speedup](speedup.png)

![Peak allocated memory](memory.png)

![Detection and segmentation quality](quality.png)

![Numerical agreement](agreement.png)
