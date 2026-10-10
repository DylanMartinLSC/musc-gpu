# Fixed-threshold segmentation usefulness benchmark

This CPU-only evaluator asks whether a predicted mask covers defects precisely,
and how often it alarms on normal parts. It reads saved prediction artifacts from
`tools/benchmark.py`; it never reruns inference or changes the reference pool.

```powershell
python tools/segmentation_usefulness.py --trials benchmarks/runs/representative/bottle-full/pair1-gpu benchmarks/runs/representative/grid-full/pair1-gpu benchmarks/runs/representative/cable-full/pair1-gpu --dataset C:/path/to/mvtec_anomaly_detection --output benchmarks/runs/segmentation_usefulness
python tools/test_segmentation_usefulness.py
```

The published release does not contain the large prediction arrays. Generate
them with the existing benchmark first, or point `--trials` at retained trial
directories containing `job.json`, `trial.json`, `maps.npy`, and `labels.npy`.
Dependencies are NumPy, Pillow, and SciPy (installed with the existing scientific
Python dependencies). No CUDA is used by this evaluator.

## Frozen operating rule

Within each category, normal images are ordered by SHA256 of `seed:path`; half
are reserved for calibration (default seed 42). All remaining normals and all
defective images are evaluated. No defective mask or evaluation score is used to
choose a threshold. The default requested image false-reject rate is 5%.

The threshold is the ceil((n+1)*(1-alpha)) order statistic of calibration normal
image maximum scores. A pixel alarms strictly when its raw score exceeds the
threshold. There is no per-image normalization, morphology, minimum area filter,
or selection of the best threshold on evaluation masks.

If the order-statistic rank exceeds the calibration sample size, a finite
threshold cannot support that target under the usual independent-sample rule.
The tool explicitly marks insufficient calibration and uses the observed maximum
only to produce diagnostic measurements. Even when the rank is available, this
shared-pool experiment does not establish a distribution-free guarantee.

Set `--target-false-reject-rate`, `--calibration-fraction`, and `--seed` before
evaluation. Do not search these settings for favorable results on the same data.

## Measurements

- Pooled pixel precision, recall, IoU and Dice, with raw TP/FP/FN counts.
- Mean IoU on defective images, keeping normal images separate.
- Fraction of 8-connected ground-truth regions with at least 50% or 90% coverage.
  These are region coverage metrics, not one-to-one instance detection AP.
- Defective images with at least one region covered by 50%.
- Normal image false-reject rate: any predicted positive pixel triggers an alarm.
- Unmatched predicted components per image: components with no ground-truth
  overlap. Partially overlapping components can still contain false-positive
  pixels, captured by pixel precision.
- Normal false-positive pixel fraction, per-defect-type metrics, and per-image
  and per-region CSVs (including region size and coverage).
- Descriptive 95% Wilson intervals for image outcomes only. Pixels and regions
  are not treated as independent observations.

Undefined ratios are null, not perfect scores. Large blanket masks are penalized
by precision and IoU even if they cover every defect. Tiny defects remain included.

## Evidence and limits

Prediction and label artifact hashes and manifest ordering are checked. Masks
are reconstructed with the original MVTec benchmark's bilinear resize to 518px,
float conversion and int32 cast. This preserves the prior evaluation convention,
including its boundary erosion; these are not measurements in physical units.
If a trial recorded a ground-truth hash, reconstruction must match it. Older
trials without that hash are explicitly marked unverified against the trial.

`results.json` is authoritative and retains thresholds, splits, paths, hashes,
configuration, and detailed measurements. `REPORT.md` and CSVs aid review.
Existing output directories are rejected to preserve evidence.

Calibration/evaluation images still shared a complete MuSc scoring pool. This is
a retrospective transductive diagnostic, not independent prospective production
validation. Original benchmark images have already been inspected in earlier
research. Pool dependence also limits the interpretation of the Wilson intervals.
There is deliberately no automatic industrial PASS or universal acceptance limit.

For a customer decision, agree tolerable false rejects, critical-defect misses,
overlap requirements, and minimum defect size in advance. Use separate production
lots/pools for calibration and final evaluation, freeze thresholds, and collect
enough normal and defective parts to bound those rates. Requalify if pool policy,
camera setup, preprocessing, model configuration, or postprocessing changes.
