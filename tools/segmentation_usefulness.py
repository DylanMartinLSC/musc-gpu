"""CPU-only, fixed-threshold retrospective evaluation of saved MuSc maps."""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy import ndimage


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ratio(a, b):
    return float(a / b) if b else None


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [max(0., center-half), min(1., center+half)]


def choose_threshold(maxima, alpha):
    """Finite order statistic; explicit diagnostic fallback if sample is too small."""
    values = np.sort(np.asarray(maxima, dtype=float))
    if not len(values) or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError('Need finite calibration maxima and 0 < alpha < 1')
    rank = math.ceil((len(values) + 1) * (1-alpha))
    sufficient = rank <= len(values)
    return float(values[min(rank, len(values))-1]), {
        'normal_images': len(values), 'requested_false_reject_rate': alpha,
        'order_statistic_rank': rank, 'rank_available': sufficient,
        'status': 'finite_rank_available' if sufficient else 'insufficient_calibration_diagnostic_max_only',
        'minimum_n_for_finite_rank': math.ceil(1/alpha)-1,
        'guarantee': 'None: predictions share a transductive pool; independent exchangeability is not established.',
    }


def score_image(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    structure = np.ones((3, 3), dtype=np.uint8)
    regions, n = ndimage.label(truth, structure)
    sizes = np.bincount(regions.ravel(), minlength=n+1)[1:]
    hits = np.bincount(regions[pred], minlength=n+1)[1:]
    coverage = hits / sizes if n else np.array([])
    components, count = ndimage.label(pred, structure)
    touching = np.unique(components[truth])
    unmatched = count - int(np.count_nonzero(touching))
    row = dict(tp=tp, fp=fp, fn=fn, pixels=truth.size,
               precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
               iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn),
               predicted_components=count, unmatched_predicted_components=unmatched,
               gt_regions=n, regions_covered_50=int(np.count_nonzero(coverage >= .5)),
               regions_covered_90=int(np.count_nonzero(coverage >= .9)),
               any_alarm=bool(pred.any()), defective=bool(truth.any()))
    return row, [{'area_pixels': int(a), 'area_fraction': float(a/truth.size),
                  'coverage': float(c)} for a, c in zip(sizes, coverage)]


def summarize(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    normal = [r for r in rows if not r['defective']]
    defects = [r for r in rows if r['defective']]
    rejects = sum(r['any_alarm'] for r in normal)
    localized = sum(r['regions_covered_50'] > 0 for r in defects)
    regions = sum(r['gt_regions'] for r in rows)
    return dict(images=len(rows), normal_images=len(normal), defective_images=len(defects),
                tp=tp, fp=fp, fn=fn, pixel_precision=ratio(tp,tp+fp),
                pixel_recall=ratio(tp,tp+fn), pixel_iou=ratio(tp,tp+fp+fn),
                pixel_dice=ratio(2*tp,2*tp+fp+fn),
                defective_image_macro_iou=float(np.mean([r['iou'] for r in defects])) if defects else None,
                normal_false_rejects=rejects, normal_false_reject_rate=ratio(rejects,len(normal)),
                normal_false_reject_wilson95=wilson(rejects,len(normal)),
                localized_defective_images=localized,
                localized_defective_image_rate=ratio(localized,len(defects)),
                localized_image_wilson95=wilson(localized,len(defects)),
                gt_regions=regions, region_coverage50_rate=ratio(sum(r['regions_covered_50'] for r in rows),regions),
                region_coverage90_rate=ratio(sum(r['regions_covered_90'] for r in rows),regions),
                unmatched_components_per_image=ratio(sum(r['unmatched_predicted_components'] for r in rows),len(rows)),
                normal_false_positive_pixel_fraction=ratio(sum(r['fp'] for r in normal),sum(r['pixels'] for r in normal)))


def load_trial(trial_dir, dataset):
    trial = json.loads((trial_dir/'trial.json').read_text())
    job = json.loads((trial_dir/'job.json').read_text())
    if trial['status'] != 'pass' or not job['plan']['complete_pool']:
        raise ValueError('Requires a successful complete-pool trial')
    for key in ('maps', 'labels'):
        if digest(trial_dir/f'{key}.npy') != trial['artifact_sha256'][key]:
            raise ValueError(f'{key} artifact hash mismatch')
    maps = np.load(trial_dir/'maps.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(trial_dir/'labels.npy', allow_pickle=False)
    paths = [row['path'] for row in job['plan']['image_manifest']]
    if len(paths) != job['plan']['images'] or len(paths) != job['plan']['full_category_images']:
        raise ValueError('Complete-pool image count differs from manifest')
    actual = [str(p).replace('\\','/') for p in trial['image_paths']]
    if len(paths) != len(set(paths)) or len(paths) != len(actual):
        raise ValueError('Duplicate paths or manifest length mismatch')
    if any(not a.endswith('/'+p) and a != p for a,p in zip(actual,paths)):
        raise ValueError('Prediction order differs from manifest')
    if maps.shape != (len(paths),1,518,518) or labels.shape != (len(paths),):
        raise ValueError('Unsupported prediction shape')
    masks = np.zeros(maps.shape, dtype=np.int32)
    for i, relative in enumerate(paths):
        parts = Path(relative).parts
        if len(parts) != 4 or parts[1] != 'test' or '..' in parts:
            raise ValueError('Expected category/test/type/image manifest')
        if parts[0] != job['plan']['category']:
            raise ValueError('Mixed category or category/manifest mismatch')
        if not np.isfinite(maps[i]).all():
            raise ValueError('Nonfinite prediction')
        if int(labels[i]) != int(parts[2] != 'good'):
            raise ValueError('Label/path mismatch')
        if parts[2] != 'good':
            mask_path = dataset/parts[0]/'ground_truth'/parts[2]/(Path(parts[3]).stem+'_mask.png')
            with Image.open(mask_path) as im:
                # Match upstream Resize bilinear -> ToTensor -> int32 conversion exactly.
                resized = np.asarray(im.resize((518,518), Image.Resampling.BILINEAR))
            masks[i,0] = (resized.astype(np.float32)/255).astype(np.int32)
    reconstructed_hash = hashlib.sha256(masks.tobytes()).hexdigest()
    recorded_hash = trial.get('ground_truth_sha256', {}).get('masks')
    if recorded_hash is not None and reconstructed_hash != recorded_hash:
        raise ValueError('Reconstructed masks differ from recorded inference ground truth')
    trial['evaluation_masks_sha256'] = reconstructed_hash
    trial['evaluation_masks_verified_against_trial'] = recorded_hash is not None
    if not np.array_equal(masks.any(axis=(1,2,3)).astype(labels.dtype),labels):
        raise ValueError('Empty defect mask or inconsistent labels')
    return trial, maps, masks, paths


def evaluate(trial_dir, dataset, seed, fraction, alpha):
    trial, maps, masks, paths = load_trial(trial_dir, dataset)
    normal = [i for i in range(len(paths)) if not masks[i].any()]
    if len(normal) < 4:
        raise ValueError('Need at least four normal images for calibration/evaluation split')
    normal.sort(key=lambda i: hashlib.sha256(f'{seed}:{paths[i]}'.encode()).hexdigest())
    ncal = max(1,min(len(normal)-1,math.floor(len(normal)*fraction)))
    calibration = set(normal[:ncal])
    threshold, cal = choose_threshold([maps[i].max() for i in sorted(calibration)],alpha)
    rows, regions = [], []
    for i, path in enumerate(paths):
        if i in calibration:
            continue
        row, image_regions = score_image(maps[i,0] > threshold, masks[i,0])
        row.update(path=path, defect_type=Path(path).parts[2])
        rows.append(row)
        for j, region in enumerate(image_regions):
            regions.append(dict(path=path, region_id=j+1, **region))
    return dict(category=Path(paths[0]).parts[0], backend=trial['backend'],
                threshold=threshold, calibration=cal,
                calibration_paths=[paths[i] for i in sorted(calibration)],
                metrics=summarize(rows),
                by_defect_type={k:summarize([r for r in rows if r['defect_type']==k])
                                for k in sorted({r['defect_type'] for r in rows})},
                images=rows, regions=regions,
                provenance=dict(trial=str(trial_dir.resolve()),trial_sha256=digest(trial_dir/'trial.json'),
                                job_sha256=digest(trial_dir/'job.json'),
                                maps_sha256=trial['artifact_sha256']['maps'],
                                masks_sha256=trial['evaluation_masks_sha256'],
                                masks_verified_against_trial=trial['evaluation_masks_verified_against_trial']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', nargs='+', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--calibration-fraction', type=float, default=.5)
    parser.add_argument('--target-false-reject-rate', type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.calibration_fraction < 1 or not 0 < args.target_false_reject_rate < 1:
        parser.error('Fractions must be strictly between zero and one')
    if args.output.exists():
        parser.error('Output must be new; existing evidence is never overwritten')
    results = [evaluate(p,args.dataset,args.seed,args.calibration_fraction,args.target_false_reject_rate)
               for p in args.trials]
    report = dict(schema_version=1, status='retrospective_diagnostic_not_production_qualification',
                  protocol=dict(seed=args.seed,calibration_fraction=args.calibration_fraction,
                                target_false_reject_rate=args.target_false_reject_rate,
                                prediction_rule='raw map > category calibration threshold; no filtering or per-image normalization',
                                connectivity=8, region_coverage_thresholds=[.5,.9],
                                split='normal images split by seeded path hash; all defective images evaluation only',
                                limitations=['Calibration and evaluation share the original transductive pool.',
                                             'Existing previously inspected data: retrospective, not preregistered validation.',
                                             'Wilson intervals are descriptive image-level intervals; shared-pool dependence remains.',
                                             'Region and pixel metrics have no independent-sample confidence claim.',
                                             'Masks use the original 518px benchmark transformation, not physical defect dimensions.',
                                             'No customer acceptance criteria supplied; no industrial pass verdict.']),
                  environment=dict(python=sys.version,numpy=np.__version__,pillow=PIL.__version__,
                                   scipy=scipy.__version__,platform=platform.platform()),
                  source_sha256=digest(__file__),command=sys.argv,results=results)
    args.output.mkdir(parents=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines = ['# Fixed-threshold segmentation usefulness', '',
             'Retrospective shared-pool diagnostic; no production qualification or threshold tuning on evaluation masks.', '',
             '| Category | Backend | Pixel precision | Pixel recall | IoU | Regions >=50% covered | Normal false rejects | Calibration |',
             '|---|---|---:|---:|---:|---:|---|---|']
    def fmt(x):
        return 'undefined' if x is None else f'{x:.3f}'
    for index, result in enumerate(results):
        m = result['metrics']
        lines.append(f"| {result['category']} | {result['backend']} | {fmt(m['pixel_precision'])} | {fmt(m['pixel_recall'])} | {fmt(m['pixel_iou'])} | {fmt(m['region_coverage50_rate'])} | {m['normal_false_rejects']}/{m['normal_images']} | {result['calibration']['status']} (n={result['calibration']['normal_images']}) |")
        for key in ('images','regions'):
            entries = result[key]
            if entries:
                with (args.output/f'{index}-{result["category"]}-{result["backend"]}-{key}.csv').open('w',newline='') as f:
                    writer = csv.DictWriter(f,fieldnames=list(entries[0]))
                    writer.writeheader()
                    writer.writerows(entries)
    lines += ['', 'A missing finite calibration rank uses the maximum normal calibration score only as a diagnostic threshold; it does not meet the requested false-reject target.',
              '', 'See results.json for fixed thresholds, calibration/evaluation paths, counts, descriptive intervals, per-type results, hashes and limitations.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
