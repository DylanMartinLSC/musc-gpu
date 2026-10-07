"""CPU-only benchmark planning, comparisons and reporting."""
from pathlib import Path
import hashlib
import json
import math
import statistics

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
              'transistor', 'wood', 'zipper']
ATOL = 2e-5
SCOPE = ('MuSc.make_category_data entry through synchronized pre-metric predictions; '
         'includes loading, decoding, preprocessing, transfers, encoding, aggregation, '
         'scoring, interpolation, RsCIN and first-use compilation; '
         'excludes imports, model loading, metric evaluation and artifact serialization')

def dump(path, value):
    """Atomic replacement prevents partial JSON after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan_pools(data_path, categories, pool_sizes=('full',), seed=42):
    """Enumerate complete pools; optional subsets never use labels for selection."""
    root = Path(data_path).resolve()
    normalized = ['full' if size == 'full' else str(int(size)) for size in pool_sizes]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Pool sizes must be distinct')
    plans = []
    for category in categories:
        folder = root / category / 'test'
        if not folder.is_dir():
            raise ValueError(f'Missing category test directory: {folder}')
        if any(not p.is_dir() for p in folder.iterdir()):
            raise ValueError(f'Upstream dataset expects only anomaly directories in {folder}')
        entries = []
        # Match upstream category/anomaly/file lexicographic ordering.
        for anomaly in sorted(p for p in folder.iterdir() if p.is_dir()):
            for image in sorted(p for p in anomaly.iterdir() if p.is_file()):
                if image.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                    raise ValueError(f'Non-image file would enter upstream dataset: {image}')
                relative = image.relative_to(root).as_posix()
                entries.append({'path': relative, 'bytes': image.stat().st_size})
        if len(entries) < 5:
            raise ValueError(f'{category}: at least five reference images are required for trimmed ranks')
        for requested in normalized:
            n = len(entries) if requested == 'full' else int(requested)
            if n < 5 or n > len(entries):
                raise ValueError(f'{category}: invalid pool size {n}; full count is {len(entries)}')
            selected = entries
            if requested != 'full':
                # Seeded path hashing is deterministic and independent of labels/masks.
                chosen = sorted(range(len(entries)), key=lambda i: hashlib.sha256(
                    f'{seed}:{entries[i]["path"]}'.encode()).digest())[:n]
                selected = [entries[i] for i in sorted(chosen)]
            signature = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
            plans.append({'id': f'{category}-{requested}', 'category': category,
                          'images': n, 'full_category_images': len(entries),
                          'complete_pool': requested == 'full', 'selection_seed': seed,
                          'image_manifest': selected, 'manifest_sha256': signature,
                          'manifest_definition': 'SHA256 of ordered relative image paths and byte sizes; not image-content hashes',
                          # Conservative capacity assessment, not a measured memory claim.
                          'estimated_gpu_gib': 4.25 + n * .065,
                          'estimated_output_disk_gib_per_trial': n * 518 * 518 * 8 / 1024**3})
    return plans

def compare_arrays(reference, candidate, chunk_images=4):
    import numpy as np
    if not reference.size or not candidate.size:
        return {'pass': False, 'error': 'Empty predictions cannot establish agreement'}
    if reference.shape != candidate.shape:
        return {'pass': False, 'shape_match': False, 'reference_shape': list(reference.shape),
                'candidate_shape': list(candidate.shape)}
    absolute, relative, squares, count = 0., 0., 0., 0
    finite = True
    for start in range(0, len(reference), chunk_images):
        a = np.asarray(reference[start:start+chunk_images], dtype=np.float64)
        b = np.asarray(candidate[start:start+chunk_images], dtype=np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            finite = False
            continue
        delta = np.abs(a-b)
        if delta.size:
            absolute = max(absolute, float(delta.max()))
            relative = max(relative, float((delta / np.maximum(np.abs(a), 1e-12)).max()))
            squares += float(np.square(delta).sum())
            count += delta.size
    return {'pass': finite and absolute <= ATOL, 'shape_match': True, 'finite': finite,
            'max_absolute_error': absolute if finite else None,
            'max_relative_error_reference_floor_1e_12': relative if finite else None,
            'rmse': math.sqrt(squares/count) if finite and count else None,
            'absolute_tolerance': ATOL, 'reference_dtype': str(reference.dtype),
            'candidate_dtype': str(candidate.dtype)}

def summarize(workload):
    pairs = workload.get('pairs', [])
    expected = workload['requested_pairs']
    valid = [p for p in pairs if p.get('agreement', {}).get('pass') and
             all(math.isfinite(p[b]['inference_seconds']) and p[b]['inference_seconds'] > 0
                 for b in ('original', 'gpu'))]
    if len(valid) != expected or workload.get('failures'):
        return {'status': 'incomplete_or_failed', 'valid_pairs': len(valid),
                'requested_pairs': expected, 'speedup': None}
    times = {b: [p[b]['inference_seconds'] for p in valid] for b in ('original', 'gpu')}
    medians = {b: statistics.median(values) for b, values in times.items()}
    ratios = [p['original']['inference_seconds']/p['gpu']['inference_seconds'] for p in valid]
    return {'status': 'pass', 'valid_pairs': len(valid), 'requested_pairs': expected,
            'median_inference_seconds': medians, 'speedup': medians['original']/medians['gpu'],
            'paired_speedups': ratios, 'minimum_paired_speedup': min(ratios),
            'all_pairs_reach_10x': all(r >= 10 for r in ratios),
            'time_range_seconds': {b: [min(t), max(t)] for b, t in times.items()},
            'pool_throughput_images_per_second': {b: workload['images']/t for b,t in medians.items()},
            'note': 'Pool throughput is not independent single-image latency; no confidence interval from two pairs.'}
