"""Bounded, serialized original-vs-GPU MuSc benchmark. See docs/BENCHMARK.md."""
import argparse
import contextlib
import ctypes
import hashlib
from importlib.metadata import version
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from urllib import request, error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from benchmark_support import CATEGORIES, SCOPE, compare_arrays, digest, dump, plan_pools, summarize
from benchmark_process import run_worker, worker_ready

def inventory():
    # Treat permission errors as errors rather than evidence of an idle machine.
    script = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell|pwsh|ollama|llama|qwen' } | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    output = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', script], text=True)
    value = json.loads(output) if output.strip() else []
    return value if isinstance(value, list) else [value]

def check_idle(device, endpoint):
    processes = inventory()
    supervisors = [p for p in processes if any(s in (p.get('CommandLine') or '').lower()
                   for s in ('run_long_hermes.ps1', 'experiment_controller.py', 'start_overnight.ps1'))]
    if supervisors:
        raise RuntimeError('Research supervisor is active; coordinate before benchmarking. No process was stopped.')
    stop = ROOT.parents[1] / 'experiments/runs/stop_gpu_research'
    if stop.exists():
        raise RuntimeError('Research stop/coordination marker exists; refusing to override it')
    urls = list(dict.fromkeys([endpoint, 'http://127.0.0.1:11434/api/ps']))
    checks = []
    responding = 0
    for url in urls:
        try:
            with request.urlopen(url, timeout=4) as response:
                models = json.load(response).get('models', [])
            responding += 1
            checks.append({'url': url, 'models': models})
            if models:
                raise RuntimeError(f'Local inference models are loaded at {url}; benchmark refused')
        except error.URLError as exc:
            reason = exc.reason
            if not isinstance(reason, OSError) or (getattr(reason, 'winerror', None) not in (10061,)
                    and getattr(reason, 'errno', None) not in (10061, 111)):
                raise
            checks.append({'url': url, 'connection_refused': True})
    if not responding and any(p['Name'].lower().startswith(('ollama', 'llama', 'qwen')) for p in processes):
        raise RuntimeError('Local model process exists but no unload API responds; cannot verify idle state')
    raw = subprocess.check_output(['nvidia-smi', '-i', str(device),
        '--query-gpu=name,uuid,memory.total,memory.used,driver_version', '--format=csv,noheader,nounits'], text=True).strip()
    name, uuid, total, used, driver = [x.strip() for x in raw.split(',')]
    if float(used) > 2048:
        raise RuntimeError(f'Baseline GPU usage is {used} MiB (>2048); benchmark refused')
    return {'processes': processes, 'unload_checks': checks, 'gpu': name, 'uuid': uuid,
            'total_mib': float(total), 'baseline_used_mib': float(used), 'driver': driver}

class GpuMutex:
    def __enter__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.CreateMutexW.restype = wintypes.HANDLE
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateMutexW(None, False, 'Local\\MuScGpuBenchmark')
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        status = self.api.WaitForSingleObject(self.handle, 0)
        if status not in (0, 0x80):
            self.api.CloseHandle(self.handle)
            raise RuntimeError('GPU benchmark mutex is owned by another job')
        return self
    def __exit__(self, *args):
        self.api.ReleaseMutex(self.handle)
        self.api.CloseHandle(self.handle)

def evaluate(labels, scores, masks, maps, include_pro):
    """Metrics outside inference. Undefined metrics are null, never fake zero."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    from utils.metrics import cal_pro_score
    result = {'values': {}, 'errors': {}, 'definition': 'upstream resized masks and predictions; threshold-maximized F1; upstream PRO at FPR<0.3'}
    for prefix, truth, pred in (('image', labels, scores), ('pixel', masks, maps)):
        truth, pred = truth.ravel(), pred.ravel()
        names = [f'{prefix}_{n}' for n in ('auroc', 'ap', 'f1_max')]
        if np.unique(truth).size != 2 or not np.isfinite(pred).all():
            result['values'].update(dict.fromkeys(names))
            result['errors'][prefix] = 'Undefined: requires two ground-truth classes and finite predictions'
            continue
        result['values'][names[0]] = float(roc_auc_score(truth, pred))
        result['values'][names[1]] = float(average_precision_score(truth, pred))
        precision, recall, _ = precision_recall_curve(truth, pred)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(precision), where=(precision+recall)>0)
        result['values'][names[2]] = float(f1.max())
    result['values']['pixel_aupro'] = None
    if include_pro:
        try:
            if not masks.any() or float(maps.max()) == float(maps.min()):
                raise ValueError('PRO requires anomalous regions and nonconstant maps')
            value = float(cal_pro_score(masks.squeeze(1), maps.squeeze(1)))
            if not np.isfinite(value):
                raise ValueError('Nonfinite PRO')
            result['values']['pixel_aupro'] = value
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            result['errors']['pixel_aupro'] = str(exc)
    else:
        result['errors']['pixel_aupro'] = 'Not requested; use --include-pro for the slower upstream metric'
    return result

def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'backend': job['backend'], 'scope': SCOPE}
    dump(out/'trial.json', report)
    try:
        if not job.get('worker_token') or os.environ.get('MUSC_BENCHMARK_WORKER_TOKEN') != job['worker_token']:
            raise RuntimeError('Internal worker must be launched by the mutex-owning benchmark coordinator')
        worker_ready(out, job['worker_token'])
        report['worker_pid'] = os.getpid()
        report['source_sha256'] = {relative: digest(ROOT/relative) for relative in job['source_sha256']}
        if report['source_sha256'] != job['source_sha256']:
            raise RuntimeError('Benchmark source changed after coordinator snapshot')
        import numpy as np
        import torch
        import yaml
        import models.musc as musc
        import models.modules._MSM as original_msm
        from models.modules._backend import select_backend
        cuda_start = time.perf_counter()
        if not torch.cuda.is_available() or select_backend('gpu', 'cuda')[0] != 'gpu':
            raise RuntimeError('Accelerated benchmark requires native Windows CUDA; fallback cannot be timed as GPU')
        torch.cuda.set_device(job['device'])
        torch.manual_seed(job['seed'])
        np.random.seed(job['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize(job['device'])
        report['cuda_setup_seconds'] = time.perf_counter()-cuda_start
        cfg = yaml.safe_load((ROOT/'configs/musc.yaml').read_text())
        cfg['datasets'].update(data_path=job['data_path'], dataset_name='mvtec_ad',
                               class_name=job['plan']['category'], divide_num=1, img_resize=518)
        cfg['models'].update(inference_backend=job['backend'], batch_size=4, backbone_name='ViT-L-14-336',
                             pretrained='openai', feature_layers=[5,11,17,23], r_list=[1,3,5])
        cfg['device'] = str(job['device'])
        cfg['testing'].update(output_dir=str(out/'model_output'), vis=False, save_excel=False)
        report['config'] = cfg
        load_start = time.perf_counter()
        model = musc.MuSc(cfg, seed=job['seed'])
        model.clip_model.eval()
        torch.cuda.synchronize(job['device'])
        report['model_load_seconds'] = time.perf_counter()-load_start
        report['environment'] = {'python': sys.version, 'executable': sys.executable,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(job['device']),
            'platform': platform.platform(), 'cpu': platform.processor(), 'torch_threads': torch.get_num_threads(),
            'cudnn': torch.backends.cudnn.version(), 'tf32': False}
        report['environment']['packages'] = {package: version(package) for package in
            ('torchvision','numpy','scikit-learn','scikit-image','timm','Pillow','PyYAML')}
        # Selection occurs inside load_datasets, preserving its setup within inference.
        load_dataset = model.load_datasets
        actual_paths = []
        def selected_dataset(category, **kwargs):
            dataset = load_dataset(category, **kwargs)
            wanted = [row['path'] for row in job['plan']['image_manifest']]
            root = Path(job['data_path']).resolve()
            indexed = {Path(row[2]).resolve().relative_to(root).as_posix(): row for row in dataset.data_to_iterate}
            if job['plan']['complete_pool'] and set(indexed) != set(wanted):
                raise RuntimeError('Dataset full pool differs from planning manifest')
            dataset.data_to_iterate = [indexed[path] for path in wanted]
            actual_paths[:] = wanted
            return dataset
        model.load_datasets = selected_dataset
        captured = {}
        class PredictionsReady(Exception):
            pass
        def capture(gt_sp, pr_sp, gt_px, pr_px):
            torch.cuda.synchronize(job['device'])
            end = time.perf_counter()
            captured.update(end=end, scores=pr_sp, maps=pr_px, labels=gt_sp, masks=gt_px)
            # End exactly before metrics; no placeholder quality numbers are printed.
            raise PredictionsReady()
        musc.compute_metrics = capture
        musc.tqdm = original_msm.tqdm = lambda x, **kw: x
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(job['device'])
        torch.cuda.synchronize(job['device'])
        report['baseline_allocated_mib'] = torch.cuda.memory_allocated(job['device'])/1024**2
        start = time.perf_counter()
        try:
            model.make_category_data(job['plan']['category'])
        except PredictionsReady:
            pass
        if not captured:
            raise RuntimeError('Pipeline failed to reach pre-metric predictions')
        report.update(inference_seconds=captured['end']-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(job['device'])/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(job['device'])/1024**2,
            images=len(actual_paths), image_paths=actual_paths)
        expected = len(actual_paths)
        for key in ('scores','labels'):
            if np.asarray(captured[key]).shape != (expected,):
                raise RuntimeError(f'Unexpected {key} shape for complete selected pool')
        for key in ('maps','masks'):
            if np.asarray(captured[key]).shape != (expected,1,518,518):
                raise RuntimeError(f'Unexpected {key} shape or image resolution')
        report['ground_truth_sha256'] = {key: hashlib.sha256(np.asarray(captured[key]).tobytes()).hexdigest()
                                         for key in ('labels','masks')}
        if job['backend'] == 'gpu':
            from models.modules._encoder_gpu import research_diagnostics
            report['dispatch'] = research_diagnostics()
            if not report['dispatch'].get('fused_softmax_calls', 0):
                raise RuntimeError('Qualified fused attention dispatch did not execute')
        for key in ('scores', 'maps', 'labels'):
            values = np.asarray(captured[key])
            if not np.isfinite(values).all():
                raise RuntimeError(f'Nonfinite {key}')
            np.save(out/f'{key}.npy', values, allow_pickle=False)
        metrics_start = time.perf_counter()
        report['metrics'] = evaluate(np.asarray(captured['labels']), np.asarray(captured['scores']),
                                     np.asarray(captured['masks']), np.asarray(captured['maps']), job['include_pro'])
        report['metric_evaluation_seconds'] = time.perf_counter()-metrics_start
        report['artifact_sha256'] = {key: digest(out/f'{key}.npy') for key in ('scores','maps','labels')}
        report['source_sha256_at_completion'] = {relative: digest(ROOT/relative) for relative in job['source_sha256']}
        if report['source_sha256_at_completion'] != job['source_sha256']:
            raise RuntimeError('Benchmark source changed during trial; result rejected')
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    dump(out/'trial.json', report)
    return 0 if report['status'] == 'pass' else 1

def run(args):
    categories = CATEGORIES if args.categories == ['all'] else args.categories
    if any(c not in CATEGORIES for c in categories) or len(set(categories)) != len(categories):
        raise ValueError('Choose distinct MVTec AD categories or --categories all')
    plans = plan_pools(args.data_path, categories, args.pool_sizes, args.seed)
    if args.plan_only:
        print(json.dumps({'timing_scope': SCOPE, 'workloads': [{k:v for k,v in p.items() if k != 'image_manifest'} for p in plans]}, indent=2))
        return 0
    if os.name != 'nt':
        raise RuntimeError('This accelerated benchmark requires native Windows')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'name': 'musc-gpu', 'status': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
        'timing_scope': SCOPE, 'protocol': 'fresh process per trial, no discarded image warmup, alternating paired order',
        'filesystem_cache': 'OS filesystem cache is not flushed; cold process is not cold storage',
        'requested_pairs': args.pairs, 'include_pro': args.include_pro, 'workloads': [],
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ('models','datasets','utils','tools','configs')
                          for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.gz','.yaml')}}
    path = out/'results.json'
    dump(path, report)
    try:
        with GpuMutex():
            idle = check_idle(args.device, args.model_endpoint)
            report['initial_idle_evidence'] = idle
            for plan in plans:
                workload = dict(plan, requested_pairs=args.pairs, pairs=[], failures=[])
                report['workloads'].append(workload)
                dump(path, report)
                available = (idle['total_mib']-idle['baseline_used_mib'])/1024
                if plan['estimated_gpu_gib'] > available*.9:
                    workload['summary'] = {'status': 'skipped_memory_assessment', 'speedup': None,
                                           'available_gib': available, 'required_estimate_gib': plan['estimated_gpu_gib']}
                    dump(path, report)
                    continue
                required_disk = plan['estimated_output_disk_gib_per_trial'] * args.pairs * 2 * 1.1
                if shutil.disk_usage(out).free / 1024**3 < required_disk + 1:
                    workload['summary'] = {'status': 'skipped_disk_capacity', 'speedup': None}
                    dump(path, report)
                    continue
                for pair_index in range(args.pairs):
                    order = ['original','gpu'] if pair_index % 2 == 0 else ['gpu','original']
                    row = {'pair': pair_index+1, 'order': order}
                    trial_dirs = {}
                    try:
                        for backend in order:
                            trial_dir = out / plan['id'] / f'pair{pair_index+1}-{backend}'
                            trial_dir.mkdir(parents=True)
                            job = {'backend': backend, 'data_path': str(Path(args.data_path).resolve()),
                                'device': args.device, 'seed': args.seed, 'plan': plan, 'include_pro': args.include_pro,
                                'output': str(trial_dir), 'source_sha256': report['source_sha256'],
                                'worker_token': str(uuid.uuid4())}
                            dump(trial_dir/'job.json', job)
                            before = check_idle(args.device, args.model_endpoint)
                            print(f'{plan["id"]}: pair {pair_index+1}/{args.pairs}, {backend}', flush=True)
                            worker_start = time.perf_counter()
                            with (trial_dir/'stdout.log').open('w') as log:
                                completed = run_worker([sys.executable, str(Path(__file__).resolve()),
                                    '--worker-job', str(trial_dir/'job.json')], cwd=ROOT, log=log,
                                    timeout=args.timeout_seconds, job_token=job['worker_token'],trial_dir=trial_dir)
                            if completed.returncode:
                                raise RuntimeError(f'{backend} worker failed ({completed.returncode}); see {trial_dir}/stdout.log')
                            trial = json.loads((trial_dir/'trial.json').read_text())
                            if trial.get('status') != 'pass':
                                raise RuntimeError(f'{backend} trial is not successful')
                            trial['idle_evidence_before'] = before
                            trial['worker_wall_seconds'] = time.perf_counter()-worker_start
                            row[backend] = trial
                            trial_dirs[backend] = trial_dir
                        import numpy as np
                        agreement = {}
                        for key in ('maps', 'scores'):
                            agreement[key] = compare_arrays(np.load(trial_dirs['original']/f'{key}.npy', mmap_mode='r'),
                                                            np.load(trial_dirs['gpu']/f'{key}.npy', mmap_mode='r'))
                        agreement['labels_identical'] = bool(np.array_equal(np.load(trial_dirs['original']/'labels.npy'),
                                                                          np.load(trial_dirs['gpu']/'labels.npy')))
                        agreement['paths_identical'] = row['original']['image_paths'] == row['gpu']['image_paths']
                        agreement['ground_truth_identical'] = row['original']['ground_truth_sha256'] == row['gpu']['ground_truth_sha256']
                        agreement['pass'] = all(agreement[k]['pass'] for k in ('maps','scores')) and agreement['labels_identical'] and agreement['paths_identical'] and agreement['ground_truth_identical']
                        row['agreement'] = agreement
                        row['speedup'] = row['original']['inference_seconds']/row['gpu']['inference_seconds'] if agreement['pass'] else None
                        row['metric_deltas_gpu_minus_original'] = {key: (value-row['original']['metrics']['values'][key]
                            if value is not None and row['original']['metrics']['values'][key] is not None else None)
                            for key,value in row['gpu']['metrics']['values'].items()}
                        workload['pairs'].append(row)
                        dump(path, report)
                        if not agreement['pass']:
                            raise RuntimeError('Full-image agreement failed; performance claim rejected')
                    except Exception:
                        workload['failures'].append({'pair': pair_index+1, 'partial_trials': row, 'error': traceback.format_exc()})
                        dump(path, report)
                        break
                workload['summary'] = summarize(workload)
                dump(path, report)
            report['final_idle_evidence'] = check_idle(args.device, args.model_endpoint)
        summaries = [w['summary'] for w in report['workloads']]
        report['status'] = 'pass' if all(s['status'] == 'pass' for s in summaries) else 'partial_or_failed'
        successful_full = [w for w in report['workloads'] if w['complete_pool'] and w['summary']['status'] == 'pass']
        # Aggregate only the same validated categories, never mix subset and full pools.
        if successful_full:
            totals = {b: sum(w['summary']['median_inference_seconds'][b] for w in successful_full) for b in ('original','gpu')}
            report['validated_full_category_aggregate'] = {'categories': [w['category'] for w in successful_full],
                'sum_of_category_median_seconds': totals, 'speedup': totals['original']/totals['gpu'],
                'note': 'Ratio of summed per-category medians; excludes skipped/failed categories; not one combined reference pool'}
    except KeyboardInterrupt:
        report['status'] = 'cancelled'
        report['error'] = 'Cancelled by user; active worker process tree cleaned up by runner'
        dump(path, report)
        print(f'cancelled: {path}', flush=True)
        return 130
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    dump(path, report)
    print(f'{report["status"]}: {path}', flush=True)
    return 0 if report['status'] == 'pass' else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--categories', nargs='+', default=['bottle','grid','cable'])
    parser.add_argument('--pool-sizes', nargs='+', default=['full'])
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--model-endpoint', default='http://127.0.0.1:11435/api/ps')
    parser.add_argument('--include-pro', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if args.data_path is None or (not args.plan_only and args.output is None):
        parser.error('--data-path and (unless --plan-only) --output are required')
    if args.pairs < 2 or args.timeout_seconds < 1:
        parser.error('At least two alternating pairs and a positive timeout are required')
    return run(args)

if __name__ == '__main__':
    raise SystemExit(main())
