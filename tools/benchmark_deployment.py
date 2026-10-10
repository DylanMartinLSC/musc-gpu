"""Bounded resident-model batch-inspection replay, with serialized GPU ownership."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
import traceback
import uuid

from benchmark import ROOT, GpuMutex, check_idle
from benchmark_process import run_worker, worker_ready
from benchmark_support import compare_arrays, digest, dump, plan_pools, CATEGORIES

SCOPE = ('Resident model; request start through CPU maps/scores written and closed as .npy files. '
         'Includes image decode/preprocess, transfers, complete-pool inference, RsCIN, '
         'shape/finite checks and output serialization to OS-buffered local storage. '
         'First request includes first-use kernel setup; no discarded warmup. '
         'No ground-truth reads, quality metrics, network transport or camera acquisition. '
         'Cumulative session time also includes inter-request instrumentation/cleanup.')


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for folder in ('models','datasets','utils','tools','configs')
            for p in (ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ('.py','.json','.yaml','.gz')}


def summarize_session(trial):
    rows = trial['requests']
    if not rows or any(not math.isfinite(r['service_seconds']) or r['service_seconds'] <= 0 for r in rows):
        raise ValueError('Invalid or missing request timing')
    total = rows[-1]['completed_at_seconds']
    if total <= 0 or any(b['completed_at_seconds'] <= a['completed_at_seconds'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Completion times must increase')
    images = sum(r['images'] for r in rows)
    return {'completed_images': images, 'elapsed_seconds': total,
            'images_per_second': images/total,
            'first_request_seconds': rows[0]['service_seconds'],
            'subsequent_request_median_seconds': statistics.median(r['service_seconds'] for r in rows[1:]) if len(rows)>1 else None,
            'peak_allocated_mib': max(r['peak_allocated_mib'] for r in rows),
            'peak_reserved_mib': max(r['peak_reserved_mib'] for r in rows)}


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    out = Path(job['output'])
    report = {'status':'running','backend':job['backend'],'requests':[], 'timing_scope':SCOPE}
    dump(out/'trial.json',report)
    try:
        if os.environ.get('MUSC_BENCHMARK_WORKER_TOKEN') != job['worker_token']:
            raise RuntimeError('Worker must be started by the GPU-owning coordinator')
        worker_ready(out,job['worker_token'])
        if source_hashes() != job['source_sha256']:
            raise RuntimeError('Source changed before worker start')
        ready = time.perf_counter()
        import numpy as np
        import torch
        import yaml
        from PIL import Image
        import models.musc as musc
        import models.modules._MSM as original_msm
        from models.modules._backend import select_backend
        if not torch.cuda.is_available() or select_backend('gpu','cuda')[0] != 'gpu':
            raise RuntimeError('Native Windows CUDA required; fallback cannot count as acceleration')
        torch.cuda.set_device(job['device'])
        torch.manual_seed(job['seed'])
        np.random.seed(job['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize(job['device'])
        report['imports_and_cuda_setup_seconds'] = time.perf_counter()-ready
        cfg = yaml.safe_load((ROOT/'configs/musc.yaml').read_text())
        cfg['datasets'].update(data_path=job['data_path'],dataset_name='mvtec_ad',
                               class_name=job['plan']['category'],divide_num=1,img_resize=518)
        cfg['models'].update(inference_backend=job['backend'],batch_size=4,
                             backbone_name='ViT-L-14-336',pretrained='openai',
                             feature_layers=[5,11,17,23],r_list=[1,3,5])
        cfg['device'] = str(job['device'])
        cfg['testing'].update(output_dir=str(out/'model_output'),vis=False,save_excel=False)
        report['config'] = cfg
        load_start = time.perf_counter()
        model = musc.MuSc(cfg,seed=job['seed'])
        model.clip_model.eval()
        torch.cuda.synchronize(job['device'])
        report['model_load_seconds'] = time.perf_counter()-load_start
        report['environment'] = {'python':sys.version,'torch':torch.__version__,
                                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(job['device']),
                                 'platform':platform.platform(),'numpy':np.__version__,
                                 'tf32':False,'torch_threads':torch.get_num_threads()}
        paths = [r['path'] for r in job['plan']['image_manifest']]
        decoded = []
        encoded = [0]
        class ImagePool(torch.utils.data.Dataset):
            def __len__(self):
                return len(paths)
            def __getitem__(self,index):
                path = Path(job['data_path'])/paths[index]
                with Image.open(path) as image:
                    value = model.preprocess(image.convert('RGB'))
                decoded.append(index)
                # Unused placeholders satisfy the unchanged caller's metric bookkeeping.
                # No dataset ground-truth directory or annotation is accessed.
                return {'image':value,'image_path':paths[index],
                        'mask':torch.zeros((1,518,518)), 'is_anomaly':0}
        model.load_datasets = lambda category, **kwargs: ImagePool()
        def count_forward(module,inputs):
            encoded[0] += int(inputs[0].shape[0])
        hook = model.clip_model.visual.register_forward_pre_hook(count_forward)
        captured = {}
        class PredictionsReady(Exception):
            pass
        def capture(gt_sp,pr_sp,gt_px,pr_px):
            torch.cuda.synchronize(job['device'])
            captured.update(scores=pr_sp,maps=pr_px,end=time.perf_counter())
            raise PredictionsReady()
        musc.compute_metrics = capture
        musc.tqdm = original_msm.tqdm = lambda x,**kwargs:x
        torch.cuda.empty_cache()
        torch.cuda.synchronize(job['device'])
        session_start = time.perf_counter()
        report['model_ready_seconds_since_worker_ready'] = session_start-ready
        for index in range(job['requests']):
            captured.clear()
            decoded.clear()
            encoded[0] = 0
            request_dir = out/f'request{index+1}'
            request_dir.mkdir()
            torch.cuda.reset_peak_memory_stats(job['device'])
            torch.cuda.synchronize(job['device'])
            baseline = torch.cuda.memory_allocated(job['device'])/1024**2
            start = time.perf_counter()
            try:
                model.make_category_data(job['plan']['category'])
            except PredictionsReady:
                pass
            if not captured:
                raise RuntimeError('Caller did not deliver predictions')
            if sorted(decoded) != list(range(len(paths))) or encoded[0] != len(paths):
                raise RuntimeError('Each request must decode and encode every image exactly once')
            shapes = {'maps':(len(paths),1,518,518),'scores':(len(paths),)}
            for key,shape in shapes.items():
                values = np.asarray(captured[key])
                if values.shape != shape or not np.isfinite(values).all():
                    raise RuntimeError(f'Invalid {key} outputs')
                np.save(request_dir/f'{key}.npy',values,allow_pickle=False)
            dump(request_dir/'image_paths.json',paths)
            end = time.perf_counter()
            row = {'request':index+1,'images':len(paths),'decoded_images':len(decoded),
                   'encoded_images':encoded[0],'started_at_seconds':start-session_start,
                   'inference_seconds':captured['end']-start,
                   'service_seconds':end-start,'completed_at_seconds':end-session_start,
                   'result_write_seconds':end-captured['end'],
                   'baseline_allocated_mib':baseline,
                   'peak_allocated_mib':torch.cuda.max_memory_allocated(job['device'])/1024**2,
                   'peak_reserved_mib':torch.cuda.max_memory_reserved(job['device'])/1024**2,
                   'post_call_allocated_mib':torch.cuda.memory_allocated(job['device'])/1024**2,
                   'artifacts':{key:f'request{index+1}/{key}.npy' for key in shapes}}
            if job['backend'] == 'gpu':
                from models.modules._encoder_gpu import research_diagnostics
                row['dispatch'] = research_diagnostics()
                if not row['dispatch'].get('fused_softmax_calls',0):
                    raise RuntimeError('Qualified GPU attention did not dispatch')
            captured.clear()
            del values
            report['requests'].append(row)
            dump(out/'trial.json',report)
            print(f"request {index+1}/{job['requests']}: {row['service_seconds']:.3f}s, {row['encoded_images']} images encoded",flush=True)
        hook.remove()
        report['summary'] = summarize_session(report)
        report['worker_ready_to_last_result_seconds'] = time.perf_counter()-ready
        report['source_sha256_at_completion'] = source_hashes()
        if report['source_sha256_at_completion'] != job['source_sha256']:
            raise RuntimeError('Source changed during session')
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    dump(out/'trial.json',report)
    return 0 if report['status'] == 'pass' else 1


def compare_sessions(original_dir,gpu_dir,requests):
    import numpy as np
    checks = []
    for index in range(1,requests+1):
        row = {'request':index}
        for key in ('maps','scores'):
            original = np.load(original_dir/f'request{index}/{key}.npy',mmap_mode='r')
            gpu = np.load(gpu_dir/f'request{index}/{key}.npy',mmap_mode='r')
            row[key] = compare_arrays(original,gpu)
            for backend,directory,current in [('original',original_dir,original),('gpu',gpu_dir,gpu)]:
                first = np.load(directory/f'request1/{key}.npy',mmap_mode='r')
                row[f'{backend}_{key}_repeat'] = compare_arrays(first,current)
        paths = [json.loads((directory/f'request{index}/image_paths.json').read_text())
                 for directory in (original_dir,gpu_dir)]
        row['paths_identical'] = paths[0] == paths[1]
        row['pass'] = row['paths_identical'] and all(v['pass'] for v in row.values() if isinstance(v,dict))
        checks.append(row)
    return {'pass':all(r['pass'] for r in checks),'requests':checks}


def run(args):
    plan = plan_pools(args.data_path,[args.category],['full'],args.seed)[0]
    for row in plan['image_manifest']:
        row['sha256'] = digest(args.data_path/row['path'])
    plan['content_manifest_sha256'] = hashlib.sha256(json.dumps(plan['image_manifest'],sort_keys=True).encode()).hexdigest()
    if args.plan_only:
        print(json.dumps(dict(plan=plan,requests_per_session=args.requests,pairs=args.pairs,
                              scope=SCOPE,processed_images_per_backend=plan['images']*args.requests*args.pairs),indent=2))
        return 0
    if os.name != 'nt':
        raise RuntimeError('Native Windows coordinator required')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report = {'schema_version':1,'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'command':sys.argv,'timing_scope':SCOPE,'plan':plan,'requests_per_session':args.requests,
              'requested_pairs':args.pairs,'pairs':[],'source_sha256':source_hashes(),
              'protocol':'original/GPU then GPU/original; fresh process each session; resident model within session',
              'limitations':['Repeated same complete pool, not unique production parts or larger reference pools.',
                             'Finite saturated workstation replay, not factory deployment or long-duration reliability.',
                             'OS disk caches not flushed; buffered output close is not durable fsync.',
                             'No camera, network, PLC or actual production arrivals; all requests ready at session start.',
                             'No inference feature caching; only model weights and runtime allocations remain resident.']}
    dump(out/'results.json',report)
    try:
        with GpuMutex():
            idle = check_idle(args.device,args.model_endpoint)
            report['initial_idle_evidence'] = idle
            available = (idle['total_mib']-idle['baseline_used_mib'])/1024
            if plan['estimated_gpu_gib'] > available*.9:
                raise RuntimeError('Conservative full-pool memory assessment failed')
            required = plan['estimated_output_disk_gib_per_trial']*args.requests*args.pairs*2*1.2+1
            if shutil.disk_usage(out).free/1024**3 < required:
                raise RuntimeError('Insufficient disk for full prediction evidence')
            for pair_index in range(args.pairs):
                pair = {'pair':pair_index+1,'sessions':{}}
                report['pairs'].append(pair)
                directories = {}
                for backend in (('original','gpu') if pair_index%2 == 0 else ('gpu','original')):
                    directory = out/f'pair{pair_index+1}-{backend}'
                    directory.mkdir()
                    directories[backend] = directory
                    before = check_idle(args.device,args.model_endpoint)
                    job = dict(plan=plan,backend=backend,requests=args.requests,seed=args.seed,
                               device=args.device,data_path=str(args.data_path.resolve()),output=str(directory),
                               worker_token=str(uuid.uuid4()),source_sha256=report['source_sha256'])
                    dump(directory/'job.json',job)
                    print(f"pair {pair_index+1}/{args.pairs}: {backend}, {args.requests} full-pool requests",flush=True)
                    with (directory/'stdout.log').open('w') as log:
                        completed = run_worker([sys.executable,str(Path(__file__).resolve()),'--worker-job',str(directory/'job.json')],
                                               cwd=ROOT,log=log,timeout=args.timeout_seconds,
                                               job_token=job['worker_token'],trial_dir=directory)
                    if completed.returncode:
                        raise RuntimeError(f'{backend} session failed: {directory}/stdout.log')
                    trial = json.loads((directory/'trial.json').read_text())
                    if trial['status'] != 'pass' or len(trial['requests']) != args.requests:
                        raise RuntimeError('Incomplete session; no performance claim')
                    trial['idle_evidence_before'] = before
                    pair['sessions'][backend] = trial
                    dump(out/'results.json',report)
                pair['agreement'] = compare_sessions(directories['original'],directories['gpu'],args.requests)
                if not pair['agreement']['pass']:
                    raise RuntimeError('Numerical agreement failed; capacity claim rejected')
                pair['delivered_speedup'] = pair['sessions']['original']['summary']['elapsed_seconds']/pair['sessions']['gpu']['summary']['elapsed_seconds']
                dump(out/'results.json',report)
            report['final_idle_evidence'] = check_idle(args.device,args.model_endpoint)
            if any(digest(args.data_path/r['path']) != r['sha256'] for r in plan['image_manifest']):
                raise RuntimeError('Input content changed during benchmark')
        report['summary'] = {}
        for backend in ('original','gpu'):
            sessions = [p['sessions'][backend]['summary'] for p in report['pairs']]
            total_images = sum(s['completed_images'] for s in sessions)
            total_seconds = sum(s['elapsed_seconds'] for s in sessions)
            report['summary'][backend] = dict(completed_images=total_images,measured_session_seconds=total_seconds,
                                             images_per_second=total_images/total_seconds,
                                             session_elapsed_seconds=[s['elapsed_seconds'] for s in sessions],
                                             peak_allocated_mib=max(s['peak_allocated_mib'] for s in sessions))
        report['summary']['delivered_throughput_ratio'] = report['summary']['gpu']['images_per_second']/report['summary']['original']['images_per_second']
        report['status'] = 'pass'
    except (Exception,KeyboardInterrupt):
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    dump(out/'results.json',report)
    print(f"{report['status']}: {out/'results.json'}",flush=True)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--category',choices=CATEGORIES,default='bottle')
    parser.add_argument('--requests',type=int,default=3)
    parser.add_argument('--pairs',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--timeout-seconds',type=int,default=600)
    parser.add_argument('--model-endpoint',default='http://127.0.0.1:11435/api/ps')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker-job',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if args.data_path is None or (args.output is None and not args.plan_only):
        parser.error('--data-path and --output required (output optional for --plan-only)')
    if args.requests < 2 or args.pairs < 2 or args.timeout_seconds <= 0:
        parser.error('At least two requests, two alternating pairs and a positive timeout required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
