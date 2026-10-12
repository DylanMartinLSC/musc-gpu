"""Two fresh-process original/super_fast pairs; approximate accuracy tradeoffs."""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import traceback
import uuid

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'tools'))
import benchmark
from benchmark_support import plan_pools,digest,dump,compare_arrays,SCOPE
from benchmark_process import run_worker


def exact_pro(masks,maps):
    """Upstream200 global thresholds/region weighting via exact sorted searches."""
    import numpy as np
    from skimage import measure
    from sklearn.metrics import auc
    low,high=float(maps.min()),float(maps.max())
    if not masks.any() or high==low:raise ValueError('Undefined PRO')
    thresholds=np.arange(low,high,(high-low)/200)
    negative=np.sort(maps[masks==0])
    fpr=(negative.size-np.searchsorted(negative,thresholds,side='right'))/negative.size
    regions=[]
    for mask,amap in zip(masks.squeeze(1),maps.squeeze(1)):
        for region in measure.regionprops(measure.label(mask)):
            values=np.sort(amap[region.coords[:,0],region.coords[:,1]])
            regions.append((values.size-np.searchsorted(values,thresholds,side='right'))/values.size)
    pro=np.mean(regions,axis=0)
    keep=fpr<0.3
    x=fpr[keep]
    return float(auc((x-x.min())/(x.max()-x.min()),pro[keep]))


def metric_preflight():
    import numpy as np
    from utils.metrics import cal_pro_score
    rng=np.random.default_rng(711)
    masks=np.zeros((3,1,12,12),dtype=np.int32)
    masks[:,0,2:5,3:7]=1
    maps=rng.random(masks.shape)
    error=abs(exact_pro(masks,maps)-cal_pro_score(masks.squeeze(1),maps.squeeze(1)))
    assert error<1e-12
    return error


def worker(job_path):
    evaluate=benchmark.evaluate
    def with_pro(labels,scores,masks,maps,include_pro):
        result=evaluate(labels,scores,masks,maps,False)
        result['values']['pixel_aupro']=exact_pro(masks,maps)
        result['errors'].pop('pixel_aupro',None)
        result['definition']+='; exact sorted evaluation of upstream200 PRO thresholds'
        return result
    benchmark.evaluate=with_pro
    code=benchmark.worker(job_path)
    job=json.loads(Path(job_path).read_text())
    path=Path(job['output'])/'trial.json'
    report=json.loads(path.read_text())
    if code==0:
        from models.modules._backend import configure_backend
        report['effective_config']=configure_backend(report['config'],'cuda')
        report['comparison_adapter_sha256']=digest(Path(__file__).resolve())
        assert report['comparison_adapter_sha256']==job['comparison_adapter_sha256']
        if job['backend']=='super_fast':
            assert report['effective_config']['models']['feature_layers']==[5]
            assert report['effective_config']['models']['r_list']==[3]
        dump(path,report)
    return code


class ResearchLease:
    def __enter__(self):
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
        api.CreateMutexW.restype=wintypes.HANDLE
        api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        api.ReleaseMutex.argtypes=[wintypes.HANDLE]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        self.api=api
        self.handle=api.CreateMutexW(None,False,'Local\\MuScContinuousHermes')
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        if api.WaitForSingleObject(self.handle,0) not in (0,128):
            api.CloseHandle(self.handle)
            raise RuntimeError('Research coordination lease is busy')
        return self
    def __exit__(self,*args):
        self.api.ReleaseMutex(self.handle)
        self.api.CloseHandle(self.handle)


def summarize(pairs):
    if len(pairs)!=2:raise ValueError('Two complete pairs are required')
    if not all(p['same_inputs'] for p in pairs):raise ValueError('Ground truth or reference pool differs')
    result={'comparison':'approximate super_fast versus original MuSc; not original-contract exact-output qualification',
            'paired_speedups':[p['original']['inference_seconds']/p['super_fast']['inference_seconds'] for p in pairs]}
    result['median_inference_seconds']={b:statistics.median(p[b]['inference_seconds'] for p in pairs) for b in ('original','super_fast')}
    result['speedup']=result['median_inference_seconds']['original']/result['median_inference_seconds']['super_fast']
    result['all_pairs_reach_10x_approximate']=all(r>=10 for r in result['paired_speedups'])
    result['median_peak_allocated_mib']={b:statistics.median(p[b]['peak_allocated_mib'] for p in pairs) for b in ('original','super_fast')}
    result['median_peak_reserved_mib']={b:statistics.median(p[b]['peak_reserved_mib'] for p in pairs) for b in ('original','super_fast')}
    result['allocated_memory_change_percent']=100*(result['median_peak_allocated_mib']['super_fast']/result['median_peak_allocated_mib']['original']-1)
    result['metrics']={b:{key:statistics.median(p[b]['metrics']['values'][key] for p in pairs) for key in pairs[0][b]['metrics']['values']} for b in ('original','super_fast')}
    result['metric_deltas_pp']={k:100*(v-result['metrics']['original'][k]) for k,v in result['metrics']['super_fast'].items()}
    result['all_metrics_within_provisional_1pp']=all(v>=-1 for v in result['metric_deltas_pp'].values())
    return result


def run(args):
    plans=plan_pools(args.data_path,['bottle'],['full'],42)
    assert len(plans)==1 and plans[0]['images']==83
    if args.plan_only:
        print(json.dumps({'scope':SCOPE,'plan':plans[0],'order':[['original','super_fast'],['super_fast','original']]},indent=2))
        return 0
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','command':sys.argv,'scope':SCOPE,'protocol':'fresh child process per backend per pair; original/super_fast then super_fast/original; no discarded warmup; metrics and model loading separate; OS file cache not flushed',
            'approximate':True,'images':83,'complete_pool':True,'pairs':[],
            'adapter_sha256':digest(Path(__file__).resolve()),
            'source_sha256':{str(p.relative_to(PACKAGE)):digest(p) for folder in ('models','datasets','utils','tools','configs') for p in (PACKAGE/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.gz','.yaml')}}
    path=out/'results.json'
    dump(path,report)
    try:
        if args.coordinator_status:
            status=json.loads(args.coordinator_status.read_text())
            if status['phase']!='stopped':raise RuntimeError('Research coordinator is active')
            report['coordinator_status']=status
        # No CUDA work occurs before process inventory, unload verification and locks.
        report['initial_idle_evidence']=benchmark.check_idle(0,'http://127.0.0.1:11435/api/ps')
        report['metric_preflight_error']=metric_preflight()
        if shutil.disk_usage(out).free<2*1024**3:raise RuntimeError('Insufficient output capacity')
        with ResearchLease(),benchmark.GpuMutex():
            for index,order in enumerate((('original','super_fast'),('super_fast','original')),1):
                pair={'pair':index,'order':list(order)}
                for backend in order:
                    before=benchmark.check_idle(0,'http://127.0.0.1:11435/api/ps')
                    trial=out/f'pair{index}-{backend}'
                    trial.mkdir()
                    job={'backend':backend,'data_path':str(args.data_path.resolve()),'device':0,'seed':42,'plan':plans[0],'include_pro':True,'output':str(trial),'source_sha256':report['source_sha256'],'comparison_adapter_sha256':report['adapter_sha256'],'worker_token':str(uuid.uuid4())}
                    dump(trial/'job.json',job)
                    print(f'Pair{index}: {backend}',flush=True)
                    with (trial/'stdout.log').open('w',encoding='utf-8') as log:
                        done=run_worker([sys.executable,str(Path(__file__).resolve()),'--worker-job',str(trial/'job.json')],cwd=PACKAGE,log=log,timeout=600,job_token=job['worker_token'],trial_dir=trial)
                    if done.returncode:raise RuntimeError(f'{backend} failed; see {trial}/stdout.log')
                    record=json.loads((trial/'trial.json').read_text())
                    if record['status']!='pass':raise RuntimeError('Worker rejected')
                    record['idle_before']=before
                    pair[backend]=record
                    report['in_progress_pair']=pair
                    dump(path,report)
                import numpy as np
                a,b=pair['original'],pair['super_fast']
                pair['same_inputs']=a['image_paths']==b['image_paths'] and a['ground_truth_sha256']==b['ground_truth_sha256']
                pair['map_difference']=compare_arrays(np.load(out/f'pair{index}-original/maps.npy',mmap_mode='r'),np.load(out/f'pair{index}-super_fast/maps.npy',mmap_mode='r'))
                pair['score_difference']=compare_arrays(np.load(out/f'pair{index}-original/scores.npy',mmap_mode='r'),np.load(out/f'pair{index}-super_fast/scores.npy',mmap_mode='r'))
                report['pairs'].append(pair)
                report.pop('in_progress_pair',None)
                dump(path,report)
            for backend in ('original','super_fast'):
                checks={k:compare_arrays(np.load(out/f'pair1-{backend}/{k}.npy',mmap_mode='r'),np.load(out/f'pair2-{backend}/{k}.npy',mmap_mode='r')) for k in ('maps','scores')}
                if not all(c['pass'] for c in checks.values()):raise RuntimeError('Repeated-call outputs disagree')
                report.setdefault('repeat_checks',{})[backend]=checks
            report['final_idle_evidence']=benchmark.check_idle(0,'http://127.0.0.1:11435/api/ps')
        report['summary']=summarize(report['pairs'])
        report['status']='complete approximate comparison'
    except Exception:
        report['status']='blocked_or_failed'
        report['error']=traceback.format_exc()
        traceback.print_exc()
    finally:dump(path,report)
    return 0 if report['status']=='complete approximate comparison' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',type=Path,default=PACKAGE/'data/mvtec_anomaly_detection')
    parser.add_argument('--output',type=Path,default=PACKAGE/'output/super_fast_benchmark')
    parser.add_argument('--coordinator-status',type=Path,help='optional external research coordinator status JSON; must be stopped')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker-job',type=Path)
    args=parser.parse_args()
    sys.exit(worker(args.worker_job) if args.worker_job else run(args))
