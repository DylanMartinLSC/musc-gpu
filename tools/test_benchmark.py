"""CPU-only regression checks for benchmark contracts; no torch/CUDA imports."""
from pathlib import Path
import tempfile
import numpy as np
from benchmark_support import compare_arrays, plan_pools, summarize, dump

def run():
    a = np.arange(24., dtype=np.float64).reshape(6,4)
    assert compare_arrays(a,a)['pass']
    assert compare_arrays(a,a+1e-5)['pass']
    assert not compare_arrays(a,a+3e-5)['pass']
    assert not compare_arrays(a,a[:2])['pass']
    assert not compare_arrays(a[:0],a[:0])['pass']
    b=a.copy()
    b[0,0]=np.nan
    assert not compare_arrays(a,b)['pass']
    w=dict(requested_pairs=2,images=83,failures=[],pairs=[
        dict(agreement={'pass':True},original={'inference_seconds':4.},gpu={'inference_seconds':1.}) for _ in range(2)])
    assert summarize(w)['speedup']==4.
    w['failures'].append({'error':'failure'})
    assert summarize(w)['speedup'] is None
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for anomaly in ('good','broken'):
            folder=root/'bottle/test'/anomaly
            folder.mkdir(parents=True)
            for i in range(6):
                (folder/f'{i}.png').write_bytes(b'fixture')
        plans=plan_pools(root,['bottle'],['full','5'])
        assert plans[0]['complete_pool'] and plans[0]['images']==12
        assert not plans[1]['complete_pool'] and plans[1]['images']==5
        assert plans[1]==plan_pools(root,['bottle'],['5'])[0]
        try:
            plan_pools(root,['bottle'],['5','05'])
            raise AssertionError('Duplicate normalized pool size accepted')
        except ValueError:
            pass
        dump(root/'roundtrip.json',{'ok':True})
        assert (root/'roundtrip.json').exists() and not (root/'roundtrip.json.tmp').exists()
    print('PASS: numerical gates, shape/nonfinite rejection, failed-claim suppression, pool selection and atomic reporting.')

if __name__=='__main__':
    run()
