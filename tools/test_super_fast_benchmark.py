import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('trial',HERE/'benchmark_super_fast.py')
trial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


class Tests(unittest.TestCase):
    def test_exact_pro(self):
        self.assertLess(trial.metric_preflight(),1e-12)

    def test_actual_speed_memory_and_percentage_points(self):
        def row(seconds,memory,metric):return {'inference_seconds':seconds,'peak_allocated_mib':memory,'peak_reserved_mib':memory+100,'metrics':{'values':{'pixel_auroc':metric}}}
        pairs=[{'same_inputs':True,'original':row(40,4000,.98),'super_fast':row(4,3000,.975)}, {'same_inputs':True,'original':row(36,4200,.98),'super_fast':row(3,3000,.975)}]
        s=trial.summarize(pairs)
        self.assertEqual(s['paired_speedups'],[10,12])
        self.assertEqual(s['speedup'],38/3.5)
        self.assertAlmostEqual(s['metric_deltas_pp']['pixel_auroc'],-.5)
        self.assertAlmostEqual(s['allocated_memory_change_percent'],100*(3000/4100-1))
        self.assertTrue(s['all_metrics_within_provisional_1pp'])
        with self.assertRaises(ValueError):trial.summarize(pairs[:1])
        pairs[0]['same_inputs']=False
        with self.assertRaises(ValueError):trial.summarize(pairs)


if __name__=='__main__':unittest.main()
