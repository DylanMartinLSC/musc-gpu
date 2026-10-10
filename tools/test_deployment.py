import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark_deployment import summarize_session, compare_sessions


class DeploymentTests(unittest.TestCase):
    def test_totals_include_first_call_and_between_request_overhead(self):
        result = summarize_session({'requests':[
            dict(images=83,service_seconds=10,completed_at_seconds=10,peak_allocated_mib=20,peak_reserved_mib=30),
            dict(images=83,service_seconds=8,completed_at_seconds=19,peak_allocated_mib=22,peak_reserved_mib=32)]})
        self.assertEqual(result['completed_images'],166)
        self.assertEqual(result['images_per_second'],166/19)
        self.assertEqual(result['first_request_seconds'],10)
        self.assertEqual(result['subsequent_request_median_seconds'],8)

    def test_invalid_timing(self):
        with self.assertRaises(ValueError):
            summarize_session({'requests':[]})
        with self.assertRaises(ValueError):
            summarize_session({'requests':[dict(service_seconds=float('nan'))]})

    def test_every_request_and_repeated_state_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backend in ('original','gpu'):
                for request in (1,2):
                    out = root/backend/f'request{request}'
                    out.mkdir(parents=True)
                    for name in ('maps','scores'):
                        np.save(out/f'{name}.npy',np.zeros((2,2)))
                    (out/'image_paths.json').write_text(json.dumps(['a','b']))
            self.assertTrue(compare_sessions(root/'original',root/'gpu',2)['pass'])
            # Both backends drift together: cross-backend agreement alone would pass.
            for backend in ('original','gpu'):
                np.save(root/backend/'request2/maps.npy',np.ones((2,2)))
            result = compare_sessions(root/'original',root/'gpu',2)
            self.assertFalse(result['pass'])
            self.assertTrue(result['requests'][1]['maps']['pass'])
            self.assertFalse(result['requests'][1]['gpu_maps_repeat']['pass'])


if __name__ == '__main__':
    unittest.main()
