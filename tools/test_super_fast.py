"""CPU contracts for opt-in preset, safe fallback and patch aggregation lifetime."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.modules._backend import configure_backend, select_backend


class SuperFastTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'models':dict(inference_backend='super_fast',backbone_name='ViT-L-14-336',pretrained='openai',batch_size=4,feature_layers=[5,11,17,23],r_list=[1,3,5]),'datasets':dict(img_resize=518,divide_num=1)}

    def test_opt_in_isolated_preset(self):
        original = deepcopy(self.cfg)
        changed = configure_backend(self.cfg, 'cuda', 'nt')
        self.assertEqual(changed['models']['feature_layers'], [5])
        self.assertEqual(changed['models']['r_list'], [3])
        self.assertEqual(self.cfg, original)
        changed['datasets']['divide_num'] = 2
        self.assertEqual(self.cfg['datasets']['divide_num'], 1)

    def test_fallback_keeps_full_configuration(self):
        for device, platform in [('cpu','nt'),('cuda','posix')]:
            self.assertEqual(select_backend('super_fast', device, platform)[0], 'original')
            self.assertIs(configure_backend(self.cfg, device, platform), self.cfg)
        for requested in ('original', 'gpu'):
            self.cfg['models']['inference_backend'] = requested
            self.assertIs(configure_backend(self.cfg, 'cuda', 'nt'), self.cfg)
            self.assertEqual(self.cfg['models']['feature_layers'], [5,11,17,23])

    def test_rejects_unsupported_profile(self):
        for group, key, value in [('models','backbone_name','ViT-B-32'),('models','pretrained','other'),('models','batch_size',8),('datasets','img_resize',336),('datasets','divide_num',2)]:
            changed=deepcopy(self.cfg)
            changed[group][key]=value
            with self.assertRaisesRegex(ValueError,'super_fast requires'):
                configure_backend(changed,'cuda','nt')
        self.assertEqual(select_backend('super_fast','cuda','nt'), ('super_fast',None))

    def test_pooling_contract_and_factory_lifetime(self):
        import torch
        from torch.nn import functional as F
        from models.modules import _super_fast as fast
        features=torch.arange(2*1369*8,dtype=torch.float32).reshape(2,1369,1,8).half()
        class Base:
            cleared=False
            def __call__(self,*args,**kwargs):return features
            def clear(self):self.cleared=True
        with patch.object(fast.deployed,'new_aggregation',side_effect=Base):
            first=fast.encode_image.new_aggregation()
            second=fast.encode_image.new_aggregation()
        self.assertIsNot(first.base,second.base)
        output=first(SimpleNamespace(r=3), [], return_cpu=False)
        expected=F.adaptive_avg_pool2d(features[:, :, 0].transpose(1,2).reshape(2,8,37,37).float(),(28,28)).half().flatten(2).transpose(1,2).unsqueeze(2).contiguous()
        self.assertTrue(torch.equal(output,expected))
        self.assertEqual(output.shape,(2,784,1,8))
        first.clear()
        self.assertTrue(first.base.cleared)
        self.assertFalse(second.base.cleared)
        with self.assertRaises(ValueError):first(SimpleNamespace(r=1), [])


if __name__=='__main__':unittest.main()
