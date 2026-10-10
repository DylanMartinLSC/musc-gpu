import unittest
import numpy as np
from segmentation_usefulness import choose_threshold, score_image, summarize, wilson


class SegmentationTests(unittest.TestCase):
    def test_perfect_and_empty_masks(self):
        mask = np.zeros((8,8),bool)
        mask[2:4,2:4] = True
        row, regions = score_image(mask,mask)
        self.assertEqual((row['iou'],row['dice'],row['regions_covered_50']), (1,1,1))
        self.assertEqual(regions[0]['coverage'],1)
        empty, _ = score_image(np.zeros_like(mask),np.zeros_like(mask))
        self.assertIsNone(empty['iou'])
        self.assertEqual(empty['unmatched_predicted_components'],0)

    def test_blanket_prediction_is_penalized(self):
        mask = np.zeros((10,10),bool)
        mask[0,0] = True
        row, _ = score_image(np.ones_like(mask),mask)
        self.assertEqual(row['regions_covered_50'],1)
        self.assertEqual(row['precision'],.01)
        self.assertEqual(row['iou'],.01)

    def test_miss_false_component_and_connectivity(self):
        truth = np.zeros((10,10),bool)
        truth[1,1] = truth[2,2] = True
        pred = np.zeros_like(truth)
        pred[8,8] = True
        row, regions = score_image(pred,truth)
        self.assertEqual(row['gt_regions'],1)
        self.assertEqual(row['unmatched_predicted_components'],1)
        self.assertEqual(row['regions_covered_50'],0)
        self.assertEqual(regions[0]['area_pixels'],2)

    def test_calibration_insufficient_and_ties(self):
        threshold, info = choose_threshold([1]*10,.05)
        self.assertEqual(threshold,1)
        self.assertFalse(info['rank_available'])
        threshold, info = choose_threshold(range(19),.05)
        self.assertEqual(threshold,18)
        self.assertTrue(info['rank_available'])
        self.assertFalse((np.array([18]) > threshold).any())
        with self.assertRaises(ValueError):
            choose_threshold([np.nan],.05)

    def test_image_denominators_and_uncertainty(self):
        normal, _ = score_image(np.zeros((4,4)),np.zeros((4,4)))
        defect, _ = score_image(np.eye(4),np.eye(4))
        result = summarize([normal,defect])
        self.assertEqual(result['normal_images'],1)
        self.assertEqual(result['localized_defective_image_rate'],1)
        self.assertGreater(wilson(0,20)[1],.16)
        self.assertIsNone(wilson(0,0))


if __name__ == '__main__':
    unittest.main()
