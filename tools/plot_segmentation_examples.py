"""Plot measured masks, not normalized heatmaps or generated illustrations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentation_usefulness import digest, load_trial, score_image


def overlay(image, mask, color):
    result = image.astype(float)/255
    result[mask] = .5*result[mask] + .5*np.asarray(color)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())['results']
    args.output.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(len(results),4,figsize=(12,3.45*len(results)+1),squeeze=False)
    selected = []
    for row, result in enumerate(results):
        # Lower median by held-out defective-image IoU; lexical path breaks ties.
        ordered = sorted([r for r in result['images'] if r['defective']],
                         key=lambda r:(r['iou'],r['path']))
        chosen = ordered[(len(ordered)-1)//2]
        gpu_dir = Path(result['provenance']['trial'])
        if digest(gpu_dir/'trial.json') != result['provenance']['trial_sha256']:
            raise ValueError('Trial changed since evaluation')
        original_dir = gpu_dir.with_name(gpu_dir.name.replace('-gpu','-original'))
        _, gpu, masks, paths = load_trial(gpu_dir,args.dataset)
        _, original, original_masks, original_paths = load_trial(original_dir,args.dataset)
        if paths != original_paths or not np.array_equal(masks,original_masks):
            raise ValueError('Original/GPU ground truth or ordering mismatch')
        index = paths.index(chosen['path'])
        threshold = result['threshold']
        gt = masks[index,0].astype(bool)
        op = original[index,0] > threshold
        gp = gpu[index,0] > threshold
        with Image.open(args.dataset/chosen['path']) as source:
            rgb = np.asarray(source.convert('RGB').resize((518,518),Image.Resampling.BICUBIC))
        original_metrics, _ = score_image(op,gt)
        gpu_metrics, _ = score_image(gp,gt)
        panels = [rgb,overlay(rgb,gt,(.15,.95,.4)),
                  overlay(rgb,op,(1,.45,.08)),overlay(rgb,gp,(1,.45,.08))]
        labels = [f"{result['category'].capitalize()} · {chosen['defect_type']}",
                  'Annotated defect area',
                  f"IoU {original_metrics['iou']:.1%}",f"IoU {gpu_metrics['iou']:.1%}"]
        for col,(panel,label) in enumerate(zip(panels,labels)):
            ax = axes[row,col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(label,fontsize=10)
        selected.append(dict(category=result['category'],path=chosen['path'],threshold=threshold,
                             selection_rank=(len(ordered)-1)//2+1,defective_images=len(ordered),
                             original_iou=original_metrics['iou'],gpu_iou=gpu_metrics['iou'],
                             thresholded_pixel_disagreements=int(np.count_nonzero(op!=gp)),
                             image_sha256=digest(args.dataset/chosen['path']),
                             original_trial_sha256=digest(original_dir/'trial.json')))
    for ax,title in zip(axes[0],['Input image','Ground truth','Original MuSc','musc-gpu']):
        ax.set_title(title,fontsize=13,weight='bold',pad=12)
    fig.suptitle('Fixed-threshold segmentation: original and accelerated MuSc',fontsize=16,weight='bold',y=.99)
    fig.text(.5,.042,'Green: ground truth. Orange: predicted defect pixels. Same category threshold for both backends.',ha='center',fontsize=10)
    fig.text(.5,.022,'One median-IoU defective image per category; retrospective shared-pool examples, not production qualification.',ha='center',fontsize=9,color='#475467')
    fig.tight_layout(rect=(0,.065,1,.965),h_pad=1.5,w_pad=.7)
    fig.savefig(args.output/'examples.png',dpi=160,facecolor='white')
    plt.close(fig)
    (args.output/'examples.json').write_text(json.dumps(dict(
        selection='Lower median held-out defective-image GPU IoU per category, path tie-break',
        display='518px resized input with 50% alpha overlays; exact evaluation masks; no heatmap normalization',
        threshold='Frozen GPU category calibration threshold applied unchanged to both backends',
        results_sha256=digest(args.results),script_sha256=digest(__file__),examples=selected),indent=2))


if __name__ == '__main__':
    main()
