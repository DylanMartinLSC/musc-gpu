"""Regenerate fixed-threshold segmentation figure from retained measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'benchmarks/segmentation_usefulness'
results = json.loads((OUT/'results.json').read_text())['results']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(9,5.3))
x = np.arange(len(results))
for offset, (key,label,color) in enumerate([
        ('pixel_precision','Pixel precision','#147d92'),
        ('pixel_recall','Pixel recall','#df8a35'),
        ('pixel_iou','Pixel IoU','#6574aa')]):
    values = [100*r['metrics'][key] for r in results]
    bars = ax.bar(x+(offset-1)*.24,values,width=.22,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=10)
ax.set_xticks(x,[r['category'].capitalize() for r in results])
ax.set_ylim(0,105)
ax.set_yticks(np.arange(0,101,20))
ax.set_ylabel('Percent · higher is better')
ax.set_title('Defect segmentation at fixed category thresholds',loc='left',pad=48,weight='bold')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False)
ax.grid(axis='y',alpha=.18)
ax.set_axisbelow(True)
fig.text(.08,.065,'Calibration uses reserved normal images; no evaluation-defect threshold tuning.',fontsize=10,color='#475467')
fig.text(.08,.025,'Shared-pool diagnostic. Bottle/grid calibration is insufficient for the requested 5% false-reject target.',fontsize=9,color='#475467')
fig.tight_layout(rect=(0,.12,1,1))
for ext in ('png','svg'):
    fig.savefig(OUT/f'segmentation.{ext}',dpi=180,facecolor='white')
plt.close(fig)
