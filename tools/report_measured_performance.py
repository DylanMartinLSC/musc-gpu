"""Generate README charts from complete paired measurements, without CUDA."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_results(root):
    current = json.loads((root / 'benchmarks/super_fast/category_results.json').read_text())
    historical = json.loads((root / 'benchmarks/representative/results.json').read_text())
    if current['status'] != 'complete approximate comparison':
        raise ValueError('Complete measured category comparisons are required')
    rows = current['workloads']
    if [(w['category'], w['images']) for w in rows] != [('bottle',83), ('grid',78), ('cable',150)]:
        raise ValueError('Expected complete bottle83, grid78 and cable150 pools')
    for w in rows:
        if len(w['pairs']) != 2 or not all(p['same_inputs'] for p in w['pairs']):
            raise ValueError('Two matched pairs are required')
        if not all(c['pass'] for checks in w['repeat_checks'].values() for c in checks.values()):
            raise ValueError('Repeated predictions must agree')
    return rows, {w['category']: w for w in historical['workloads']}

def main():
    rows, historical = load_results(ROOT)
    x = np.arange(len(rows))
    colors = {'original':'#64748b', 'gpu':'#2563eb', 'super_fast':'#16a34a'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios':[1.2,1]})
    ax = axes[0]
    for offset, backend, label in [(-.26,'original','Original (Oct 12 paired trials)'),
                                    (0,'gpu','Default GPU (Oct 7 prior trials)'),
                                    (.26,'super_fast','Super fast (Oct 12, approximate)')]:
        values = [historical[w['category']]['summary']['median_inference_seconds']['gpu']
                  if backend == 'gpu' else w['summary']['median_inference_seconds'][backend] for w in rows]
        bars = ax.bar(x+offset, values, .24, color=colors[backend], label=label,
                      hatch='//' if backend == 'super_fast' else None)
        ax.bar_label(bars, labels=[f'{v:.2f}s' for v in values], padding=4, fontsize=9)
    ax.set_title('Complete-pool inference latency')
    ax.set_ylabel('Seconds (lower is better)')
    ax.set_ylim(0, max(w['summary']['median_inference_seconds']['original'] for w in rows)*1.22)
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    ax = axes[1]
    for offset, backend, label in [(-.14,'gpu','Default GPU / original (Oct 7)'),
                                  (.14,'super_fast','Super fast / original (Oct 12)')]:
        values = [historical[w['category']]['summary']['speedup'] if backend == 'gpu'
                  else w['summary']['speedup'] for w in rows]
        bars = ax.bar(x+offset, values, .26, color=colors[backend],label=label,
                      hatch='//' if backend == 'super_fast' else None)
        ax.bar_label(bars, labels=[f'{v:.2f}×' for v in values],padding=4,fontsize=10)
    ax.set_title('Speedup against each trial’s original baseline')
    ax.set_ylabel('Ratio of medians (higher is better)')
    ax.set_ylim(0, max(w['summary']['speedup'] for w in rows)*1.28)
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xticks(x,[f"{w['category']}\n({w['images']} images)" for w in rows])
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True,alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Measured inference performance · RTX 3090',fontsize=15)
    failed = ', '.join(w['category'] for w in rows if not w['summary']['all_metrics_within_provisional_1pp'])
    fig.text(.06,.04,'Two alternating fresh-process pairs per category; first-use setup included; model loading and metrics separate.\n'
             f'Super fast changes predictions. Categories failing the provisional 1pp accuracy tolerance: {failed or "none"}. See the full report.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94))
    out = ROOT / 'benchmarks/super_fast'
    for extension in ('png','svg'):
        fig.savefig(out / f'measured_performance.{extension}',dpi=180,
                    metadata={'Date':None} if extension == 'svg' else None)
        if extension == 'svg':
            path=out / 'measured_performance.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    render_backend_accuracy(rows,historical,out)
    render_accuracy_deltas(rows,out)

def render_backend_accuracy(rows,historical,out):
    keys=['image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro']
    labels=['Image\nAUROC','Image\nAP','Image\nF1 max','Pixel\nAUROC','Pixel\nAP','Pixel\nF1 max','Region\nAUPRO']
    x=np.arange(len(keys))
    fig,axes=plt.subplots(3,1,figsize=(13,10.5),sharex=True,sharey=True)
    for ax,w in zip(axes,rows):
        for offset,backend,label,color in [(-.25,'original','Original MuSc (Oct 12)','#64748b'),
                                          (0,'gpu','Default GPU (Oct 7)','#2563eb'),
                                          (.25,'super_fast','Super fast (Oct 12, approximate)','#16a34a')]:
            metrics=({key:statistics.median(p['gpu']['metrics']['values'][key]
                       for p in historical[w['category']]['pairs']) for key in keys}
                     if backend=='gpu' else w['summary']['metrics'][backend])
            values=[100*metrics[key] for key in keys]
            bars=ax.bar(x+offset,values,.23,color=color,label=label,
                        hatch='//' if backend=='super_fast' else None)
            ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,rotation=90,fontsize=8)
        ax.set_title(f"{w['category']} · complete {w['images']}-image pool",loc='left',fontsize=12)
        ax.set_ylim(0,118)
        ax.set_yticks([0,20,40,60,80,100])
        ax.set_ylabel('Accuracy (%)')
        ax.yaxis.grid(True,alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(x,labels)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    fig.suptitle('Measured accuracy · all three backends · higher is better',fontsize=16,y=.995)
    fig.text(.07,.035,'Medians from two repeated trials per backend and pool. Original / super fast: Oct 12; default GPU: Oct 7.\n'
             'Labels rounded to 0.1%; original and default GPU agree closely. F1 maximized over evaluation thresholds; AUPRO at FPR < 0.3.',fontsize=9)
    fig.tight_layout(rect=(0,.095,1,.93))
    for extension in ('png','svg'):
        path=out/f'backend_accuracy.{extension}'
        fig.savefig(path,dpi=180,metadata={'Date':None} if extension=='svg' else None)
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

def render_accuracy_deltas(rows,out):
    keys = ['image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro']
    values = np.array([[w['summary']['metric_deltas_pp'][key] for key in keys] for w in rows])
    fig, ax = plt.subplots(figsize=(11,3.6))
    bound = max(1,float(np.abs(values).max()))
    ax.imshow(values,cmap='RdYlGn',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(7),['Image AUROC','Image AP','Image F1','Pixel AUROC','Pixel AP','Pixel F1','AUPRO'])
    ax.set_yticks(range(3),[w['category'] for w in rows])
    for i in range(3):
        for j in range(7):
            ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=11)
    ax.set_title('Super fast accuracy change versus paired original (percentage points)')
    fig.text(.08,.04,'Positive means higher accuracy; F1 maximized over evaluation thresholds. Provisional tolerance: no metric below −1pp.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1))
    for extension in ('png','svg'):
        fig.savefig(out / f'category_accuracy.{extension}',dpi=180,
                    metadata={'Date':None} if extension == 'svg' else None)
        if extension == 'svg':
            path=out / 'category_accuracy.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

if __name__ == '__main__':
    main()
