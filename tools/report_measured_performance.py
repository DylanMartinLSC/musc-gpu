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
    current = json.loads((root / 'benchmarks/rank_deferred/super_fast_results.json').read_text())
    # Default GPU values come from the latest original/default receipt where its paired
    # category passed (bottle, grid); cable keeps the previous representative receipt.
    latest = json.loads((root / 'benchmarks/rank_deferred/results.json').read_text())
    historical = json.loads((root / 'benchmarks/representative/results.json').read_text())
    passed = {w['category']: w for w in latest['workloads'] if w['summary']['status'] == 'pass'}
    historical['workloads'] = [passed.get(w['category'], w) for w in historical['workloads']]
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
    for offset, backend, label in [(-.26,'original','Original (paired trials)'),
                                    (0,'gpu','Default GPU (separate paired benchmark; cable: previous scoring)'),
                                    (.26,'super_fast','Super fast (approximate)')]:
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
    for offset, backend, label in [(-.14,'gpu','Default GPU / original'),
                                  (.14,'super_fast','Super fast / original')]:
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
    render_profile_context(out)

def render_backend_accuracy(rows,historical,out):
    keys=['image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro']
    labels=['Image\nAUROC','Image\nAP','Image\nF1 max','Pixel\nAUROC','Pixel\nAP','Pixel\nF1 max','Region\nAUPRO']
    x=np.arange(len(keys))
    fig,axes=plt.subplots(3,1,figsize=(13,10.5),sharex=True,sharey=True)
    for ax,w in zip(axes,rows):
        for offset,backend,label,color in [(-.25,'original','Original MuSc','#64748b'),
                                          (0,'gpu','Default GPU','#2563eb'),
                                          (.25,'super_fast','Super fast (approximate)','#16a34a')]:
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
    fig.text(.07,.035,'Medians from two repeated trials per backend and pool. Original / super fast: paired comparison; default GPU: prior benchmark.\n'
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

def render_profile_context(out):
    original=json.loads((out/'historical_context.json').read_text())['original']
    variants=json.loads((out/'tradeoff_summary.json').read_text())['variants']
    default=next(v for v in variants if v['id']=='default')
    fast=next(v for v in variants if v['id']=='grid28_r3_layer5')
    labels=['Original','GPU default','Super fast']
    colors=['#64748b','#2563eb','#0d9488']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,values,title,unit in [(axes[0],[v['median_seconds'] for v in [original,default,fast]],'Full-pool inference','seconds'),
                                (axes[1],[v['peak_allocated_mib']/1024 for v in [original,default,fast]],'Peak allocated GPU memory','GiB')]:
        bars=ax.bar(labels,values,color=colors,width=.65)
        ax.set_title(title,fontsize=12)
        ax.set_ylabel(unit)
        ax.set_ylim(0,max(values)*1.2)
        ax.bar_label(bars,fmt='%.2f',padding=4)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    x=np.arange(2)
    for index,v in enumerate([original,default,fast]):
        bars=axes[2].bar(x+(index-1)*.24,[100*v['metrics'][k] for k in ['pixel_ap','pixel_aupro']],
                         width=.24,color=colors[index],label=labels[index])
        axes[2].bar_label(bars,fmt='%.2f',padding=4,fontsize=8)
    axes[2].set_xticks(x,['Pixel AP','Region AUPRO'])
    axes[2].set_ylim(0,110)
    axes[2].set_title('Full-resolution accuracy',fontsize=12)
    axes[2].set_ylabel('%')
    axes[2].legend(loc='lower right',fontsize=8)
    axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle('Bottle83 on RTX3090: historical comparison',fontsize=16,y=.98)
    fig.text(.5,.035,'Original and resident-model profile experiments have different protocols; these timings are historical context.',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.08,1,.92))
    for extension in ['png','svg']:
        path=out/f'comparison.{extension}'
        fig.savefig(path,dpi=180,metadata={'Date':None} if extension=='svg' else None)
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

if __name__ == '__main__':
    main()
