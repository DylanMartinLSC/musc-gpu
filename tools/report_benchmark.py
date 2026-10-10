"""Generate tables and figures from benchmark.py JSON without running inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

def render(source, output):
    data = json.loads(source.read_text())
    if data.get('schema_version') != 1 or data.get('name') != 'musc-gpu':
        raise ValueError('Expected a musc-gpu benchmark.py schema_version=1 report')
    output.mkdir(parents=True, exist_ok=True)
    successful = [w for w in data['workloads'] if w.get('summary', {}).get('status') == 'pass']
    source_link = os.path.relpath(source.resolve(), output.resolve()).replace('\\','/')
    lines = ['# musc-gpu benchmark', '', f'Run status: **{data["status"]}**.', '',
        f'[Authoritative raw JSON]({source_link})', '',
        data['timing_scope'] + '.', '',
        'Fresh processes include first-use compilation on every trial. No discarded image warmup. '
        'Operating-system file cache is not flushed. Metrics and model loading are separate.', '',
        '| Pool | Images | Scope | Status | Original (s) | musc-gpu (s) | Speedup | Worst pair |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    csv_rows = []
    for w in data['workloads']:
        summary = w.get('summary', {'status': 'incomplete'})
        passed = summary['status'] == 'pass'
        scope = 'complete category' if w['complete_pool'] else 'explicit subset'
        times = summary.get('median_inference_seconds', {})
        values = [f'{times[b]:.3f}' if passed else 'â€”' for b in ('original','gpu')]
        lines.append(f'| {w["id"]} | {w["images"]} | {scope} | {summary["status"]} | '
                     + ' | '.join(values) + f' | {summary["speedup"]:.3f}Ã— | {summary["minimum_paired_speedup"]:.3f}Ã— |' if passed else
                     f'| {w["id"]} | {w["images"]} | {scope} | {summary["status"]} | â€” | â€” | â€” | â€” |')
        for backend in ('original','gpu'):
            trials = [p[backend] for p in w['pairs']] if passed else []
            csv_rows.append({'pool': w['id'], 'images': w['images'], 'complete_pool': w['complete_pool'],
                'status': summary['status'], 'backend': backend,
                'median_inference_seconds': times.get(backend),
                'speedup': summary.get('speedup'),
                'median_model_load_seconds': statistics.median(t['model_load_seconds'] for t in trials) if trials else None,
                'median_metric_seconds': statistics.median(t['metric_evaluation_seconds'] for t in trials) if trials else None,
                'peak_allocated_mib': max(t['peak_allocated_mib'] for t in trials) if trials else None,
                'peak_reserved_mib': max(t['peak_reserved_mib'] for t in trials) if trials else None,
                **{key: statistics.median(t['metrics']['values'][key] for t in trials)
                    if trials and all(t['metrics']['values'][key] is not None for t in trials) else None
                   for key in ('image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro')}})
    if 'validated_full_category_aggregate' in data:
        aggregate = data['validated_full_category_aggregate']
        lines.extend(['', f'Aggregate speedup for validated full categories ({", ".join(aggregate["categories"])}): '
                       f'**{aggregate["speedup"]:.3f}Ã—**.', '', aggregate['note'] + '.'])
    lines.extend(['', '## Quality and resource measurements', '',
                  'Metric values below are medians across the measured trials (0â€“1 scale). '
                  'Undefined/not requested metrics remain blank. F1 is maximized over evaluation thresholds; '
                  'this does not establish a production threshold.', '',
                  '| Pool / backend | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Pixel AUPRO | Peak allocated GiB | Model load (s) | Metrics (s) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in csv_rows:
        def fmt(key, decimals=4):
            return f'{row[key]:.{decimals}f}' if row[key] is not None else 'â€”'
        lines.append(f'| {row["pool"]} / {row["backend"]} | {fmt("image_auroc")} | {fmt("image_ap")} | '
            f'{fmt("pixel_auroc")} | {fmt("pixel_ap")} | {fmt("pixel_aupro")} | '
            f'{row["peak_allocated_mib"]/1024:.2f}' + f' | {fmt("median_model_load_seconds",2)} | {fmt("median_metric_seconds",2)} |'
            if row['peak_allocated_mib'] is not None else f'| {row["pool"]} / {row["backend"]} | â€” | â€” | â€” | â€” | â€” | â€” | â€” | â€” |')
    lines.extend(['', '## Numerical agreement', '',
                  '| Pool | Max map error | Max score error | Tolerance |', '|---|---:|---:|---:|'])
    for w in successful:
        errors = {key: max(p['agreement'][key]['max_absolute_error'] for p in w['pairs']) for key in ('maps','scores')}
        lines.append(f'| {w["id"]} | {errors["maps"]:.6e} | {errors["scores"]:.6e} | 2e-5 |')
    lines.extend(['', 'No speed claim is made for failed, incomplete or skipped pools. '
                  'A subset changes the reference pool and cannot establish full-category quality or the bottle83 objective.', '',
                  'Full per-pass timings, dispatch diagnostics, quality values, paths, hashes and errors are in the input JSON. '
                  'Raw prediction arrays and logs remain in the original benchmark run directory.', ''])
    fields = ['pool','images','complete_pool','status','backend','median_inference_seconds',
              'speedup','median_model_load_seconds','median_metric_seconds','peak_allocated_mib',
              'peak_reserved_mib','image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap',
              'pixel_f1_max','pixel_aupro']
    if data.get('error'):
        lines.extend(['', '## Run failure', '', '```text', data['error'], '```', ''])
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    if successful:
        plot(successful, csv_rows, output)
        lines.extend(['![Pool latency](latency.png)', '', '![Paired speedup](speedup.png)', '',
                      '![Peak allocated memory](memory.png)', '', '![Detection and segmentation quality](quality.png)', '',
                      '![Numerical agreement](agreement.png)', ''])
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {output}/REPORT.md and summary.csv' + (' plus measured comparison figures.' if successful else '; no successful workload to plot.'))

def plot(workloads, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(len(workloads))
    labels = [w['id'] + f'\nN={w["images"]}, ' + ('complete' if w['complete_pool'] else 'subset') for w in workloads]
    colors = ['#667085','#147d92']
    def save(fig, name, footer):
        fig.text(.5, .015, footer, ha='center', fontsize=9, color='#475467')
        fig.tight_layout(rect=(0,.09,1,1))
        for suffix in ('png','svg'):
            target = output/f'{name}.{suffix}'
            fig.savefig(target, dpi=180)
            if suffix == 'svg':
                target.write_bytes(('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(x)*1.8),5))
    for i,b in enumerate(('original','gpu')):
        values = [w['summary']['median_inference_seconds'][b] for w in workloads]
        bars = ax.bar(x+(i-.5)*.36, values, width=.36, color=colors[i], label='Original' if i==0 else 'musc-gpu')
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Complete inference (seconds; lower is better)')
    ax.set_title('Measured pool inference latency')
    ax.set_ylim(0, ax.get_ylim()[1]*1.13)
    ax.legend(frameon=False)
    save(fig,'latency','Median of alternating fresh-process trials; first-use setup included.\nModel loading and metric evaluation excluded. Complete pools and explicit subsets are labeled.')
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(x)*1.25+1.5)))
    remaining = [100 / w['summary']['speedup'] for w in workloads]
    annotation_x = max(100, max(remaining)) + 4
    ax.barh(x, [100]*len(x), height=.55, color='#eaecf0', label='Original runtime = 100%')
    bars = ax.barh(x, remaining, height=.55, color=colors[1], label='musc-gpu runtime')
    for index, (w, value) in enumerate(zip(workloads, remaining)):
        times = w['summary']['median_inference_seconds']
        ax.text(value/2, index, f'{value:.1f}%', ha='center', va='center',
                color='white', weight='bold', fontsize=12)
        ax.text(value+2, index, f"{times['original']:.2f} s → {times['gpu']:.2f} s",
                va='center', color='#344054', fontsize=11)
        ax.text(annotation_x, index-.09, f"{w['summary']['speedup']:.2f}× faster",
                va='center', weight='bold', color=colors[1], fontsize=13)
        ratios = w['summary']['paired_speedups']
        ax.text(annotation_x, index+.20, f'Pairs: {min(ratios):.3f}–{max(ratios):.3f}×',
                va='center', fontsize=9, color='#667085')
    ax.set_yticks(x, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 139)
    ax.set_xticks([0,25,50,75,100], ['0%','25%','50%','75%','100%'])
    ax.set_xlabel('Share of original inference time · lower is better')
    ax.set_title('The same pool, in about a quarter of the time', loc='left', pad=20,
                 weight='bold')
    # Keep the headline valid for arbitrary successful reports, including slower runs.
    if not all(3.5 <= w['summary']['speedup'] <= 4.5 for w in workloads):
        ax.set_title('Inference time relative to original MuSc', loc='left', pad=20)
    ax.set_xlim(0, max(139, max(remaining)*1.35))
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig,'speedup','Bars and seconds: median latency; each original pool is normalized to 100%.\nPaired ranges show observed repeatability, not confidence intervals. Model loading and metrics excluded.')
    lookup = {(r['pool'],r['backend']):r for r in rows}
    fig, ax = plt.subplots(figsize=(max(8,len(x)*1.8),5))
    for i,b in enumerate(('original','gpu')):
        values = [lookup[w['id'],b]['peak_allocated_mib']/1024 for w in workloads]
        bars = ax.bar(x+(i-.5)*.36,values,width=.36,color=colors[i],label='Original' if i==0 else 'musc-gpu')
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,labels)
    ax.set_ylabel('Peak allocated GPU memory (GiB)')
    ax.set_title('Measured memory cost of acceleration')
    ax.set_ylim(0,ax.get_ylim()[1]*1.13)
    ax.legend(frameon=False)
    save(fig,'memory','Maximum recorded allocation across measured trials.\nReserved memory is reported separately in JSON/CSV; this is not total device usage.')
    metrics = [('image_auroc','Image AUROC'), ('image_ap','Image AP'),
               ('image_f1_max','Image F1'), ('pixel_auroc','Pixel AUROC'),
               ('pixel_ap','Pixel AP'), ('pixel_f1_max','Pixel F1'),
               ('pixel_aupro','Pixel AUPRO')]
    # Numeric scorecards reveal tiny differences that overlapping markers conceal.
    fig, axes = plt.subplots(len(workloads), 1,
                             figsize=(10, 3.15*len(workloads)+1), squeeze=False)
    for ax,w in zip(axes.flat,workloads):
        ax.axis('off')
        scope = 'complete pool' if w['complete_pool'] else 'explicit subset'
        ax.set_title(f"{w['id']}  ·  {w['images']} images  ·  {scope}",
                     loc='left', fontsize=13, weight='bold', pad=10)
        cells=[]
        for key,title in metrics:
            original, gpu = (lookup[w['id'],b][key] for b in ('original','gpu'))
            delta = None if original is None or gpu is None else (gpu-original)*100
            cells.append([title,
                          '—' if original is None else f'{original*100:.4f}',
                          '—' if gpu is None else f'{gpu*100:.4f}',
                          '—' if delta is None else ('0 (exact)' if delta == 0 else f'{delta:+.2e}')])
        table=ax.table(cellText=cells,
                       colLabels=['Metric (0–100)', 'Original', 'musc-gpu', 'Change (points)'],
                       colWidths=[.31,.21,.21,.27], cellLoc='center', bbox=[0,0,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(2)
            cell.set_facecolor('#f2f4f7' if row%2 else '#ffffff')
            if row==0:
                cell.set_facecolor('#1d2939')
                cell.set_text_props(color='white',weight='bold')
            elif col==2:
                cell.set_facecolor('#e6f4f3')
                cell.set_text_props(color='#086b70',weight='bold')
            elif col==0:
                cell.set_text_props(ha='left')
    fig.suptitle('Accuracy, with the differences made explicit', x=.04, ha='left',
                 fontsize=18, weight='bold', y=.995)
    save(fig,'quality','Medians across measured trials; higher metric values are better. Change = musc-gpu − original.\nValues shown on a 0–100 scale; changes are percentage points, computed before rounding.\nF1 uses the best evaluation threshold; — means unavailable. Tiny nonzero changes are retained.')
    fig,ax=plt.subplots(figsize=(10,max(4.8,len(x)*1.1+1.5)))
    for i,key in enumerate(('maps','scores')):
        values=[max(p['agreement'][key]['max_absolute_error'] for p in w['pairs']) for w in workloads]
        positions=x+(i-.5)*.30
        bars=ax.barh(positions,[v/2e-5*100 for v in values],height=.26,
                     color=colors[i],label='Anomaly maps' if key=='maps' else 'Image scores')
        for bar,value in zip(bars,values):
            ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,
                    f'{value:.2e}  ({value/2e-5*100:.2f}% of limit)',va='center',fontsize=10)
    ax.axvline(100,color='#be5a19',linestyle='--',label='Agreement limit: 2e-5')
    ax.set_xlim(0,140)
    ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
    ax.set_yticks(x,labels)
    ax.invert_yaxis()
    ax.set_xlabel('Share of the absolute-error limit · lower is better')
    ax.set_title('Prediction differences stay below the agreement limit',loc='left',pad=20,weight='bold')
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    save(fig,'agreement','Worst absolute error across measured pairs; labels include the unscaled error.\nThis is prediction agreement, separate from detection quality. Out-of-tolerance outputs reject speed claims.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.report,args.output)
