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
        values = [f'{times[b]:.3f}' if passed else '—' for b in ('original','gpu')]
        lines.append(f'| {w["id"]} | {w["images"]} | {scope} | {summary["status"]} | '
                     + ' | '.join(values) + f' | {summary["speedup"]:.3f}× | {summary["minimum_paired_speedup"]:.3f}× |' if passed else
                     f'| {w["id"]} | {w["images"]} | {scope} | {summary["status"]} | — | — | — | — |')
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
                       f'**{aggregate["speedup"]:.3f}×**.', '', aggregate['note'] + '.'])
    lines.extend(['', '## Quality and resource measurements', '',
                  'Metric values below are medians across the measured trials (0–1 scale). '
                  'Undefined/not requested metrics remain blank. F1 is maximized over evaluation thresholds; '
                  'this does not establish a production threshold.', '',
                  '| Pool / backend | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Pixel AUPRO | Peak allocated GiB | Model load (s) | Metrics (s) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in csv_rows:
        def fmt(key, decimals=4):
            return f'{row[key]:.{decimals}f}' if row[key] is not None else '—'
        lines.append(f'| {row["pool"]} / {row["backend"]} | {fmt("image_auroc")} | {fmt("image_ap")} | '
            f'{fmt("pixel_auroc")} | {fmt("pixel_ap")} | {fmt("pixel_aupro")} | '
            f'{row["peak_allocated_mib"]/1024:.2f}' + f' | {fmt("median_model_load_seconds",2)} | {fmt("median_metric_seconds",2)} |'
            if row['peak_allocated_mib'] is not None else f'| {row["pool"]} / {row["backend"]} | — | — | — | — | — | — | — | — |')
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
            fig.savefig(output/f'{name}.{suffix}', dpi=180)
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
    fig, ax = plt.subplots(figsize=(max(8,len(x)*1.8),5))
    values = [w['summary']['speedup'] for w in workloads]
    bars = ax.bar(x, values, color=colors[1], width=.5)
    ax.bar_label(bars, fmt='%.2f×', padding=5)
    for index,w in enumerate(workloads):
        ratios = w['summary']['paired_speedups']
        ax.scatter([index]*len(ratios), ratios, color='white',edgecolor='#101828',zorder=3)
    ax.set_xticks(x,labels)
    ax.set_ylabel('Original / musc-gpu (higher is better)')
    ax.set_title('Speedup with individual paired measurements')
    ax.axhline(1,color='#667085',linestyle='--')
    largest = max([*values, *(ratio for w in workloads for ratio in w['summary']['paired_speedups'])])
    ax.set_ylim(0, largest*1.25)
    save(fig,'speedup','Bars: ratio of median latency. Dots: individual paired speedups.\nTwo pairs describe repeatability; they do not establish a confidence interval.')
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
    fig, axes = plt.subplots(2,2,figsize=(max(10,len(x)*2),8))
    for ax,key,title in zip(axes.flat,('image_auroc','image_ap','pixel_auroc','pixel_ap'),
                           ('Image classification AUROC','Image average precision',
                            'Pixel segmentation AUROC','Pixel average precision')):
        for i,b in enumerate(('original','gpu')):
            values = [lookup[w['id'],b][key] for w in workloads]
            ax.scatter(x+(i-.5)*.16,[v if v is not None else np.nan for v in values],
                       color=colors[i],s=55,label='Original' if i==0 else 'musc-gpu',zorder=3)
        ax.set_xticks(x,labels,fontsize=9)
        ax.set_ylim(0,1.04)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,loc='lower right')
    save(fig,'quality','Evaluation after inference, using the same image labels and resized masks.\nCompare marker heights; exact values and metric deltas are in the report.')
    fig,ax=plt.subplots(figsize=(max(8,len(x)*1.8),5))
    for i,key in enumerate(('maps','scores')):
        values=[max(p['agreement'][key]['max_absolute_error'] for p in w['pairs']) for w in workloads]
        ax.scatter(x+(i-.5)*.16,[max(v,1e-12) for v in values],s=55,label='Maps' if key=='maps' else 'Scores')
    ax.set_yscale('log')
    ax.set_ylim(1e-12,1e-4)
    ax.axhline(2e-5,color='#be5a19',linestyle='--',label='Tolerance 2e-5')
    ax.set_xticks(x,labels)
    ax.set_ylabel('Maximum absolute error vs original')
    ax.set_title('Numerical agreement across pools')
    ax.legend(frameon=False)
    save(fig,'agreement','Worst error across measured paired trials; exact zero is plotted at the 1e-12 display floor.\nNonfinite or out-of-tolerance outputs reject speed claims.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.report,args.output)
