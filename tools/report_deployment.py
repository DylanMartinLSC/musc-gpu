"""Render measured resident-model delivery throughput; never extrapolate a soak."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    if data['status'] != 'pass' or len(data['pairs']) != data['requested_pairs'] or not all(p['agreement']['pass'] for p in data['pairs']):
        raise ValueError('Refusing performance chart for incomplete or inaccurate benchmark')
    args.output.mkdir(parents=True,exist_ok=True)
    summary = data['summary']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,(ax,bx) = plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1.7,1]})
    colors = {'original':'#64748b','gpu':'#147d92'}
    names = {'original':'Original MuSc','gpu':'musc-gpu'}
    count = data['plan']['images']
    for backend in ('original','gpu'):
        for index,pair in enumerate(data['pairs']):
            rows = pair['sessions'][backend]['requests']
            x = [0]+[r['completed_at_seconds'] for r in rows]
            y = [0]+[count*(i+1) for i in range(len(rows))]
            ax.step(x,y,where='post',color=colors[backend],linestyle='-' if index==0 else '--',
                    linewidth=2.2,label=f'{names[backend]} · session {index+1}')
            ax.scatter(x[1:],y[1:],color=colors[backend],s=20,zorder=3)
    ax.set_title('Measured result deliveries',loc='left',weight='bold',pad=12)
    ax.set_xlabel('Seconds after first request starts (model loaded)')
    ax.set_ylabel('Image-processing completions')
    ax.set_yticks([count*i for i in range(data['requests_per_session']+1)])
    ax.set_ylim(0,count*data['requests_per_session']*1.12)
    ax.legend(loc='lower right',fontsize=9,frameon=True)
    rates = [summary[b]['images_per_second'] for b in ('original','gpu')]
    bars = bx.bar([names[b] for b in ('original','gpu')],rates,color=list(colors.values()),width=.58)
    bx.bar_label(bars,labels=[f'{r:.2f}' for r in rates],padding=7,weight='bold')
    for x,backend in enumerate(('original','gpu')):
        sessions = [p['sessions'][backend]['summary']['images_per_second'] for p in data['pairs']]
        bx.scatter([x]*len(sessions),sessions,color='white',edgecolor='#0f172a',s=30,zorder=4)
    bx.set_ylim(0,max(rates)*1.3)
    bx.set_ylabel('Delivered images / second')
    bx.set_title(f"{summary['delivered_throughput_ratio']:.2f}× measured throughput",loc='left',weight='bold',pad=12)
    for axis in (ax,bx):
        axis.grid(axis='y',alpha=.18)
        axis.set_axisbelow(True)
    env = data['pairs'][0]['sessions']['gpu']['environment']
    fig.suptitle(f"Batch inspection replay · {env['gpu'].replace('NVIDIA GeForce ','')}",fontsize=16,weight='bold',y=.98)
    fig.text(.065,.095,f"{data['requested_pairs']} alternating sessions/backend × {data['requests_per_session']} requests × {count} images. First request retained; maps and scores written locally.",fontsize=10)
    fig.text(.065,.06,'Same full pool reprocessed each request; no feature cache. Model loading excluded; initial kernel setup included.',fontsize=10)
    fig.text(.065,.025,'Finite workstation replay, not unique production parts, larger reference pools, single-image latency, or a factory throughput guarantee.',fontsize=9,color='#475467')
    fig.tight_layout(rect=(0,.14,1,.93),w_pad=3)
    for suffix in ('png','svg'):
        fig.savefig(args.output/f'deployment.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    lines = ['# Resident-model batch inspection replay','',
             '**Measured workstation replay, not a deployed factory trial.**', '',
             f"Each backend processed {summary['gpu']['completed_images']} image instances across {data['requested_pairs']} sessions. Every request recomputed the same complete {count}-image pool; these are not distinct new parts.", '',
             '| Backend | Delivered images/s | Session seconds | First-request seconds | Later-request seconds | Peak allocated GiB |',
             '|---|---:|---|---|---|---:|']
    for backend in ('original','gpu'):
        trials = [p['sessions'][backend] for p in data['pairs']]
        first = [t['requests'][0]['service_seconds'] for t in trials]
        later = [r['service_seconds'] for t in trials for r in t['requests'][1:]]
        elapsed = summary[backend]['session_elapsed_seconds']
        lines.append(f"| {names[backend]} | {summary[backend]['images_per_second']:.3f} | {', '.join(f'{v:.2f}' for v in elapsed)} | {min(first):.2f}–{max(first):.2f} | {min(later):.2f}–{max(later):.2f} | {summary[backend]['peak_allocated_mib']/1024:.2f} |")
    lines += ['',f"Delivered-throughput ratio: **{summary['delivered_throughput_ratio']:.3f}×**. Per-session paired speedups: " + ', '.join(f"{p['delivered_speedup']:.3f}×" for p in data['pairs'])+'.',
              '', '![Measured delivery timeline and throughput](deployment.png)', '',
              'Throughput divides actual completed images by total measured session time, including inter-request overhead. No hourly extrapolation, selected warmup discard, or percentile claim from this small sample.', '',
              '## Setup outside the resident-service clock','',
              '| Session | Imports/CUDA setup (s) | Model load (s) | Worker ready to last result (s) |',
              '|---|---:|---:|---:|']
    for pair in data['pairs']:
        for backend,t in pair['sessions'].items():
            completion = t['model_ready_seconds_since_worker_ready']+t['summary']['elapsed_seconds']
            lines.append(f"| Pair {pair['pair']} {names[backend]} | {t['imports_and_cuda_setup_seconds']:.2f} | {t['model_load_seconds']:.2f} | {completion:.2f} |")
    max_error = max(r['maps']['max_absolute_error'] for p in data['pairs'] for r in p['agreement']['requests'])
    lines += ['', '## Correctness and scope','',
              f'Every request passed full-map and score agreement at 2e-5; maximum cross-backend map error was {max_error:.8g}. Repeated outputs were also checked against the first request in their session.', '',
              data['timing_scope'], '',
              'Input image content and runtime sources are hashed. Visual-forward and decode counts verify that every request processes every image. No defect annotations are read by the request adapter; zero placeholders satisfy unused upstream bookkeeping.', '',
              'Runtime code is unchanged. The benchmark adapter uses the existing CLIP preprocessing and pipeline with one shared pool and batch size 4. Removing real ground-truth loading makes this a distinct timing protocol from the published fresh-process benchmark.', '',
              '## Limits',''] + ['- '+s for s in data['limitations']]
    lines += ['', 'A larger unique reference pool needs its own memory and accuracy measurement. Pool fill time, camera transfer, network delivery and PLC deadlines must be added for an actual production line. Two short sessions do not establish long-run stability or tail latency.', '',
              'Authoritative evidence: [results.json](results.json).']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
