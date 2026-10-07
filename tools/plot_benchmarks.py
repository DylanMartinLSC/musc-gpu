"""Regenerate comparison figures from accepted JSON; no inference is run."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'benchmarks/bottle83/results.json').read_text())
OUT = ROOT / 'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#ffffff', 'axes.titleweight': 'bold'})
colors = ['#667085', '#147d92']

def save(fig, name, footer):
    fig.text(.5, .02, footer, ha='center', fontsize=9, color='#475467')
    fig.tight_layout(rect=(0, .08, 1, 1))
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=180)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.6))
medians = [data['median_seconds'][key] for key in ('original', 'candidate')]
bars = ax.barh(['Original MuSc', 'MuSc-GPU'], medians, color=colors, height=.55)
ax.invert_yaxis()
ax.set_xlim(0, 46)
ax.set_xlabel('Complete 83-image pool inference (seconds; lower is better)')
ax.set_title('3.84× faster end-to-end inference on bottle83', loc='left', pad=18)
for bar, value in zip(bars, medians):
    ax.text(value + .7, bar.get_y()+bar.get_height()/2, f'{value:.2f} s', va='center', weight='bold')
for i, key in enumerate(('original', 'candidate')):
    ax.scatter([p[key]['seconds'] for p in data['pairs']], [i-.10, i+.10],
               s=24, color='white', edgecolor='#101828', zorder=3)
ax.grid(axis='x', alpha=.18)
ax.set_axisbelow(True)
save(fig, 'latency', 'Bars: median of two alternating passes; dots: each pass. RTX 3090 / Windows / PyTorch 2.7.1+cu118.\nIncludes preprocessing and first-use setup; excludes model loading and metrics.')

fig, ax = plt.subplots(figsize=(8, 4.6))
values = [max(p[key]['peak_allocated_mb'] for p in data['pairs'])/1024 for key in ('original', 'candidate')]
bars = ax.bar(['Original MuSc', 'MuSc-GPU'], values, color=colors, width=.55)
ax.set_ylim(0, max(values)*1.24)
ax.set_ylabel('Peak allocated GPU memory (GiB; lower is better)')
ax.set_title('Acceleration uses more allocated GPU memory', loc='left', pad=18)
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, value+.12, f'{value:.2f} GiB', ha='center', weight='bold')
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
save(fig, 'memory', 'Maximum recorded peak across the same two full-pool passes.\nAllocated tensor memory, not reserved memory or total device usage; bottle83 only.')

fig, ax = plt.subplots(figsize=(8, 4.6))
errors = [max(p[key] for p in data['pairs']) for key in ('original_map_error', 'original_score_error')]
bars = ax.bar(['Anomaly maps', 'Image scores'], errors, color=['#147d92', '#30a48d'], width=.55)
ax.set_yscale('log')
ax.set_ylim(5e-8, 1e-4)
ax.axhline(2e-5, color='#be5a19', linestyle='--', label='Full-image tolerance: 2e-5')
ax.legend(loc='upper right', frameon=False)
ax.set_ylabel('Maximum absolute error vs original (log scale)')
ax.set_title('Output differences remain below the accepted tolerance', loc='left', pad=18)
for bar, value in zip(bars, errors):
    ax.text(bar.get_x()+bar.get_width()/2, value*1.3, f'{value:.3e}', ha='center', weight='bold')
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
save(fig, 'agreement', 'Maximum error across the two full bottle83 passes; lower is better.\nThis measures numerical agreement, not AUROC or other newly evaluated quality metrics.')
print('Wrote three figures in PNG and SVG from accepted bottle83 JSON.')
