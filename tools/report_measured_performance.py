"""Render the README comparison from preserved measured JSON; no inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    historical = json.loads((ROOT / 'benchmarks/representative/results.json').read_text())
    study = json.loads((ROOT / 'benchmarks/super_fast/tradeoff_summary.json').read_text())
    fast = next(v for v in study['variants'] if v['id'] == 'grid28_r3_layer5')
    pools = historical['workloads']
    x = np.arange(len(pools))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, backend, label, color in [(-.26, 'original', 'Original MuSc (Oct 7)', '#64748b'),
                                         (0, 'gpu', 'musc-gpu (Oct 7)', '#2563eb')]:
        values = [w['summary']['median_inference_seconds'][backend] for w in pools]
        bars = ax.bar(x + offset, values, .24, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.2f}s' for v in values], padding=4, fontsize=10)
    bottle = next(i for i, w in enumerate(pools) if w['category'] == 'bottle')
    bar = ax.bar([bottle + .26], [fast['median_seconds']], .24,
                 label='super_fast, approximate (Oct 12)', color='#16a34a', hatch='//')
    ax.bar_label(bar, labels=[f"{fast['median_seconds']:.2f}s"], padding=4, fontsize=10)
    for i, w in enumerate(pools):
        if w['category'] != 'bottle':
            ax.text(i + .26, 2, 'Not tested', rotation=90, ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, [f"{w['category']} ({w['images']} images)" for w in pools])
    ax.set_ylabel('Complete-pool inference time (seconds; lower is better)')
    ax.set_title('Measured inference performance · RTX 3090')
    ax.set_ylim(0, max(w['summary']['median_inference_seconds']['original'] for w in pools) * 1.23)
    ax.legend(loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.08, .065, 'Oct 7: fresh-process original/GPU pairs. Oct 12: resident-model profile sweeps.\n'
             'Super fast: 2.63–2.70× vs default GPU in its own study; fresh original pairing pending.', fontsize=10)
    fig.tight_layout(rect=(0, .14, 1, 1))
    out = ROOT / 'benchmarks/super_fast'
    fig.savefig(out / 'measured_performance.png', dpi=180)
    fig.savefig(out / 'measured_performance.svg')
    plt.close(fig)
    manifest_path = ROOT / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    for relative in ['README.md', 'tools/report_measured_performance.py',
                     'benchmarks/super_fast/measured_performance.png',
                     'benchmarks/super_fast/measured_performance.svg']:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if relative in manifest:
            manifest[relative]['packaged_sha256'] = digest
        else:
            manifest[relative] = {'source': relative, 'source_sha256': digest, 'packaged_sha256': digest}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')

if __name__ == '__main__':
    main()
