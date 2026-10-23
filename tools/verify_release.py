"""CPU-only release integrity and backend-selection checks."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
failures = []
for relative, entry in manifest.items():
    path = ROOT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['packaged_sha256']:
        failures.append(f'Source hash mismatch: {relative}')
# The scoring module was replaced by trimmed-rank deferred scoring. Earlier receipts
# stay immutable and may pin its previous packaged hash only through the manifest
# link; the new complete-pool receipt below must pin the current file.
SCORING = 'models/modules/_MSM_gpu.py'
RANK_DEFERRED = json.loads((ROOT / 'benchmarks/rank_deferred/results.json').read_text())
RANK_SOURCES = {k.replace('\\', '/'): v for k, v in RANK_DEFERRED['source_sha256'].items()}

def matches_or_previous_scoring(relative, expected):
    normalized = relative.replace('\\', '/')
    actual = hashlib.sha256((ROOT / normalized).read_bytes()).hexdigest()
    if actual == expected:
        return True
    return (normalized == SCORING and manifest[SCORING].get('previous_packaged_sha256') == expected
            and RANK_SOURCES.get(SCORING) == actual)

for path in ROOT.rglob('*.py'):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    if any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('experiments')
           or isinstance(node, ast.Import) and any(alias.name.startswith('experiments') for alias in node.names)
           for node in ast.walk(tree)):
        failures.append(f'External experiment dependency: {path.relative_to(ROOT)}')
spec = importlib.util.spec_from_file_location('backend', ROOT / 'models/modules/_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
assert backend.select_backend('original', 'cuda', 'nt') == ('original', None)
assert backend.select_backend('gpu', 'cuda', 'nt') == ('gpu', None)
assert backend.select_backend('super_fast', 'cuda', 'nt') == ('super_fast', None)
assert backend.select_backend('super_fast', 'cpu', 'nt')[0] == 'original'
assert backend.select_backend('super_fast', 'cuda', 'posix')[0] == 'original'
assert backend.select_backend('gpu', 'cpu', 'nt')[0] == 'original'
assert backend.select_backend('gpu', 'cuda', 'posix')[0] == 'original'
try:
    backend.select_backend('invalid', 'cuda', 'nt')
    failures.append('Invalid backend accepted')
except ValueError:
    pass
encoder = (ROOT / 'models/modules/_encoder_gpu.py').read_text()
assert 'encode_image.new_aggregation = new_aggregation' in encoder
for relative, entry in manifest.items():
    if 'normalized_ast_sha256' in entry:
        text = (ROOT / relative).read_text().replace('models.modules._cuda_runtime', 'experiments.research.cuda_runtime')
        actual = hashlib.sha256(ast.dump(ast.parse(text), include_attributes=False).encode()).hexdigest()
        assert actual == entry['normalized_ast_sha256'], relative
data = json.loads((ROOT / 'benchmarks/bottle83/results.json').read_text())
assert data['images'] == 83 and len(data['pairs']) == 2
assert all(p['original_map_error'] <= 2e-5 and p['original_score_error'] <= 2e-5 for p in data['pairs'])
for name in ('latency', 'memory', 'agreement'):
    for suffix in ('png', 'svg'):
        assert (ROOT / 'docs/figures' / f'{name}.{suffix}').is_file()
representative = json.loads((ROOT/'benchmarks/representative/results.json').read_text())
integration = json.loads((ROOT/'benchmarks/super_fast/integration.json').read_text())
assert integration['status'] == 'pass' and len(integration['image_paths']) == 83
assert [r['backend'] for r in integration['runs']] == ['gpu','super_fast','super_fast']
assert all(r['pass'] and all(c['max_abs'] <= 2e-5 for c in r['agreement'].values())
           for r in integration['runs'])
assert integration['runs'][-1]['repeat_map_max_abs'] == 0
assert integration['runs'][-1]['repeat_score_max_abs'] == 0
for relative, expected in integration['source_sha256'].items():
    assert matches_or_previous_scoring(relative, expected), relative
# Bottle and grid are paired against original. Cable's original worker exceeded the
# capped experiment job, so cable is covered by a GPU-only previous/new scoring check.
assert [(w['category'], w['images']) for w in RANK_DEFERRED['workloads']] == [('bottle',83),('grid',78),('cable',150)]
assert RANK_DEFERRED['workloads'][2]['summary']['status'] == 'incomplete_or_failed'
cable = json.loads((ROOT / 'benchmarks/rank_deferred/cable_gpu_check.json').read_text())
assert cable['status'] == 'pass' and cable['images'] == 150 and cable['complete_pool'] and len(cable['image_paths']) == 150
assert len(cable['pairs']) == 2 and [p['order'] for p in cable['pairs']] == [['previous','new'],['new','previous']]
assert all(p['maps_bitwise'] and p['scores_bitwise'] and p['labels_equal'] for p in cable['pairs'])
assert cable['source_sha256']['previous_scoring'] == manifest[SCORING]['previous_packaged_sha256']
assert cable['source_sha256'][SCORING] == hashlib.sha256((ROOT / SCORING).read_bytes()).hexdigest()
rank_fast = json.loads((ROOT / 'benchmarks/rank_deferred/super_fast_results.json').read_text())
previous_fast = json.loads((ROOT / 'benchmarks/super_fast/category_results.json').read_text())
assert rank_fast['status'] == 'complete approximate comparison'
assert [(w['category'], w['images']) for w in rank_fast['workloads']] == [('bottle',83),('grid',78),('cable',150)]
for workload, previous in zip(rank_fast['workloads'], previous_fast['workloads']):
    assert workload['complete_pool'] and len(workload['pairs']) == 2
    assert [p['order'] for p in workload['pairs']] == [['original','super_fast'],['super_fast','original']]
    assert all(c['pass'] for checks in workload['repeat_checks'].values() for c in checks.values())
    # Rank-deferred scoring returns identical minima: super fast metrics are unchanged.
    assert workload['summary']['metrics'] == previous['summary']['metrics'], workload['category']
    for pair in workload['pairs']:
        assert pair['same_inputs']
        for name in ('original', 'super_fast'):
            trial = pair[name]
            assert trial['status'] == 'pass' and trial['images'] == workload['images']
            for relative, expected in trial['source_sha256'].items():
                if relative.replace('\\','/').startswith(('models/','datasets/','utils/','configs/')):
                    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
for workload in RANK_DEFERRED['workloads'][:2]:
    assert workload['complete_pool'] and len(workload['pairs']) == 2
    assert workload['summary']['status'] == 'pass'
    for pair in workload['pairs']:
        assert pair['agreement']['pass']
        assert pair['agreement']['maps']['max_absolute_error'] <= 2e-5
        assert pair['agreement']['scores']['max_absolute_error'] <= 2e-5
for relative, expected in RANK_SOURCES.items():
    if relative.startswith(('models/','datasets/','utils/')):
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
assert representative['status'] == 'pass' and len(representative['workloads']) == 3
for workload in representative['workloads']:
    assert workload['complete_pool'] and len(workload['pairs']) == 2
    assert workload['summary']['status'] == 'pass'
    for pair in workload['pairs']:
        assert pair['agreement']['pass']
        assert pair['agreement']['maps']['max_absolute_error'] <= 2e-5
        assert pair['agreement']['scores']['max_absolute_error'] <= 2e-5
for relative, expected in representative['source_sha256'].items():
    if relative.replace('\\','/').startswith(('models/','datasets/','utils/')):
        normalized = relative.replace('\\','/')
        actual = hashlib.sha256((ROOT/normalized).read_bytes()).hexdigest()
        if actual != expected and normalized == SCORING:
            assert matches_or_previous_scoring(normalized, expected), relative
        elif actual != expected:
            # Historical receipts remain immutable. Only the caller/selector
            # may change, with original packaged hash plus fresh default/fast
            # output equivalence retained by the integration receipt above.
            assert normalized in ('models/musc.py','models/modules/_backend.py'), relative
            assert manifest[normalized]['previous_packaged_sha256'] == expected, relative
            assert integration['source_sha256'][normalized] == actual, relative
smoke=json.loads((ROOT/'benchmarks/harness_validation/small_pool_results.json').read_text())
assert smoke['status']=='pass' and not smoke['workloads'][0]['complete_pool']
assert all(p['agreement']['ground_truth_identical'] for p in smoke['workloads'][0]['pairs'])
process_checks=json.loads((ROOT/'benchmarks/harness_validation/process_tree_tests.json').read_text())
assert all(p['pass'] and not p['worker_alive_after'] and not p['descendant_alive_after'] for p in process_checks)
fast_categories=json.loads((ROOT/'benchmarks/super_fast/category_results.json').read_text())
assert fast_categories['status']=='complete approximate comparison'
assert [(w['category'],w['images']) for w in fast_categories['workloads']]==[('bottle',83),('grid',78),('cable',150)]
for workload in fast_categories['workloads']:
    assert workload['complete_pool'] and len(workload['pairs'])==2
    assert [p['order'] for p in workload['pairs']]==[['original','super_fast'],['super_fast','original']]
    assert all(c['pass'] for checks in workload['repeat_checks'].values() for c in checks.values())
    for pair in workload['pairs']:
        assert pair['same_inputs']
        for name in ('original','super_fast'):
            trial=pair[name]
            assert trial['status']=='pass' and trial['images']==workload['images']
            assert len(trial['image_paths'])==workload['images'] and not trial['metrics']['errors']
            assert len(trial['metrics']['values'])==7
            for relative,expected in trial['source_sha256'].items():
                if relative.replace('\\','/').startswith(('models/','datasets/','utils/','configs/')):
                    assert matches_or_previous_scoring(relative,expected),relative
        assert pair['super_fast']['effective_config']['models']['feature_layers']==[5]
        assert pair['super_fast']['effective_config']['models']['r_list']==[3]
assert [w['summary']['all_metrics_within_provisional_1pp'] for w in fast_categories['workloads']]==[True,False,False]
for path in ROOT.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if not target.startswith(('http:', 'https:', '#')):
            assert (path.parent / target.split('#')[0]).exists(), target
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: {len(manifest)} source/evidence hashes, all Python syntax, local README links,')
print('backend selection, deployed AST equivalence, aggregation factory, evidence tolerances and figures.')
print('Historical full-category evidence, unchanged kernels, super_fast/default integration, harness smoke and process-tree receipts also pass.')
print('Twelve fresh original/super_fast calls, matched complete pools, repeatability, metrics and inference source hashes also pass.')
print('Trimmed-rank deferred scoring: bottle/grid original pairs, cable bitwise previous/new check, super fast rerun')
print('with unchanged metrics, and current source hashes also pass.')
print('CPU-only artifact verification; no additional CUDA trials.')
