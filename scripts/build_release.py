import sys
import shutil
import tempfile
from release_common import ROOT, run, archive
run(sys.executable, '-m', 'build')
wheel = next((ROOT / 'dist').glob('*.whl'))
run(sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', wheel)
# Run outside the checkout so source files cannot mask missing wheel contents.
with tempfile.TemporaryDirectory() as directory:
    run(sys.executable, '-c', 'import musc_gpu; from musc_gpu.cli import main; main()',
        '--help', cwd=directory)
folder = ROOT / 'dist' / 'python-package'
folder.mkdir(exist_ok=True)
for path in (ROOT / 'dist').iterdir():
    if path.suffix == '.whl' or path.name.endswith('.tar.gz'):
        shutil.copy2(path, folder)
(folder / 'INSTALL.txt').write_text('Install the CUDA torch/torchvision pair from the README first.\n'
    'Then: python -m pip install musc_gpu-0.1.0-py3-none-any.whl\n'
    'Run: musc-gpu --help\nModel weights download on first inference; dataset not included.\n', encoding='utf-8')
archive(folder, 'musc-gpu-python.zip')
