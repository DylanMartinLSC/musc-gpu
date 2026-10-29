"""Run the bundled research entry point without relying on the checkout or cwd."""
from pathlib import Path
import runpy
import sys

def main():
    root = Path(__file__).parent / '_runtime'
    args = sys.argv[1:]
    if '--config' not in args and not any(x.startswith('--config=') for x in args):
        args = ['--config', str(root / 'configs' / 'musc.yaml'), *args]
    sys.path.insert(0, str(root))
    sys.argv = [str(root / 'examples' / 'musc_main.py'), *args]
    runpy.run_path(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
