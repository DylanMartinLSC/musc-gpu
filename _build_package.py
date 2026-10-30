from pathlib import Path
import shutil
from setuptools.command.build_py import build_py

class RuntimeBuild(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        target = Path(self.build_lib) / 'musc_gpu' / '_runtime'
        for name in ('models', 'datasets', 'utils', 'configs', 'examples', 'licenses'):
            shutil.copytree(root / name, target / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            if name in ('models', 'datasets', 'utils'):
                # Prefer this bundled runtime over unrelated top-level packages.
                (target / name / '__init__.py').touch(exist_ok=True)
        for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
            shutil.copy2(root / name, target / name)
