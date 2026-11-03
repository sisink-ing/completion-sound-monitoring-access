from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sys

def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]

def workspace_root() -> Path:
    return Path(os.environ.get('PAA_WORK_ROOT', Path.cwd())).resolve()

def external_data_root(name: str) -> Path:
    key = 'PAA_' + name.upper() + '_DATA'
    folder = 'freddie_mac' if name == 'freddie' else name
    return Path(os.environ.get(key, repository_root() / 'data/raw' / folder)).resolve()

def source_directories() -> list[Path]:
    root = repository_root()
    relative = ['src', 'src/completion_monitoring/engines', 'experiments/synthetic', 'experiments/audits', 'experiments/analysis', 'experiments/baselines', 'experiments/real_world', 'extensions/bts', 'extensions/freddie_mac', 'third_party/creating_drift_benchmarks']
    return [root / p for p in relative]

def source_file(filename: str) -> Path:
    matches = [p / filename for p in source_directories() if (p / filename).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one source named {filename}; found {len(matches)}')
    return matches[0]

def activate_sources() -> None:
    paths = [str(p) for p in source_directories()]
    sys.path[:0] = [p for p in paths if p not in sys.path]
    prior = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + ([prior] if prior else []))

def prepare_workspace(output: Path, output_name: str | None=None, references: bool=True) -> Path:
    """Copy reference inputs into a fresh work area; never write through to data/."""
    root = repository_root()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / 'work'
    if work.exists():
        raise FileExistsError(f'Output already contains a work directory: {work}. Select a new --output directory.')
    work.mkdir()
    result = work / 'result'
    result.mkdir()
    if references:
        for p in (root / 'data/reference').iterdir():
            if not p.is_dir() or p.name in {'statistics', 'retailrocket', output_name}:
                continue
            shutil.copytree(p, result / p.name)
    frozen = root / 'data/frozen'
    mappings = {'fixed': 'fixed_stream_inputs', 'lrdt': 'unseen_stream_pipeline', 'arf': 'arf_holdout'}
    for kind, folder in mappings.items():
        shutil.copytree(frozen / kind, result / folder / 'streams', dirs_exist_ok=True)
    shutil.copytree(root / 'data/raw/benchmarks', work / 'stream_ds')
    for folder in source_directories():
        for source in folder.glob('*.py'):
            target = work / source.name
            if target.exists():
                continue
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
    os.environ['PAA_WORK_ROOT'] = str(work)
    os.environ['CREATING_ROOT'] = str(work)
    return work

def write_environment(path: Path) -> None:
    import importlib.metadata
    import platform
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'river', 'scikit-multiflow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    path.write_text(json.dumps({'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}, indent=2))
