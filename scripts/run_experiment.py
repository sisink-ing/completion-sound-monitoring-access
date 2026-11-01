from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback
from _bootstrap import ROOT
from completion_monitoring.paths import activate_sources, prepare_workspace, source_file, write_environment

def main() -> int:
    registry = json.loads((ROOT / 'configs/experiments.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', choices=sorted(registry))
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args, forwarded = parser.parse_known_args()
    if args.list:
        for name, config in registry.items():
            print(f"{name:22s} {config['description']}")
        return 0
    if args.experiment is None:
        parser.error('Select an experiment or use --list.')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    config = registry[args.experiment]
    if args.smoke and (not config.get('smoke')):
        parser.error('This archived driver has no smoke mode. Use scripts/check_protocol.py for a short protocol check.')
    missing = [name for name in config.get('requires', []) if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('Missing dependencies: ' + ', '.join(missing) + '. See docs/ENVIRONMENTS.md.')
    output = (args.output or ROOT / 'results' / args.experiment).resolve()
    activate_sources()
    work = prepare_workspace(output, config.get('output_name'))
    if args.smoke:
        for key in config.get('smoke_env', []):
            os.environ[key] = '1'
    os.environ['ARF_GENERATION_JOBS'] = str(args.jobs)
    native_args = list(config.get('arguments', []))
    if config.get('native_jobs'):
        native_args += ['--jobs', str(args.jobs)]
    if args.smoke and config.get('native_smoke'):
        native_args.append('--smoke')
    if args.experiment == 'arf' and args.smoke:
        target = work / 'result/arf_holdout/smoke/streams'
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'data/frozen/arf/SEA_streamseed_6101.npz', target / 'SEA_streamseed_6101.npz')
    request = {'experiment': args.experiment, 'smoke': args.smoke, 'jobs': args.jobs, 'driver': config['source'], 'arguments': native_args + forwarded, 'reference_inputs_copied': True, 'work_directory': str(work)}
    (output / 'run.json').write_text(json.dumps(request, indent=2))
    write_environment(output / 'environment.json')
    source = source_file(config['source'])
    previous = Path.cwd()
    try:
        os.chdir(work)
        sys.argv = [str(source)] + native_args + forwarded
        if config.get('top_level'):
            runpy.run_path(str(source), run_name='__main__')
        else:
            module = importlib.import_module(source.stem)
            target = getattr(module, 'step23', module) if config.get('call') == 'step23.main' else module
            for attribute in ['CPU_JOBS', 'EVALUATION_JOBS', 'GENERATION_JOBS']:
                if hasattr(target, attribute):
                    setattr(target, attribute, args.jobs)
            target.main()
    finally:
        os.chdir(previous)
    print(f'Results: {output}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
