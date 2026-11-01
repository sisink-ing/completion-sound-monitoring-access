import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from _bootstrap import ROOT
from completion_monitoring.paths import activate_sources, prepare_workspace, source_file, write_environment
EXTENSIONS = {'bts_prepare': 'prepare_bts.py', 'bts_static': 'static_baselines.py', 'bts_schedule': 'build_feedback_schedule.py', 'bts_matched': 'matched_window_evaluation.py', 'bts_adaptive': 'adaptive_replay.py', 'freddie_replay': 'chronological_replay.py', 'freddie_feasibility': 'task_feasibility.py', 'freddie_cohort': 'cohort_feasibility.py', 'freddie_missingness': 'missing_label_audit.py', 'freddie_outcomes': 'reconstruct_outcomes.py', 'freddie_protocol': 'protocol_replay.py', 'freddie_calendar': 'calendar_replay.py'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=sorted(EXTENSIONS))
    p.add_argument('--output', type=Path)
    args, forwarded = p.parse_known_args()
    out = (args.output or ROOT / 'results/extensions' / args.experiment).resolve()
    source = EXTENSIONS[args.experiment]
    activate_sources()
    work = prepare_workspace(out, references=False)
    if args.experiment.startswith('bts_'):
        shutil.copytree(ROOT / 'data/raw/bts', work / 'data/raw_bts')
    for entry in (ROOT / 'data/extensions').iterdir():
        if entry.name != Path(source).stem:
            shutil.copytree(entry, work / 'result' / entry.name)
    (out / 'run.json').write_text(json.dumps({'extension': args.experiment, 'source': source, 'scope': 'outside manuscript benchmarks'}, indent=2))
    write_environment(out / 'environment.json')
    os.chdir(work)
    sys.argv = [str(source_file(source))] + forwarded
    module = importlib.import_module(Path(source).stem)
    if hasattr(module, 'main'):
        module.main()
    print(f'Extension results: {out}')
if __name__ == '__main__':
    main()
