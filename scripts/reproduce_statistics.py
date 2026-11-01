import argparse
import json
from pathlib import Path
import subprocess
import sys
from _bootstrap import ROOT
from completion_monitoring.paths import write_environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/statistics')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'experiments/statistics'
    data = ROOT / 'data/statistics'
    commands = [[sys.executable, str(source / 'paired_arf_statistics.py'), '--input', str(data / 'source_stream_scores.csv'), '--archive', str(data / 'archived_table8_statistics.csv'), '--output', str(out / 'table8')], [sys.executable, str(source / 'paired_lrdt_statistics.py'), '--input', str(data / 'source_stream_cells.csv'), '--output', str(out / 'tableS1')]]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    verification = {name: json.loads((out / name / 'reanalysis_environment_and_verification.json').read_text()) for name in ('table8', 'tableS1')}
    write_environment(out / 'environment.json')
    (out / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(f'Statistical reconstruction completed: {out}')
if __name__ == '__main__':
    main()
