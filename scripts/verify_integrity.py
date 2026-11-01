"""Verify data hashes, Python syntax and repository checksums."""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from _bootstrap import ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = list(csv.DictReader((ROOT / 'data/manifest.csv').open()))
    for row in records:
        path = ROOT / row['file']
        assert path.is_file(), path
        assert path.stat().st_size == int(row['bytes']), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
    syntax = 0
    for folder in ['src', 'scripts', 'experiments', 'extensions', 'tests', 'third_party']:
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            assert not re.match(r'^\d', path.name), path
            syntax += 1
    frozen = {name: len(list((ROOT / 'data/frozen' / name).glob('*.npz'))) for name in ['lrdt', 'arf', 'fixed']}
    assert frozen == {'lrdt': 50, 'arf': 50, 'fixed': 5}
    seal = ROOT / 'SHA256SUMS'
    if seal.exists():
        for line in seal.read_text().splitlines():
            digest, filename = line.split('  ', 1)
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    report = {'passed': True, 'data_files': len(records), 'python_files': syntax, 'frozen_streams': frozen}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
