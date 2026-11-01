import argparse
import importlib.util
from pathlib import Path
import shutil
import zipfile

def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='original archive containing events.csv')
    parser.add_argument('--output', type=Path, default=root / 'results/retailrocket_preparation')
    args = parser.parse_args()
    source = args.input.resolve()
    with zipfile.ZipFile(source) as archive:
        if 'events.csv' not in archive.namelist():
            parser.error('The input ZIP must contain events.csv at its root.')
    out = args.output.resolve()
    if out.exists():
        parser.error('Choose a new output directory to preserve prior preprocessing results.')
    (out / 'raw').mkdir(parents=True)
    (out / 'processed').mkdir()
    shutil.copy2(source, out / 'raw/RetailRocket.zip')
    (out / 'processed/catalog.json').write_text('[]\n')
    builder = root / 'experiments/real_world/task_construction/query_builder.py'
    spec = importlib.util.spec_from_file_location('retail_causal_query_builder', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.H = out
    module.main()
    print(f"Prepared catalog: {out / 'processed'}")
if __name__ == '__main__':
    main()
