class Log:

    def __init__(self, file, verbos):
        self.file = file
        self.verbos = verbos
        self.all = []
        if file != None:
            with open(self.file, 'w') as f:
                f.write('log')

    def write(self, string, verb_level):
        self.all.append(string)
        if self.file != None:
            with open(self.file, 'a') as f:
                f.write('\n\n' + string)
        if self.verbos >= verb_level:
            print(string)

    def get_all(self):
        return self.all[:]

def assert_(crit, msg):
    if not crit:
        log.write(msg, 0)
        assert crit, msg
log = Log(None, 0)
try:
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Stream Generator for Concept Drift\n-----------------------------------\nGenerates realistic synthetic data streams with controlled concept drift.\n\nSupports:\n  - Real drift (label distribution changes)\n  - Virtual drift (feature distribution changes)\n  - Combined and randomized drift patterns\n\nBased on the paper:\n'Real vs. Virtual Drift: Creating Realistic Stream Learning Benchmarks'", formatter_class=argparse.RawTextHelpFormatter, epilog='Examples:\n  python stream_generator.py SEA __v__r__v__ output.npz -n 200\n  python stream_generator.py kdd99_http.csv.xz ____v___v_v___ output.npz -n 100 -t service\n')
    parser.add_argument('input', help='Input data source. Can be a file path or a reference to a registered data loader. If no loader with the given name is found, the input isinterpreted as a file path and loaded using the default file loader.')
    parser.add_argument('pattern', help="Drift pattern to apply to the data stream:\n  '_' : no drift\n  'v' : virtual drift only (feature distribution changes)\n  'r' : real drift only (conditional label distribution changes)\n  'c' : combined drift (both real and virtual)\n  'a' : any drift (randomly real or virtual)")
    parser.add_argument('out', help='Path to output file (NumPy .npz format).', default=None)
    parser.add_argument('-v', '--verbose', help='Verbosity level (e.g., 0 = silent, 1 = default, 2 = full info).', default=1)
    parser.add_argument('-l', '--log', help='Optional path to a log file for recording drift generation details.', default=None)
    parser.add_argument('-n', '--batch_size', help='Number of samples per batch in the generated stream.', default=500)
    parser.add_argument('-t', '--target', help='Name or index of the target (label) column in the dataset.', default=None)
    parser.add_argument('--ignore-columns', help='Comma-separated list of columns to exclude from the input data before stream generation.', default='')
    params = parser.parse_args().__dict__
    params['verbose'] = int(params['verbose'])
    assert_(params['verbose'] in [0, 1, 2], 'invalid verbosity level')
    params['log'] = log = Log(params['log'], params['verbose'])
    import pandas as pd
    import unicodedata
    import re
    import numpy as np
    from gen_data import FileDataSource, datasources
    try:
        if params['verbose'] >= 0:
            from tqdm import tqdm
        else:

            def tqdm(l):
                return l
    except:

        def tqdm(l):
            return l
        log.write('WARN: tqdm not found', 2)
    PANDAS_IO = {'csv': (pd.read_csv, pd.DataFrame.to_csv), 'tsv': (lambda f, **kw: pd.read_csv(f, sep='\t', **kw), lambda df, f, **kw: df.to_csv(f, sep='\t', **kw)), 'json': (pd.read_json, pd.DataFrame.to_json), 'xlsx': (pd.read_excel, pd.DataFrame.to_excel), 'xls': (pd.read_excel, pd.DataFrame.to_excel), 'parquet': (pd.read_parquet, pd.DataFrame.to_parquet), 'feather': (pd.read_feather, pd.DataFrame.to_feather), 'xml': (pd.read_xml, pd.DataFrame.to_xml), 'pkl': (pd.read_pickle, pd.DataFrame.to_pickle), 'pickle': (pd.read_pickle, pd.DataFrame.to_pickle)}

    def to_ascii_lower(s: str) -> str:
        normalized = unicodedata.normalize('NFKD', s)
        ascii_bytes = normalized.encode('ascii', 'ignore')
        return ascii_bytes.decode('ascii').lower()

    def extract_file_format(file_name):
        file_name = to_ascii_lower(file_name).split('.')
        file_name.reverse()
        for part in file_name:
            if part in PANDAS_IO.keys():
                return part

    def read_pandas(file_name):
        frmt = extract_file_format(file_name)
        df = PANDAS_IO[frmt][0](file_name)
        return df
    params['batch_size'] = int(params['batch_size'])
    assert_(params['batch_size'] > 1, 'batch size must be at least 1')
    assert_(re.fullmatch('[_vrca]+', params['pattern']), f"invalid pattern '{params['pattern']}'")
    params['ignore_columns'] = [c for c in re.split('[ \\n\\t]*,[ \\n\\t]*', params['ignore_columns']) if len(c) > 1]
    if params['target']:
        params['target'] = re.split('[ \\n\\t]*,[ \\n\\t]*', params['target'])
    if params['input'] in datasources.keys():
        log.write('creating loader', 1)
        source = datasources[params['input']]()
    else:
        in_format = extract_file_format(params['input'])
        assert_(in_format, 'cannot read input file')
        log.write('read input', 1)
        df = read_pandas(params['input'])
        log.write(repr(df), 2)
        for c in params['ignore_columns']:
            assert_(c in df.columns, f"Column not in columns: '{c}' {df.columns}")
        df = df[[c for c in df.columns if c not in params['ignore_columns'] and c != 'Unnamed: 0']]
        if not params['target']:
            params['target'] = [c for c in df.columns if to_ascii_lower(c) in ['y', 'label', 'out', 'value']]
        assert_(len(params['target']) == 1, 'No/Multi target not supported')
        params['target'] = params['target'][0]
        log.write('creating loader', 1)
        source = FileDataSource(params['input'], params['target'], df)
    log.write(source.get_info(), 2)
    log.write('testing loader', 1)
    source.take(100)
    source.generate_drift()
    source.take(100)
    log.write('createing stream', 1)
    batch_size = params['batch_size']
    stream, dd_true = ([], [])
    for segm_type in tqdm(params['pattern'].replace('_', '-')):
        v, r = source.generate_drift(segm_type)
        dd_true.append((v + 2 * r) * np.ones(batch_size))
        segment = source.take(batch_size)
        stream.append(segment)
    data = np.concatenate(list(map(lambda x: x[0], stream)), axis=0)
    label = np.concatenate(list(map(lambda x: x[1], stream)), axis=0)
    dd_true = np.hstack(dd_true)
    log.write('saving stream', 1)
    np.savez(params['out'], X=data, y=label, dd=dd_true, log=log.get_all())
except Exception as e:
    log.write(str(e), 0)
