from pathlib import Path
import os, sys, json, hashlib, importlib, time, ast, copy, warnings
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'deps'))
sys.path.insert(0, str(HERE / 'original'))
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score, precision_score, recall_score
warnings.filterwarnings('ignore', category=UserWarning)
ORIGINAL_CWD = Path.cwd()
(HERE / 'work').mkdir(exist_ok=True)
os.chdir(HERE / 'work')
m = importlib.import_module('cohort_protocol')
b = m.base
ORIG_FACTORY = b.create_model
ORIG_STATE = m.create_state
ORIG_UPDATE = m.perform_update
ORIG_BOUNDS = b.evaluate_bounds
DEFAULT_STRATEGIES = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
NAMES = {'STATIC': 'Static', 'FULL_SOFT_Q30': 'Soft-q30', 'FULL_COMPLETE_COHORT': 'Complete Cohort', 'WAIT_ALL_COMPLETE_COHORT': 'Wait-All', 'ARRIVED_ONLY_COMPLETE_COHORT': 'Arrived-Only', 'DELAYED_ADWIN_RIVER': 'Delayed-ADWIN (River)'}
CACHE = {}
CTX = {}
STATES = {}

class ExactFastModel:

    def __init__(self, learner):
        self.learner = learner
        self.raw = ORIG_FACTORY(learner)
        self.tag = CTX.get('active')
        self.fit_checks = 0

    def fit(self, X, y):
        self.raw.fit(X, y)
        if self.learner == 'LR':
            s = self.raw.named_steps['scaler']
            c = self.raw.named_steps['classifier']
            self.mean = s.mean_
            self.scale = s.scale_
            self.coef = c.coef_[0]
            self.intercept = c.intercept_[0]
            self.classes = c.classes_
        else:
            self.tree = self.raw.tree_
            self.classes = self.raw.classes_
        assert np.array_equal(self._classify(X), self.raw.predict(X)), 'Fast inference differs from sklearn'
        self.fit_checks += 1
        return self

    def _score(self, X):
        if self.learner == 'LR':
            return (np.asarray(X) - self.mean) / self.scale @ self.coef + self.intercept
        val = self.tree.predict(np.asarray(X, dtype=np.float32))
        return val[:, 1] / val.sum(axis=1)

    def _classify(self, X):
        if self.learner == 'LR':
            return self.classes[(self._score(X) > 0).astype(int)]
        val = self.tree.predict(np.asarray(X, dtype=np.float32))
        return self.classes[val.argmax(axis=1)]

    def predict(self, X):
        if self.learner == 'LR':
            score = self._score(X)
            pred = self.classes[(score > 0).astype(int)]
            proba = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
        else:
            val = self.tree.predict(np.asarray(X, dtype=np.float32))
            pred = self.classes[val.argmax(axis=1)]
            proba = val[:, 1] / val.sum(axis=1)
        if len(X) == 1 and self.tag and CTX.get('record', False):
            CTX['trace'].setdefault(self.tag, []).append((int(pred[0]), float(proba[0])))
        return pred

    def predict_proba(self, X):
        score = self._score(X)
        p = 1 / (1 + np.exp(-np.clip(score, -700, 700))) if self.learner == 'LR' else score
        return np.column_stack([1 - p, p])

def factory(learner):
    return ExactFastModel(learner)

def create_state(X, y, learner):
    tag = m.STRATEGIES[CTX['state_counter']]
    CTX['state_counter'] += 1
    CTX['active'] = tag
    s = ORIG_STATE(X, y, learner)
    STATES[tag] = s
    return s

def perform_update(strategy, state, *args):
    CTX['active'] = strategy
    return ORIG_UPDATE(strategy, state, *args)
b.create_model = factory
m.create_state = create_state
m.perform_update = perform_update

def get_data(name):
    if name not in CACHE:
        cat = json.loads((HERE / 'processed/catalog.json').read_text())
        info = next((x for x in cat if x['name'] == name))
        CACHE[name] = (pd.read_pickle(HERE / f'processed/{name}.pkl'), info)
    return CACHE[name]

def transform(df, tr, info):
    nums = info['numeric_features']
    cats = info['categorical_features']
    num = df[nums].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = num.iloc[tr].median().fillna(0)
    num = num.fillna(med).fillna(0)
    if info['family'] == 'GasDrift':
        num = np.sign(num) * np.log1p(np.abs(num))
    else:
        for col in ['price', 'freight', 'weight_g', 'volume_cm3', 'duration_ms', 'campaign', 'previous']:
            if col in num:
                num[col] = np.log1p(num[col].clip(lower=0))
    arrays = [num.to_numpy(dtype=float)]
    names = list(nums)
    enc = None
    if cats:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
        enc.fit(df.iloc[tr][cats].fillna('MISSING').astype(str))
        arrays.append(enc.transform(df[cats].fillna('MISSING').astype(str)))
        names += list(enc.get_feature_names_out(cats))
    X = np.column_stack(arrays)
    assert np.isfinite(X).all()
    return (X, names, {'numeric_impute_medians': med.to_dict(), 'categorical_vocabulary': dict(zip(cats, [a.tolist() for a in enc.categories_])) if enc is not None else {}, 'numeric_transforms': 'Gas signed log1p all inputs; selected positive-skew price/freight/weight/volume/duration/campaign/previous log1p; otherwise identity. LR further uses original StandardScaler.'})

def make_case(task):
    d, info = get_data(task['dataset'])
    n = len(d)
    cut = info['outer_cut']
    seed = task.get('delay_seed', 5001)
    mode = task['delay_mode']
    if mode == 'NATIVE':
        clock = d.clock.to_numpy(dtype=np.int64)
        av = d.feedback_clock.to_numpy(dtype=np.int64)
    else:
        clock = np.arange(n, dtype=np.int64)
        l0, l1 = {'UNIFORM': (100, 100), 'SEVERE': (20, 200), 'REVERSED': (200, 20)}[mode]
        rng = np.random.default_rng(seed)
        av = clock + rng.poisson(np.where(d.label.to_numpy() == 0, l0, l1))
    if task['stage'] == 'screen':
        start = task['start']
        tr = np.arange(start, start + 200)
        if len(tr) != 200 or tr[-1] >= cut:
            raise ValueError('insufficient_train')
        ca_start = int(np.searchsorted(clock, av[tr].max(), side='right'))
        ca = np.arange(ca_start, ca_start + 200)
        if ca[-1] >= cut:
            raise ValueError('insufficient_calibration')
        online_start = int(np.searchsorted(clock, av[ca].max(), side='right'))
        end = online_start + task['online_n']
        if end > cut:
            raise ValueError('insufficient_screen_online')
        online = np.arange(online_start, end)
    else:
        online = np.arange(cut, n)
        tr = ca = None
        for lookback in [1000, 2000, 4000, 8000, 16000, cut]:
            start = max(0, cut - lookback)
            tt = np.arange(start, start + 200)
            if tt[-1] >= cut:
                continue
            cs = int(np.searchsorted(clock, av[tt].max(), side='right'))
            cc = np.arange(cs, cs + 200)
            if cc[-1] < cut and av[cc].max() < clock[cut]:
                tr, ca = (tt, cc)
                break
        if tr is None:
            raise ValueError('no_causal_holdout_warmup')
    if len(np.unique(d.label.iloc[tr])) != 2:
        raise ValueError('single_class_initial_train')
    if len(online) < 400:
        raise ValueError('short_online')
    selected = np.concatenate([tr, ca, online])
    dd = d.iloc[selected].copy().reset_index(drop=True)
    dd['phase'] = ['train'] * 200 + ['calibration'] * 200 + ['online'] * len(online)
    dd['source_stream_row'] = selected
    dd['processed_event_id'] = np.arange(len(dd))
    dd['availability_clock'] = av[selected]
    xx, feature_names, pre = transform(dd, np.arange(200), info)
    yy = dd.label.to_numpy(dtype=int)
    pts = clock[online]
    ats = av[online]
    slots = np.searchsorted(pts, ats, side='left') + 400
    slots = np.maximum(slots, np.arange(400, len(dd)))
    after = slots >= len(dd)
    tail = np.unique(ats[after])
    slots[after] = len(dd) + np.searchsorted(tail, ats[after])
    amap = dict(zip(range(400, len(dd)), map(int, slots)))
    groups = {}
    for i, a in amap.items():
        groups.setdefault(a, []).append(i)
    wall = {400 + i: int(v) for i, v in enumerate(pts)}
    wall.update({len(dd) + i: int(v) for i, v in enumerate(tail)})
    clockscale = 1000000000.0 if mode == 'NATIVE' else 1
    audit = {'task': task, 'dataset_metadata': info, 'selected_rows': len(dd), 'online_n': len(online), 'source_train_start': int(tr[0]), 'source_train_end': int(tr[-1]), 'source_calibration_start': int(ca[0]), 'source_calibration_end': int(ca[-1]), 'source_online_start': int(online[0]), 'source_online_end': int(online[-1]), 'train_class1_rate': float(yy[:200].mean()), 'cal_class1_rate': float(yy[200:400].mean()), 'online_class1_rate': float(yy[400:].mean()), 'test_majority_diagnostic': float(max(yy[400:].mean(), 1 - yy[400:].mean())), 'adjacent_feedback_inversions': int(np.sum(np.diff(ats) < 0)), 'post_horizon_labels': int(after.sum()), 'feedback_time_unit': 'seconds' if mode == 'NATIVE' else 'source-event steps; post-horizon drain is rank-based', 'preprocessing': pre, 'features': feature_names, 'causal_checks': {'train_ready_before_cal': bool(av[tr].max() < clock[ca[0]]), 'cal_ready_before_online': bool(av[ca].max() < clock[online[0]]), 'no_holdout_training': bool(tr[-1] < cut and ca[-1] < cut) if task['stage'] != 'screen' else None}}
    assert audit['causal_checks']['train_ready_before_cal'] and audit['causal_checks']['cal_ready_before_online']
    return (dd, xx, yy, amap, groups, wall, clockscale, audit)

class ADWINCompat:

    def __init__(self, delta=0.002):
        from river.drift import ADWIN
        self.d = ADWIN(delta=delta)

    def add_element(self, v):
        self.d.update(v)

    def detected_change(self):
        return self.d.drift_detected

    @property
    def width(self):
        return self.d.width

    @property
    def estimation(self):
        return self.d.estimation

def build_adwin():
    source = (HERE / 'original/delayed_adwin.py').read_text()
    tree = ast.parse(source)
    node = next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'adwin_one'))
    idx = next((i for i, x in enumerate(node.body) if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Tuple) and any((isinstance(v, ast.Name) and v.id == 'reference' for v in x.targets[0].elts))))
    body = copy.deepcopy(node.body[idx:])
    filtered = []
    for stmt in body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Tuple) and any((isinstance(v, ast.Name) and v.id == 'arrival_map' for v in stmt.targets[0].elts)):
            continue
        filtered.append(stmt)

    class LearnerReplace(ast.NodeTransformer):

        def visit_Constant(self, node):
            if node.value == 'ARF':
                return ast.copy_location(ast.Name(id='learner', ctx=ast.Load()), node)
            if node.value == 'skmultiflow.AdaptiveRandomForestClassifier':
                return ast.copy_location(ast.Constant(value='common sklearn learner; River0.22 ADWIN'), node)
            return node
    fn = ast.FunctionDef(name='run_adwin_replay', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ['family', 'X', 'y', 'learner', 'arrival_map', 'arrival_groups', 'scenario', 'delay_seed']], kwonlyargs=[], kw_defaults=[], defaults=[]), body=filtered, decorator_list=[])
    fn = LearnerReplace().visit(fn)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'ADWIN': ADWINCompat, 'base': b, 'np': np, 'time': time, 'accuracy_score': accuracy_score, 'f1_score': f1_score, 'matthews_corrcoef': matthews_corrcoef, 'delta': 0.002, 'stream_seed': 0, 'lambda0': 0, 'lambda1': 0, 'NEW': 'DELAYED_ADWIN_RIVER', 'sha': 'paired-input', 'stored_hash': 'paired-input', 'SEED': 2026, 'KEY': ['dataset', 'stream_seed', 'scenario', 'delay_seed']}
    exec(compile(module, 'adapted_original_step30', 'exec'), namespace)
    return namespace['run_adwin_replay']
ADWIN_RUN = None

def run_task(task, save_dir=None, include_adwin=False):
    global ADWIN_RUN
    started = time.time()
    ident = task['task_id']
    try:
        dd, X, y, amap, groups, wall, scale, audit = make_case(task)
        CTX.clear()
        CTX.update(trace={}, record=True, state_counter=0)
        STATES.clear()
        b.generate_delay = lambda *a, **kw: (amap, groups)
        b.evaluate_bounds = ORIG_BOUNDS
        m.STRATEGIES = DEFAULT_STRATEGIES.copy()
        m.ADAPTIVE_STRATEGIES = DEFAULT_STRATEGIES[1:]
        runs, dec, updates, skips = m.run_one(task['dataset'], X, y, task['learner'], task['delay_mode'], 0, 0, task['delay_seed'])
        m.STRATEGIES = ['FULL_COMPLETE_COHORT']
        m.ADAPTIVE_STRATEGIES = ['FULL_COMPLETE_COHORT']
        CTX['state_counter'] = 0
        b.evaluate_bounds = b.evaluate_arrived_only
        oldtrace = CTX['trace']
        CTX['trace'] = {}
        rr, de, up, sk = m.run_one(task['dataset'], X, y, task['learner'], task['delay_mode'], 0, 0, task['delay_seed'])
        for lists in [rr, de, up, sk]:
            for row in lists:
                row['strategy'] = 'ARRIVED_ONLY_COMPLETE_COHORT'
        arrtrace = CTX['trace'].get('FULL_COMPLETE_COHORT', [])
        CTX['trace'] = oldtrace
        CTX['trace']['ARRIVED_ONLY_COMPLETE_COHORT'] = arrtrace
        runs += rr
        dec += de
        updates += up
        skips += sk
        b.evaluate_bounds = ORIG_BOUNDS
        if include_adwin:
            if ADWIN_RUN is None:
                ADWIN_RUN = build_adwin()
            input_sha = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
            ADWIN_RUN.__globals__['sha'] = input_sha
            ADWIN_RUN.__globals__['stored_hash'] = input_sha
            CTX['active'] = 'DELAYED_ADWIN_RIVER'
            row, alarms, up = ADWIN_RUN(task['dataset'], X, y, task['learner'], amap, groups, task['delay_mode'], task['delay_seed'])
            runs.append(row)
            for u in up:
                u['strategy'] = 'DELAYED_ADWIN_RIVER'
            updates += up
        pred_rows = []
        decdf = pd.DataFrame(dec)
        if len(decdf):
            decdf['close_tick'] = (decdf.decision_time - decdf.decision_delay).astype(int)
            decdf['matched_wait_all_tick'] = [max(int(c), max((amap[i] for i in range(int(c) - 99, int(c) + 1)))) for c in decdf.close_tick]
            decdf['matched_advance_clock'] = [(wall[int(w)] - wall[int(t)]) / scale for w, t in zip(decdf.matched_wait_all_tick, decdf.decision_time)]
            decdf['confirmation_delay_clock'] = [(wall[int(t)] - wall[int(c)]) / scale for t, c in zip(decdf.decision_time, decdf.close_tick)]
            assert (decdf.matched_advance_clock >= -1e-08).all()
        for row in runs:
            tag = row['strategy']
            trace = np.asarray(CTX['trace'][tag])
            assert len(trace) == len(y) - 400, (tag, len(trace), len(y) - 400)
            yp = trace[:, 0].astype(int)
            prob = trace[:, 1]
            truth = y[400:]
            acc = float(accuracy_score(truth, yp))
            assert abs(acc - row['accuracy']) < 1e-12, (tag, 'trace vs engine accuracy')
            row.update(task_id=ident, stage=task['stage'], learner=task['learner'], delay_mode=task['delay_mode'], start=task.get('start', -1), requested_online_n=task.get('online_n', -1), online_n=len(truth), balanced_accuracy=float(balanced_accuracy_score(truth, yp)), average_precision=float(average_precision_score(truth, prob)), positive_recall=float(recall_score(truth, yp, zero_division=0)), positive_precision=float(precision_score(truth, yp, zero_division=0)), majority_diagnostic=audit['test_majority_diagnostic'], online_positive_rate=audit['online_class1_rate'], feature_count=X.shape[1], total_seconds=time.time() - started)
            sub = decdf[decdf.strategy == tag] if len(decdf) else pd.DataFrame()
            row['mean_matched_advance_clock'] = float(sub.matched_advance_clock.mean()) if len(sub) else None
            row['early_confirmation_rate'] = float((sub.matched_wait_all_tick > sub.decision_time).mean()) if len(sub) else None
            row['online_resolved_windows'] = int((sub.decision_time < len(y)).sum()) if len(sub) else 0
            if save_dir:
                pred_rows.append(pd.DataFrame({'processed_event_id': np.arange(400, len(y)), 'source_id': dd.source_id.iloc[400:].to_numpy(), 'label': truth, 'strategy': tag, 'prediction': yp, 'probability_class1': prob}))
        if save_dir:
            path = Path(save_dir) / ident
            path.mkdir(parents=True, exist_ok=True)
            dd.to_csv(path / 'source_rows_with_split.csv', index=False, encoding='utf-8-sig')
            feature = pd.DataFrame(X, columns=audit['features'])
            feature.insert(0, 'processed_event_id', np.arange(len(y)))
            feature['label'] = y
            feature['phase'] = dd.phase
            feature['source_id'] = dd.source_id
            feature.to_csv(path / 'processed_features.csv', index=False, encoding='utf-8-sig')
            np.savez_compressed(path / 'processed_arrays.npz', X=X, y=y, arrival=np.array([amap.get(i, -1) for i in range(len(y))]))
            pd.DataFrame({'processed_event_id': range(400, len(y)), 'source_id': dd.source_id.iloc[400:].to_numpy(), 'release_tick': [amap[i] for i in range(400, len(y))], 'event_clock': dd.clock.iloc[400:].to_numpy() if task['delay_mode'] == 'NATIVE' else dd.source_stream_row.iloc[400:].to_numpy(), 'label_available_clock': dd.availability_clock.iloc[400:].to_numpy()}).to_csv(path / 'feedback_schedule.csv', index=False)
            pd.concat(pred_rows, ignore_index=True).to_csv(path / 'historical_predictions.csv', index=False)
            decdf.to_csv(path / 'window_decisions.csv', index=False)
            pd.DataFrame(updates).to_csv(path / 'updates.csv', index=False)
            pd.DataFrame(runs).to_csv(path / 'metrics.csv', index=False)
            (path / 'protocol.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        return {'task_id': ident, 'status': 'ok', 'rows': runs, 'audit': {k: v for k, v in audit.items() if k not in ['features', 'preprocessing', 'dataset_metadata']}}
    except Exception as e:
        import traceback
        return {'task_id': ident, 'status': 'skipped' if isinstance(e, ValueError) else 'error', 'error': repr(e), 'traceback': traceback.format_exc()[-1800:], 'task': task}
    finally:
        b.evaluate_bounds = ORIG_BOUNDS
        m.STRATEGIES = DEFAULT_STRATEGIES.copy()
        m.ADAPTIVE_STRATEGIES = DEFAULT_STRATEGIES[1:]
        CTX['record'] = False
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    task = {'task_id': 'smoke_electricity', 'stage': 'screen', 'dataset': 'Electricity', 'learner': 'LR', 'delay_mode': 'SEVERE', 'delay_seed': 5001, 'start': 0, 'online_n': 1800}
    r = run_task(task, save_dir=HERE / 'smoke', include_adwin=True)
    if r['status'] == 'ok':
        print(pd.DataFrame(r['rows'])[['strategy', 'accuracy', 'macro_f1', 'mcc', 'decision_accuracy', 'retrain_count']].to_string(index=False))
        print('ADWIN_OK')
    else:
        print(json.dumps(r, indent=2))
