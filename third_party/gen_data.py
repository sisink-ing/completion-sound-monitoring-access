import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer
try:
    from river.datasets.synth import Agrawal, SEA, Mixed, RandomTree, Sine
except:
    print('River not available, potentially limited functionality!')

def normalize_drift_states(drift_states):
    if type(drift_states) is int:
        drift_states = range(drift_states)
    if hasattr(drift_states, '__iter__'):
        drift_states = list(drift_states)
    return drift_states

class DataSource:

    def __init__(self, virtual_drift_states, real_drift_states):
        self.ds_initparams = {'virtual_drift_states': virtual_drift_states, 'real_drift_states': real_drift_states}
        self.drift_states = dict(enumerate([(v, r) for v in normalize_drift_states(virtual_drift_states) for r in normalize_drift_states(real_drift_states)]))
        self.drift_state_range = max(list(self.drift_states.keys())) + 1
        self.drift_state = np.random.choice(self.drift_state_range)

    def take_some(self, n_datapoints):
        raise NotImplemented()

    def take(self, n_datapoints=1):
        rec = 0
        Xs, ys = ([], [])
        while rec < n_datapoints:
            X, y = self.take_some(n_datapoints - rec)
            assert X.shape[0] == y.shape[0]
            rec += X.shape[0]
            Xs.append(X)
            ys.append(y)
        X = np.vstack(Xs)
        y = np.hstack(ys)
        if X.shape[0] > n_datapoints:
            sel = np.random.choice(range(X.shape[0]), size=n_datapoints, replace=False)
            X, y = (X[sel], y[sel])
        return (X, y)

    def get_drift_state(self):
        return self.drift_state

    def set_drift_state(self, new_state):
        assert new_state in range(self.drift_state_range)
        self.drift_state = new_state

    def get_drift_states(self):
        return range(self.drift_state_range)

    def get_drift_state_info(self, state=None):
        if state is None:
            state = self.get_drift_state()
        return self.drift_states[state]

    def generate_drift(self, drift_type='a'):
        if drift_type == '-':
            return (False, False)
        states = self.get_drift_states()
        old_state_info = self.get_drift_state_info()
        for new_state in np.random.permutation(states):
            new_state_info = self.get_drift_state_info(new_state)
            if drift_type == 'a' and (new_state_info[0] != old_state_info[0] or new_state_info[1] != old_state_info[1]):
                self.set_drift_state(new_state)
                return (new_state_info[0] != old_state_info[0], new_state_info[1] != old_state_info[1])
            elif drift_type == 'v' and (new_state_info[0] != old_state_info[0] and new_state_info[1] == old_state_info[1]):
                self.set_drift_state(new_state)
                return (new_state_info[0] != old_state_info[0], new_state_info[1] != old_state_info[1])
            elif drift_type == 'r' and (new_state_info[0] == old_state_info[0] and new_state_info[1] != old_state_info[1]):
                self.set_drift_state(new_state)
                return (new_state_info[0] != old_state_info[0], new_state_info[1] != old_state_info[1])
            elif drift_type == 'c' and (new_state_info[0] != old_state_info[0] and new_state_info[1] != old_state_info[1]):
                self.set_drift_state(new_state)
                return (new_state_info[0] != old_state_info[0], new_state_info[1] != old_state_info[1])
        raise ValueError('No combination found')

    def size(self):
        raise NotImplemented()

    def get_info(self):
        return {**self.ds_initparams, 'class': self.__class__.__name__}

def gen_window_matrix(l1, l2, n_perm, cache=dict()):
    if (l1, l2, n_perm) not in cache.keys():
        w = np.array(l1 * [1.0 / l1] + l2 * [-1.0 / l2])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
        cache[l1, l2, n_perm] = W
    return cache[l1, l2, n_perm]

def mmd(X1, X2, n_perm=2500):
    X = np.vstack((X1, X2))
    L = RandomTreesEmbedding().fit(X).apply(X)
    K = (L[:, None, :] == L[None, :, :]).mean(axis=2)
    W = gen_window_matrix(X1.shape[0], X2.shape[0], n_perm)
    s = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (s[0] < s).sum() / n_perm
    return p

class FunctionDataSource(DataSource):

    def __init__(self, generator_functions, label_functions, real_drift_warning_level=0.01, virtual_drift_verificaiton_p_val=0.001, test_size=1000):
        super().__init__(len(generator_functions), len(label_functions))
        self.generator_functions = generator_functions
        self.label_functions = label_functions
        self.real_drift_warning_level = real_drift_warning_level
        self.virtual_drift_verificaiton_p_val = virtual_drift_verificaiton_p_val
        self.test_size = test_size

    def take_some(self, n_datapoints):
        X = self.generator_functions[self.get_drift_state_info()[0]](n_datapoints)
        y = self.label_functions[self.get_drift_state_info()[1]](X)
        return (X, y)

    def generate_drift(self, drift_type='a'):
        old_state = self.get_drift_state()
        X0, _ = self.take(self.test_size)
        max_diff, max_state = (-1, -1)
        for _ in range(100):
            v, r = super().generate_drift(drift_type)
            if v or r:
                X, y1 = self.take(self.test_size)
            if v:
                if mmd(X0, X) > self.virtual_drift_verificaiton_p_val:
                    self.set_drift_state(old_state)
                    continue
            if r:
                y2 = self.label_functions[self.get_drift_state_info(old_state)[1]](X)
                diff = (y1 != y2).mean()
                if diff > max_diff:
                    max_state, max_diff = (self.get_drift_state(), diff)
                if diff < self.real_drift_warning_level:
                    self.set_drift_state(old_state)
                    continue
            return (v, r)
        self.set_drift_state(max_state)
        if max_diff == 0:
            raise ValueError('Unable to create real drift')
        else:
            X, y1 = self.take(self.test_size)
            y2 = self.label_functions[self.get_drift_state_info(old_state)[1]](X)
            diff = (y1 != y2).mean()
            print(f'WARN: Unable to create (noticable) drift. Intensity: real: {diff}, virtual: {mmd(X0, X)}')
        cs = self.get_drift_state_info()
        os = self.get_drift_state_info(old_state)
        return (cs[i] != os[i] for i in range(2))

    def size(self):
        return np.infty

def filter_array(X, filter_function):
    return X[filter_function(X)]

class FilterDataSource(FunctionDataSource):

    def __init__(self, generator_function, filter_functions, label_functions):
        super().__init__([lambda n_datapoints, filter_function=filter_function: filter_array(generator_function(max(100, n_datapoints)), filter_function) for filter_function in filter_functions], label_functions)

class KMeansFilterDataSource(FilterDataSource):

    def __init__(self, generator_function, label_functions, n_clusters=20, n_variants=2, init_size=1000):
        self.kmfilter_initparams = {'n_clusters': n_clusters, 'init_size': init_size}
        n_clusters = (n_clusters // n_variants + 1) * n_variants
        clustering = MiniBatchKMeans(n_clusters=n_clusters).fit(generator_function(init_size))
        super().__init__(generator_function, [lambda X, i=i: clustering.predict(X) % n_variants == i for i in range(n_variants)], label_functions)

    def get_info(self):
        return {**super().get_info(), **self.kmfilter_initparams}

def river_to_array(sample, columns=None, label_name=None):
    sample = list(sample)
    df = pd.DataFrame([x for x, _ in sample])
    if label_name is not None:
        assert label_name not in df.columns
        df[label_name] = np.array([y for _, y in sample])
    if columns is None:
        columns = list(df.columns)[:]
        columns.sort(key=lambda c: str(c))
    df = df[columns]
    return df.values

class SEADataSource(KMeansFilterDataSource):

    def __init__(self, n_clusters=20, n_variants=2, init_size=1000):
        sea = SEA()
        sea_thresholds = {0: 8, 1: 9, 2: 7, 3: 9.5}
        super().__init__(generator_function=lambda n_datapoints: river_to_array(sea.take(n_datapoints)), label_functions=[lambda X, i=i: (X[:, 0] + X[:, 1] > sea_thresholds[i]).astype(int) for i in range(len(sea_thresholds))], n_clusters=n_clusters, n_variants=n_variants, init_size=init_size)

class AgrawalDataSource(KMeansFilterDataSource):

    def __init__(self, n_clusters=20, n_variants=2, init_size=1000):
        agrawal = Agrawal()
        super().__init__(generator_function=lambda n_datapoints: river_to_array(agrawal.take(n_datapoints), columns=['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']), label_functions=[lambda X, fun=fun: np.array(list(map(lambda x: fun(*x), X))) for fun in agrawal._classification_functions[:-1]], n_clusters=n_clusters, n_variants=n_variants, init_size=init_size)

class MixedDataSource(KMeansFilterDataSource):

    def __init__(self, n_clusters=20, n_variants=2, init_size=1000):
        mixed = Mixed()
        super().__init__(generator_function=lambda n_datapoints: river_to_array(mixed.take(n_datapoints), columns=[0, 1, 2, 3]), label_functions=[lambda X, fun=fun: np.array(list(map(lambda x: fun(*x), X))) for fun in mixed._functions], n_clusters=n_clusters, n_variants=n_variants, init_size=init_size)

class SineDataSource(KMeansFilterDataSource):

    def __init__(self, n_clusters=20, n_variants=2, init_size=1000):
        sine = Sine()
        super().__init__(generator_function=lambda n_datapoints: river_to_array(sine.take(n_datapoints), columns=[0, 1]), label_functions=[lambda X, fun=fun: np.array(list(map(lambda x: fun(*x), X))) for fun in sine._functions], n_clusters=n_clusters, n_variants=n_variants, init_size=init_size)

def compute_tree(tree, columns, X):
    df = pd.DataFrame(X, columns=columns)
    for c in df.columns:
        if c[:6] == 'x_cat_':
            df[c] = df[c].map(int)
        elif c[:6] != 'x_num_':
            raise ValueError()
    return np.array(list(map(lambda x: tree._classify_instance(tree.tree_root, x), df.to_dict(orient='records'))))

class RandomTreeDataSource(KMeansFilterDataSource):

    def __init__(self, n_clusters=20, n_virtual_variants=2, n_real_variants=2, init_size=1000, **kwds):
        self.rt_initparams = kwds
        trees = [RandomTree(seed_tree=int(seed), **kwds) for seed in np.random.choice(10000, n_real_variants, replace=False)]
        columns = trees[0].feature_names
        for tree in trees:
            assert tree.feature_names == columns
            tree._generate_random_tree()
        super().__init__(generator_function=lambda n_datapoints: river_to_array(trees[0].take(n_datapoints), columns=columns), label_functions=[lambda X, i=i: compute_tree(trees[i], columns, X) for i in range(n_real_variants)], n_clusters=n_clusters, n_variants=n_virtual_variants, init_size=init_size)

    def get_info(self):
        return {**super().get_info(), **self.rt_initparams}

def normalize_block(block, n_rows):
    if type(block) is int:
        block = np.linspace(0, 1, block + 2)[1:-1]
    else:
        block = np.array(list(block))
    block.sort()
    assert block[0] > 0
    assert np.diff(block).min() > 0
    if block[-1] < 1:
        block = (block * n_rows).astype(int)
    assert block[-1] < n_rows
    return block.astype(int)

def load_file(file, cache=dict()):
    if file not in cache.keys():
        if '.csv' in file:
            df = pd.read_csv(file)
        elif '.pkl' in file:
            df = pd.read_pickle(file)
        elif '.npz' in file:
            npf = np.load(file)
            dats = []
            cols = []
            for f in npf.files:
                X = npf[f]
                X = X.reshape(X.shape[0], -1)
                dats.append(X)
                cols.extend([f'{f}_{i}' for i in range(X.shape[1])])
            df = pd.DataFrame(np.hstack(dats), columns=cols)
        else:
            raise ValueError(f'No loader known for {file}')
        cache[file] = df
    return cache[file].copy()

def sample(p):
    assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
    return (np.cumsum(p, axis=1) < np.random.random(size=p.shape[0])[:, None]).sum(axis=1)

def random_derangement(n):
    arr = np.arange(n)
    perm = np.random.permutation(arr)
    while True:
        fixed_points = perm == arr
        if not np.any(fixed_points):
            return perm
        indices = np.where(fixed_points)[0]
        if len(indices) == 1:
            swap_idx = np.random.choice(np.setdiff1d(np.arange(n), indices))
            perm[indices[0]], perm[swap_idx] = (perm[swap_idx], perm[indices[0]])
        else:
            np.random.shuffle(indices)
            perm[indices] = np.roll(perm[indices], shift=-1)

def add_flip(b, X, y, n_clusters, n_ignore=5):
    y = y.astype(int).flatten()
    b = b.flatten()
    assert X.shape[0] == y.shape[0] and X.shape[0] == b.shape[0]
    encoder = LabelEncoder().fit(y)
    y = encoder.transform(y)
    n_classes = np.unique(y).shape[0]
    n_blocks = np.unique(y).shape[0]
    assert n_classes == y.max() + 1
    indices = b == MiniBatchKMeans(n_clusters=n_clusters).fit_predict(X) % (n_blocks + n_ignore)
    perm = random_derangement(n_classes)
    y[indices] = perm[y[indices]]
    return encoder.inverse_transform(y)

class FileDataSource(FunctionDataSource):

    def __init__(self, file, label, data=None, model=RandomForestClassifier(min_samples_leaf=10, n_jobs=-1), real_block=5, virtual_block=5, n_clust=0, n_ignore=3):
        self.file_initparams = {'file': file, 'label': label, 'model': {'type': type(model).__name__, 'params': model.get_params()}, 'real_block': real_block, 'virtual_block': virtual_block, 'n_clust': n_clust, 'n_ignore': n_ignore}
        if data is not None:
            self.data = load_file(file)
        else:
            self.data = data
        self.process_data_()
        columns_no_label = list(self.data.columns)[:]
        columns_no_label.remove(label)
        block = list(normalize_block(virtual_block, self.data.shape[0]))
        generators = []
        for a, b in zip([0] + block, block + [self.data.shape[0]]):
            generators.append(lambda n_datapoints, a=a, b=b: self.data[columns_no_label].values[np.random.choice(range(a, b), size=n_datapoints, replace=True)])
        block_boundaries = normalize_block(real_block, self.data.shape[0])
        block = (np.arange(self.data.shape[0])[:, None] >= block_boundaries[None, :]).sum(axis=1).reshape(-1, 1)
        X, y = (self.data[columns_no_label].values, self.data[label].values)
        if n_clust > 0:
            y = add_flip(block, X, y, (n_ignore + 1) * real_block * n_clust, n_ignore * real_block)
        model.fit(np.hstack((block, X)), y)
        self.score = {score: float(get_scorer(score)(model, np.hstack((block, X)), y)) for score in ['accuracy', 'balanced_accuracy', 'neg_log_loss', 'roc_auc_ovr']}
        block_perms = [np.random.permutation(block) for _ in range(50)]
        self.pfi = {'mean': {}, 'std': {}, 'scores': {score: [] for score in ['accuracy', 'balanced_accuracy', 'neg_log_loss', 'roc_auc_ovr']}}
        for pblock in block_perms:
            for score, perm_scores in self.pfi['scores'].items():
                perm_scores.append(get_scorer(score)(model, np.hstack((pblock, X)), y))
        for score, perm_scores in self.pfi['scores'].items():
            self.pfi['mean'][score] = float(self.score[score] - np.mean(perm_scores))
            self.pfi['std'][score] = float(np.std(perm_scores))
            self.pfi['scores'][score] = list(map(float, perm_scores))
        labeling = []
        for i in range(block_boundaries.shape[0]):
            labeling.append(lambda X, i=i: sample(model.predict_proba(np.hstack((i * np.ones((X.shape[0], 1)), X)))))
        super().__init__(generators, labeling)

    def process_data_(self):
        pass

    def size(self):
        return self.data.shape[0]

    def get_info(self):
        return {**super().get_info(), **self.file_initparams, 'size': self.size(), 'score': self.score, 'real drift score': self.pfi, 'columns': list(self.data.columns)}

class CreditCardDataSource(FileDataSource):

    def __init__(self, model=ExtraTreesClassifier(min_samples_leaf=3, n_jobs=-1), real_block=2, virtual_block=3, n_clust=0):
        super().__init__('stream_ds/creditcard.csv.xz', 'Class', model=model, real_block=real_block, virtual_block=virtual_block, n_clust=n_clust)

    def process_data_(self):
        columns = list(self.data.columns)
        columns.remove('Time')
        columns.remove('Unnamed: 0')
        self.data = self.data[columns]

class ElectricityDataSource(FileDataSource):

    def __init__(self, model=RandomForestClassifier(min_samples_leaf=10, n_jobs=-1), real_block=3, virtual_block=2, n_clust=0):
        super().__init__('stream_ds/electricity.csv.xz', 'class', model=model, real_block=real_block, virtual_block=virtual_block, n_clust=n_clust)

    def process_data_(self):
        columns = list(self.data.columns)
        columns.remove('date')
        columns.remove('Unnamed: 0')
        self.data = self.data[columns]
        self.data['class'] = LabelEncoder().fit_transform(self.data['class'])

class ForestCoverTypeDataSource(FileDataSource):

    def __init__(self, model=ExtraTreesClassifier(min_samples_leaf=3, n_jobs=-1), real_block=3, virtual_block=2, n_clust=0):
        super().__init__('stream_ds/forestcovertype.csv.xz', 'Cover_Type', model=model, real_block=real_block, virtual_block=virtual_block, n_clust=n_clust)

    def process_data_(self):
        columns = list(self.data.columns)
        columns.remove('Id')
        columns.remove('Unnamed: 0')
        self.data = self.data[columns]

class HttpDataSource(FileDataSource):

    def __init__(self, model=ExtraTreesClassifier(min_samples_leaf=3, n_jobs=-1), real_block=3, virtual_block=2, n_clust=0):
        super().__init__('stream_ds/kdd99_http.csv.xz', 'service', model=model, real_block=real_block, virtual_block=virtual_block, n_clust=n_clust)

    def process_data_(self):
        columns = list(self.data.columns)
        columns.remove('Unnamed: 0')
        self.data = self.data[columns]

class WeatherDataSource(FileDataSource):

    def __init__(self, model=ExtraTreesClassifier(min_samples_leaf=3, n_jobs=-1), real_block=3, virtual_block=3, n_clust=0):
        super().__init__('stream_ds/neweather.csv.xz', 'y', model=model, real_block=real_block, virtual_block=virtual_block, n_clust=n_clust)

    def process_data_(self):
        columns = list(self.data.columns)
        columns.remove('Unnamed: 0')
        self.data = self.data[columns]
datasources = {'SEA': SEADataSource, 'Agrawal': AgrawalDataSource, 'Mixed': MixedDataSource, 'Sine': SineDataSource, 'RandomTree': RandomTreeDataSource, 'Electricity': ElectricityDataSource, 'CreditCard': CreditCardDataSource, 'ForestCover': ForestCoverTypeDataSource, 'Http': HttpDataSource, 'Weather': WeatherDataSource, 'ElectricityFlip': lambda: ElectricityDataSource(n_clust=20), 'CreditCardFlip': lambda: CreditCardDataSource(n_clust=20), 'ForestCoverFlip': lambda: ForestCoverTypeDataSource(n_clust=20), 'HttpFlip': lambda: HttpDataSource(n_clust=20), 'WeatherFlip': lambda: WeatherDataSource(n_clust=20)}
