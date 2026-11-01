"""Seed-controlled synthetic stream generation and holdout evaluation."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import importlib
import numpy as np
step23 = importlib.import_module('unseen_stream_pipeline')
gen_data = step23.gen_data
from river.datasets.synth import SEA as RiverSEA, Agrawal as RiverAgrawal, Mixed as RiverMixed, Sine as RiverSine, RandomTree as RiverRandomTree
ORIGINAL_GENERATE_ONE_STREAM = step23.generate_one_stream

def derive_seed(*values):
    values = [int(v) for v in values]
    sequence = np.random.SeedSequence(values)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

class DeterministicTakeWrapper:

    def __init__(self, dataset, master_seed, seed_attribute):
        self._dataset = dataset
        self._seed_attribute = seed_attribute
        self._rng = np.random.default_rng(int(master_seed))

    def take(self, k):
        subseed = int(self._rng.integers(1, 2 ** 31 - 1))
        setattr(self._dataset, self._seed_attribute, subseed)
        return self._dataset.take(k)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

def install_seeded_river_generators(stream_seed):
    stream_seed = int(stream_seed)

    def SEA_factory(*args, **kwargs):
        base_seed = derive_seed(stream_seed, 101)
        kwargs['seed'] = base_seed
        dataset = RiverSEA(*args, **kwargs)
        return DeterministicTakeWrapper(dataset, master_seed=derive_seed(stream_seed, 1001), seed_attribute='seed')

    def Agrawal_factory(*args, **kwargs):
        base_seed = derive_seed(stream_seed, 102)
        kwargs['seed'] = base_seed
        dataset = RiverAgrawal(*args, **kwargs)
        return DeterministicTakeWrapper(dataset, master_seed=derive_seed(stream_seed, 1002), seed_attribute='seed')

    def Mixed_factory(*args, **kwargs):
        base_seed = derive_seed(stream_seed, 103)
        kwargs['seed'] = base_seed
        dataset = RiverMixed(*args, **kwargs)
        return DeterministicTakeWrapper(dataset, master_seed=derive_seed(stream_seed, 1003), seed_attribute='seed')

    def Sine_factory(*args, **kwargs):
        base_seed = derive_seed(stream_seed, 104)
        kwargs['seed'] = base_seed
        dataset = RiverSine(*args, **kwargs)
        return DeterministicTakeWrapper(dataset, master_seed=derive_seed(stream_seed, 1004), seed_attribute='seed')

    def RandomTree_factory(*args, **kwargs):
        tree_seed = kwargs.get('seed_tree', 0)
        sample_seed = derive_seed(stream_seed, 105, int(tree_seed))
        kwargs['seed_sample'] = sample_seed
        dataset = RiverRandomTree(*args, **kwargs)
        return DeterministicTakeWrapper(dataset, master_seed=derive_seed(stream_seed, 1005, int(tree_seed)), seed_attribute='seed_sample')
    gen_data.SEA = SEA_factory
    gen_data.Agrawal = Agrawal_factory
    gen_data.Mixed = Mixed_factory
    gen_data.Sine = Sine_factory
    gen_data.RandomTree = RandomTree_factory

def generate_one_stream_v2(dataset, seed):
    install_seeded_river_generators(seed)
    result = ORIGINAL_GENERATE_ONE_STREAM(dataset, seed)
    if result['generation_ok']:
        result['generator_version'] = 'V2_explicit_river_subseeds'
    return result
step23.generate_one_stream = generate_one_stream_v2
if __name__ == '__main__':
    print('=' * 150)
    print('synthetic-stream pipeline')
    print('Explicit reproducible River stream-generation control')
    print('=' * 150)
    step23.main()
