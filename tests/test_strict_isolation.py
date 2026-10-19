import copy
import hashlib
import unittest
import numpy as np

from experiments.strict_engine.engine import (Pool, keep, FrozenEmbeddings, build_query_text, validate_observations, SEEDS)
from experiments.strict_engine.run import grid


class StrictIsolationTests(unittest.TestCase):
    def setUp(self):
        self.fps = [dict(scenario_id=str(i), leak_node=f'n{i}', leak_rate_Ls=1+i, demand_period='day',
                         leak_partition=i % 2, sensor_fingerprint=[float(i+1), -float(i+1)],
                         max_pressure_drop=1, num_affected_nodes=1) for i in range(4)]
        self.docs = {f['scenario_id']: dict(f, summary_text=f"scenario {f['scenario_id']}") for f in self.fps}
        self.vec = {str(i): [float(i), 1.] for i in range(4)}
        self.topo = {'partitions': {'0': {'node_count': 3}, '1': {'node_count': 4}}}

    def pool(self, fps, docs, vec, proto='H2'):
        sub = [f for f in fps if keep(f, fps[0], proto)]
        ids = [f['scenario_id'] for f in sub]
        return Pool(sub, {s: docs[s] for s in ids}, {s: vec[s] for s in ids}, self.topo, [[0, 1], [1, 0]])

    def test_h0_keeps_query_node_and_h2_excludes_it(self):
        self.assertTrue(keep(self.fps[0], self.fps[0], 'H0'))
        self.assertFalse(keep(self.fps[0], self.fps[0], 'H2'))

    def test_excluded_record_poison_does_not_change_retained_artifacts(self):
        original = self.pool(self.fps, self.docs, self.vec)
        fps, docs, vec = copy.deepcopy((self.fps, self.docs, self.vec))
        fps[0]['sensor_fingerprint'] = [1e30, 1e30]
        docs['0']['summary_text'] = 'POISON'
        docs['0']['max_pressure_drop'] = 1e30
        vec['0'] = [1e30, 1e30]
        changed = self.pool(fps, docs, vec)
        self.assertEqual(original.artifact_digest, changed.artifact_digest)
        for p in (original, changed):
            p.set_embeddings({t: [1., 0.] for t in p.summary_texts.values()})
        a = original.score(original.prepare(np.array([-.3, .2]), 'day', np.array([1., 0.]), np.array([0, 1])))
        b = changed.score(changed.prepare(np.array([-.3, .2]), 'day', np.array([1., 0.]), np.array([0, 1])))
        self.assertEqual(a[:2], b[:2])

    def test_summary_cache_uses_exact_retained_text_sha(self):
        txt = 'retained summary'
        emb = FrozenEmbeddings({hashlib.sha256(txt.encode()).hexdigest(): np.array([1., 2.])})
        np.testing.assert_array_equal(emb[txt], [1., 2.])
        with self.assertRaisesRegex(ValueError, 'SHA'):
            emb['full corpus summary']

    def test_frozen_observations_reject_changed_fingerprint(self):
        fps = [self.fps[0]]
        arr = dict(seed=np.array(SEEDS), query_index=np.zeros(5, int),
                   observations=np.array([np.array(fps[0]['sensor_fingerprint']) + np.random.default_rng(s).normal(0, .02, (1, 2))[0] for s in SEEDS]),
                   embeddings=np.zeros((5, 2)))
        validate_observations(arr, fps, ['a', 'b'])
        changed = copy.deepcopy(fps)
        changed[0]['sensor_fingerprint'][0] += .5
        with self.assertRaisesRegex(ValueError, 'observations'):
            validate_observations(arr, changed, ['a', 'b'])

    def test_query_text_depends_only_on_observations_and_period(self):
        actual = build_query_text({'a': -.4, 'b': -.05}, 'day', {'a': 0, 'b': 1})
        self.assertEqual(actual, 'A small leak during day. Max drop 0.400m at node a (P#0). Sensors: Sensor a (P#0): -0.400m, Sensor b (P#1): -0.050m. Affected partitions: [0].')

    def test_all_prespecified_sensitivity_settings_present(self):
        self.assertEqual(len(grid()), 34)


class StrictCheckpointTests(unittest.TestCase):
    def test_local_checkpoint_requires_original_file_hashes(self):
        import tempfile
        from pathlib import Path
        from experiments.strict_engine.engine import verify_model_checkpoint
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            (model/'config.json').write_text('fixed', encoding='utf-8')
            expected = {'config.json': hashlib.sha256(b'fixed').hexdigest()}
            self.assertEqual(verify_model_checkpoint(model, expected), expected)
            (model/'config.json').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                verify_model_checkpoint(model, expected)


if __name__ == '__main__':
    unittest.main()
