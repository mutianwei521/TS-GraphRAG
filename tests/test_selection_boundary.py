import unittest
import numpy as np
from experiments.node_selection.run import split_nodes, select_fusion, select_masked, evaluate_frozen, subset_features


class SelectionTests(unittest.TestCase):
    def features(self):
        x = np.zeros((4, 9, 8, 2))
        x[:, :, 6, 0] = 1
        x[1::2, :, 6, :] = [0, 1]
        return dict(X=x, A=np.zeros((4, 9, 2, 2)), MO=np.ones((4, 9)), RMIN=1-x[:, :, 6, :],
                    truth=np.array([0, 1, 0, 1]), dev=np.array([True, False, True, False]),
                    seed=np.array([41, 41, 141, 141]), qidx=np.array([0, 1, 0, 1]),
                    node=np.array(['a', 'b', 'a', 'b']), labels=np.arange(2), protos=np.array([f'H{i}' for i in range(9)]))

    def test_split_has_one_dev_and_one_test_node_per_partition(self):
        fps = [dict(leak_partition=p, leak_node=n) for p in range(15) for n in [f'{p}a', f'{p}b']]
        dev, test = split_nodes(fps)
        self.assertEqual(len(dev), 15)
        self.assertEqual(len(test), 15)
        self.assertFalse(dev & test)

    def test_poisoned_test_rows_do_not_change_selected_fusion_configuration(self):
        fp = self.features()
        config, _ = select_fusion(fp, 8)
        fp['truth'][~fp['dev']] = 999
        fp['X'][~fp['dev']] = 1e30
        other, _ = select_fusion(fp, 8)
        self.assertEqual(config, other)

    def test_frozen_replay_scores_only_test_rows(self):
        fp = self.features()
        conf = {'weights_channels': dict(zip(['prior','centroid','rank','fingerprint','semantic','community','projection','cosine_sq_max'], [0,0,0,0,0,0,1,0])), 'beta': 0.}
        rows = evaluate_frozen(fp, conf)
        self.assertTrue(all(r['test_top1_selected'] == 1. for r in rows))
        fp['truth'][fp['dev']] = 999
        self.assertEqual(rows, evaluate_frozen(fp, conf))

    def test_limit_keeps_each_seed_each_split(self):
        fp = self.features()
        fp = subset_features(fp, 1)
        self.assertEqual(fp['X'].shape[0], 4)
        self.assertEqual(int(fp['dev'].sum()), 2)

    def test_masked_selection_does_not_consult_test_rows(self):
        fp = self.features()
        ms = dict(PRJ=np.repeat(fp['RMIN'][:, :, None, :], 4, axis=2),
                  COS=np.repeat(fp['X'][:, :, 6, None, :], 4, axis=2), taus=np.array([0.,1.,2.,3.]))
        norms = np.ones(4)
        conf, _ = select_masked(fp, ms, norms, 8)
        fp['truth'][~fp['dev']] = 999
        ms['PRJ'][~fp['dev']] = 1e20
        other, _ = select_masked(fp, ms, norms, 8)
        self.assertEqual(conf, other)


if __name__ == '__main__': unittest.main()
