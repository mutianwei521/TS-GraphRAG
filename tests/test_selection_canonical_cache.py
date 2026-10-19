import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.node_selection import run as ns
from experiments.strict_engine.engine import SEEDS, DEFAULT_DATA_DIR


def canonical_fixture():
    fps=[dict(leak_partition=p,leak_node=f'{p}{n}') for p in range(2) for n in ['a','b']]
    n=len(fps); rows=len(SEEDS)*n; qi=np.tile(np.arange(n),len(SEEDS)); nodes=np.array([fps[i]['leak_node'] for i in qi])
    dev,_=ns.split_nodes(fps)
    fp=dict(X=np.zeros((rows,9,8,2)),A=np.zeros((rows,9,2,2)),MO=np.ones((rows,9)),RMIN=np.ones((rows,9,2)),
            truth=np.array([fps[i]['leak_partition'] for i in qi]),node=nodes,seed=np.repeat(SEEDS,n),qidx=qi,
            dev=np.array([node in dev for node in nodes]),protos=np.array(ns.PROTOS),channels=np.array(ns.CH),labels=np.arange(2))
    return fps,fp


class SelectionCanonicalGuardTests(unittest.TestCase):
    def test_truncated_duplicated_grid_is_rejected(self):
        fps,fp=canonical_fixture(); bad=ns.slice_rows(fp,[0,0,1,1])
        with self.assertRaisesRegex(ValueError,'canonical|grid|shape'):ns.validate_features(bad,fps)

    def test_full_length_duplicate_identity_is_rejected(self):
        fps,fp=canonical_fixture()
        for key in ['truth','node','seed','qidx','dev']:fp[key][1]=fp[key][0]
        with self.assertRaisesRegex(ValueError,'canonical|grid|identity'):ns.validate_features(fp,fps)

    def test_channel_and_label_order_are_required(self):
        for key in ['channels','labels']:
            with self.subTest(key=key):
                fps,fp=canonical_fixture();fp[key]=fp[key][::-1]
                with self.assertRaisesRegex(ValueError,'channel|label'):ns.validate_features(fp,fps)

    def test_shape_and_nonfinite_features_are_rejected(self):
        for kind in ['shape','NaN','infinity']:
            with self.subTest(kind=kind):
                fps,fp=canonical_fixture()
                if kind=='shape':fp['X']=fp['X'][:,:,:,:1]
                else:fp['X'][0,0,0,0]=np.nan if kind=='NaN' else np.inf
                with self.assertRaisesRegex(ValueError,'shape|finite'):ns.validate_features(fp,fps)

    def test_missing_channel_schema_is_rejected(self):
        fps,fp=canonical_fixture();del fp['channels']
        with self.assertRaisesRegex(ValueError,'schema|channel'):ns.validate_features(fp,fps)

    def test_fractional_query_indices_are_rejected(self):
        fps,fp=canonical_fixture();fp['qidx']=fp['qidx'].astype(float);fp['qidx'][0]=.5
        with self.assertRaisesRegex(ValueError,'integer|identity|grid'):ns.validate_features(fp,fps)

    def test_canonical_residual_eligibility_sentinel_is_allowed(self):
        fps,fp=canonical_fixture();fp['RMIN'][0,0,0]=np.inf
        ns.validate_features(fp,fps)
        fp['RMIN'][0,0,0]=-np.inf
        with self.assertRaisesRegex(ValueError,'residual|sentinel|finite'):ns.validate_features(fp,fps)

    def test_full_grid_is_valid_before_balanced_limit(self):
        fps,fp=canonical_fixture();ns.validate_features(fp,fps)
        limited=ns.subset_features(fp,1)
        self.assertEqual(len(limited['seed']),10)
        self.assertEqual(len({(int(s),int(q)) for s,q in zip(limited['seed'],limited['qidx'])}),10)


class SelectionMaskedGuardTests(unittest.TestCase):
    def masked(self):
        fps,fp=canonical_fixture();n=len(fp['truth'])
        return fps,dict(PRJ=np.ones((n,9,4,2)),COS=np.zeros((n,9,4,2)),truth=fp['truth'],
                        seed=fp['seed'],qidx=fp['qidx'],taus=np.array(ns.TAUS),labels=fp['labels'])

    def test_masked_duplicate_truncated_grid_is_rejected(self):
        fps,ms=self.masked();ms=ns.slice_rows(ms,[0,0,1,1])
        with self.assertRaisesRegex(ValueError,'canonical|Canonical|grid'):ns.validate_masked(ms,fps)

    def test_masked_reversed_labels_are_rejected(self):
        fps,ms=self.masked();ms['labels']=ms['labels'][::-1]
        with self.assertRaisesRegex(ValueError,'label'):ns.validate_masked(ms,fps)

    def test_masked_nan_and_wrong_signed_infinity_are_rejected(self):
        for key,value in [('PRJ',np.nan),('PRJ',-np.inf),('COS',np.nan),('COS',np.inf)]:
            with self.subTest(key=key,value=value):
                fps,ms=self.masked();ms[key][0,0,0,0]=value
                with self.assertRaisesRegex(ValueError,'residual|cosine'):ns.validate_masked(ms,fps)

    def test_masked_matching_eligibility_sentinels_are_valid(self):
        fps,ms=self.masked();ms['PRJ'][0,0,0,0]=np.inf;ms['COS'][0,0,0,0]=-np.inf
        ns.validate_masked(ms,fps)
        ms['COS'][0,0,0,0]=0.
        with self.assertRaisesRegex(ValueError,'sentinels'):ns.validate_masked(ms,fps)


class SelectionRequiredParityTests(unittest.TestCase):
    def test_replay_primary_archive_mismatch_fails(self):self.bad_archive('test_results.json','test_top1_selected')
    def test_replay_masked_archive_mismatch_fails(self):self.bad_archive('test_masked_results.json','test_top1_frozen')

    def test_limited_replay_still_requires_full_cache_archive_parity(self):
        self.bad_archive('test_results.json','test_top1_selected',1)

    def bad_archive(self,name,key,limit=0):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);frozen=base/'frozen';frozen.mkdir()
            for src in ns.DEFAULT_SELECTION_FROZEN.iterdir():
                if src.is_file():shutil.copyfile(src,frozen/src.name)
            rows=json.loads((frozen/name).read_text());rows[0][key]=-1.
            (frozen/name).write_text(json.dumps(rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'[Pp]arity|archive'):
                ns.main(['--stage','replay','--selection-frozen-dir',str(frozen),'--output-dir',str(base/'out'),'--limit',str(limit)])
            self.assertFalse((base/'out'/'run_completion.json').exists())


class SelectionPreSubsetTests(unittest.TestCase):
    def test_limit_cannot_hide_nan_in_unselected_cache_row(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);frozen=base/'frozen';frozen.mkdir()
            fp=ns.load_npz(ns.DEFAULT_SELECTION_FROZEN/'features.npz');fp['X'][-1,0,0,0]=np.nan
            np.savez_compressed(frozen/'features.npz',**fp)
            with self.assertRaisesRegex(ValueError,'finite'):
                ns.main(['--stage','replay','--limit','1','--selection-frozen-dir',str(frozen),'--output-dir',str(base/'out')])


if __name__ == '__main__':unittest.main()
