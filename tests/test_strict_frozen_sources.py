import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.strict_engine.engine import load_sources


class FrozenSourceCompatibilityTests(unittest.TestCase):
    def test_changed_document_text_cannot_reuse_frozen_vectors(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); data = base/'reference'; frozen = base/'frozen'; data.mkdir(); frozen.mkdir()
            fp = {'sensor_nodes': ['s'], 'fingerprints': [{'scenario_id':'a', 'sensor_fingerprint':[-1.]}]}
            original = [{'scenario_id':'a', 'summary_text':'audited text'}]
            changed = [{'scenario_id':'a', 'summary_text':'different text'}]
            for path, obj in [(data/'sensor_fingerprints.json', fp), (data/'scenario_summaries.json', changed),
                              (frozen/'frozen_scenario_documents.json', original)]:
                path.write_text(json.dumps(obj), encoding='utf-8')
            np.savez(frozen/'frozen_scenario_vectors.npz', ids=np.array(['a']), vectors=np.array([[1.,2.]]))
            with self.assertRaisesRegex(ValueError, 'Frozen document incompatibility'):
                load_sources(data, frozen)

    def test_reordered_vector_ids_are_rejected_before_pool_construction(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); data=base/'reference'; frozen=base/'frozen'; data.mkdir(); frozen.mkdir()
            fps={'sensor_nodes':['s'],'fingerprints':[{'scenario_id':'a'},{'scenario_id':'b'}]}
            docs=[{'scenario_id':sid,'summary_text':sid} for sid in ['a','b']]
            for path,obj in [(data/'sensor_fingerprints.json',fps),(data/'scenario_summaries.json',docs),
                             (frozen/'frozen_scenario_documents.json',docs)]: path.write_text(json.dumps(obj),encoding='utf-8')
            np.savez(frozen/'frozen_scenario_vectors.npz', ids=np.array(['b','a']),vectors=np.ones((2,2)))
            with self.assertRaisesRegex(ValueError,'IDs/order'):
                load_sources(data,frozen)


if __name__ == '__main__': unittest.main()
