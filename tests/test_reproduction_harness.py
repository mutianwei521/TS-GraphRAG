import json
from pathlib import Path
import pytest
from scripts.verify_outputs import verify_assets, compare_csv, compare_predictions


def test_asset_verification_detects_changed_input(tmp_path):
    import hashlib
    (tmp_path/'input.txt').write_text('frozen')
    manifest=tmp_path/'assets.json'
    manifest.write_text(json.dumps({'assets':[{'path':'input.txt','sha256':hashlib.sha256(b'frozen').hexdigest(),'bytes':6}]}))
    assert verify_assets(tmp_path,manifest)['checked']==1
    (tmp_path/'input.txt').write_text('edited')
    with pytest.raises(AssertionError,match='input.txt'):
        verify_assets(tmp_path,manifest)


def test_csv_verification_matches_by_key_and_rejects_metric_change(tmp_path):
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    a.write_text('protocol,seed,top1_pct\nH0,41,88.0\nH2,41,76.0\n')
    b.write_text('protocol,seed,top1_pct\nH2,41,76.0\nH0,41,88.0\n')
    assert compare_csv(a,b,keys=['protocol','seed'],atol=1e-10)['rows']==2
    b.write_text('protocol,seed,top1_pct\nH2,41,76.1\nH0,41,88.0\n')
    with pytest.raises(AssertionError,match='top1_pct'):
        compare_csv(a,b,keys=['protocol','seed'],atol=1e-10)


def test_compressed_prediction_comparison(tmp_path):
    import gzip
    actual=tmp_path/'actual.csv.gz';expected=tmp_path/'expected.csv'
    expected.write_text('query,correct\na,1\n')
    with gzip.open(actual,'wt',encoding='utf-8') as stream:stream.write(expected.read_text())
    assert compare_csv(actual,expected,keys=['query'])['rows']==1


def test_raw_network_is_part_of_asset_verification(tmp_path):
    import hashlib
    data=tmp_path/'data';data.mkdir()
    inp=data/'Exa7.inp';inp.write_text('network')
    asset=tmp_path/'sample.txt';asset.write_text('frozen')
    manifest=tmp_path/'assets.json'
    manifest.write_text(json.dumps({'network_sha256':hashlib.sha256(b'network').hexdigest(),
        'assets':[{'path':'sample.txt','sha256':hashlib.sha256(b'frozen').hexdigest(),'bytes':6}]}))
    verify_assets(tmp_path,manifest)
    inp.write_text('modified')
    with pytest.raises(AssertionError,match='Exa7.inp'):verify_assets(tmp_path,manifest)


def test_prediction_audit_rejects_changed_or_duplicated_query(tmp_path):
    actual=tmp_path/'actual.csv';expected=tmp_path/'expected.csv'
    expected.write_text('seed,protocol,query_index,method,predicted_partition,true_partition,correct\n41,H2,0,projection,2,2,1\n')
    actual.write_text(expected.read_text())
    assert compare_predictions(actual,expected)['predictions']==1
    actual.write_text('seed,protocol,query_index,method,predicted_partition,true_partition,correct\n41,H2,0,projection,3,2,0\n')
    with pytest.raises(AssertionError,match='prediction'):compare_predictions(actual,expected)
    actual.write_text(expected.read_text()+expected.read_text().splitlines()[1]+'\n')
    with pytest.raises(AssertionError,match='Duplicate'):compare_predictions(actual,expected)
