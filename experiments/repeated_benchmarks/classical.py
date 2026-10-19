"""Fit the original E4 classical models across five fixed noise/model seeds."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import concurrent.futures as cf
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from .primitives import projection_residual

SOURCE = ROOT / 'data/reference/sensor_fingerprints.json'
CHECKPOINT = ROOT / 'data/frozen/historical/expH_gnn2_results.json.partial'
SEEDS = [41, 581, 1121, 1661, 2201]
MAIN_PROTOCOLS = ['H0', 'H2', 'H3', 'H4', 'H8']
ALL_PROTOCOLS = ['H' + str(i) for i in range(10)]
METHODS = ['projection', 'cosine 1-NN', 'k-NN (k=5)', 'Random Forest',
           'SVM (RBF)', 'MLP', 'Nearest Centroid', 'Logistic Regression']
_DATA = None


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def load_data():
    global _DATA
    if _DATA is None:
        fps = json.loads(SOURCE.read_text(encoding='utf-8'))['fingerprints']
        X = np.array([r['sensor_fingerprint'] for r in fps], dtype=float)
        y = np.array([r['leak_partition'] for r in fps])
        assert X.shape == (540, 30) and np.isfinite(X).all()
        checkpoint = json.loads(CHECKPOINT.read_text())
        qidx = checkpoint['qidx']
        assert len(qidx) == len(set(qidx)) == 90
        assert checkpoint['done_idx'] == qidx
        assert all(sum(y[qidx] == k) == 6 for k in range(15))
        _DATA = fps, X, y, qidx
    return _DATA


def retain(qi, proto):
    fps, X, y, qidx = load_data()
    q = fps[qi]
    sn = np.array([a['leak_node'] == q['leak_node'] for a in fps])
    sq = np.array([a['leak_rate_Ls'] == q['leak_rate_Ls'] for a in fps])
    st = np.array([a['demand_period'] == q['demand_period'] for a in fps])
    sp = y == y[qi]
    mask = {'H0': np.ones(len(fps), dtype=bool), 'H1': ~(sn & sq & st),
            'H2': ~sn, 'H3': ~sq, 'H4': ~st, 'H5': ~(sn | sq),
            'H6': ~(sn | st), 'H7': ~(sq | st), 'H8': ~(sn | sq | st),
            'H9': ~sp}[proto]
    return np.flatnonzero(mask)


def noise_seed(run_seed, qi):
    # Distinct 540-seed blocks across repeats; seed 41 reproduces the legacy query noise.
    return run_seed + qi


def query_value(run_seed, qi):
    _, X, _, _ = load_data()
    return X[qi] + np.random.RandomState(noise_seed(run_seed, qi)).normal(0, 0.02, X.shape[1])


def make_models(run_seed):
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=run_seed, n_jobs=1),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=run_seed)),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64),
                             max_iter=400, early_stopping=True, random_state=run_seed)),
        'Nearest Centroid': make_pipeline(StandardScaler(), NearestCentroid()),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                     random_state=run_seed)),
    }


def configs(run_seed):
    return {name: mdl.get_params(deep=True) for name, mdl in make_models(run_seed).items()}


def serializable_configs(run_seed):
    return json.loads(json.dumps(configs(run_seed), default=str))


def source_metadata():
    paths = [SOURCE, CHECKPOINT, Path(__file__), Path(__file__).with_name('primitives.py')]
    return {'source_sha256': {str(p.relative_to(ROOT)): digest_bytes(p.read_bytes()) for p in paths},
            'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'platform': platform.platform(),
            'noise_rule': 'RandomState(actual_run_seed + original_query_index).normal(0, 0.02, 30)',
            'model_seeds': SEEDS, 'thread_limits': 1,
            'configs_by_seed': {str(s): serializable_configs(s) for s in SEEDS}}


def task_key(seed, idx):
    # Exact retained index order controls the fit and nearest-neighbour tie handling.
    return 's' + str(seed) + '_' + digest_bytes(np.asarray(idx, dtype='<i8').tobytes())[:20]


def build_tasks(qindices, protocols, seeds=SEEDS):
    grouped = {}
    for seed in seeds:
        for proto in protocols:
            for qi in qindices:
                idx = retain(qi, proto).tolist()
                key = task_key(seed, idx)
                grouped.setdefault(key, {'key': key, 'seed': seed, 'pool_indices': idx, 'pairs': []})
                grouped[key]['pairs'].append([proto, qi])
    return list(grouped.values())


def eval_task(task):
    t0 = time.perf_counter()
    fps, X, y, _ = load_data()
    idx = np.asarray(task['pool_indices'], dtype=int)
    seed = task['seed']
    qis = sorted({qi for proto, qi in task['pairs']})
    if not idx.size or len(set(y[idx])) < 2:
        raise ValueError('Empty or single-class pool: ' + task['key'])
    queries = np.array([query_value(seed, qi) for qi in qis])
    result = {'key': task['key'], 'binding': task.get('binding'), 'seed': seed, 'pool_indices': idx.tolist(),
              'pool_sha256': digest_bytes(np.asarray(idx, dtype='<i8').tobytes()),
              'query_indices': qis, 'predictions': {}, 'models': {}, 'fit_errors': []}
    D = X[idx]
    result['predictions']['projection'] = [int(y[idx[np.argmin(projection_residual(v, D))]]) for v in queries]
    nrm = np.linalg.norm(D, axis=1)
    result['predictions']['cosine 1-NN'] = []
    for v in queries:
        cs = np.where(nrm > 1e-12, (D @ v) / np.maximum(nrm * np.linalg.norm(v), 1e-18), -np.inf)
        result['predictions']['cosine 1-NN'].append(int(y[idx[np.argmax(cs)]]))

    for name, model in make_models(seed).items():
        mt = time.perf_counter()
        rec = {'config_sha256': digest_json(json.loads(json.dumps(model.get_params(deep=True), default=str))),
               'warnings': [], 'stderr': '', 'fit_seconds': None}
        stderr = io.StringIO()
        captured_splits = []
        # Record the exact internal MLP split without changing its algorithm or RNG calls:
        # append row indices as one additional array, then strip only those two outputs.
        import sklearn.neural_network._multilayer_perceptron as mlpmod
        original_split = mlpmod.train_test_split
        def recording_split(*arrays, **kw):
            out = original_split(*arrays, np.arange(len(arrays[0])), **kw)
            captured_splits.append({'training_source_indices': idx[out[-2]].tolist(),
                                    'validation_source_indices': idx[out[-1]].tolist()})
            return out[:-2]
        try:
            with warnings.catch_warnings(record=True) as seen, contextlib.redirect_stderr(stderr), threadpool_limits(1):
                warnings.simplefilter('always')
                if name == 'MLP':
                    mlpmod.train_test_split = recording_split
                try:
                    model.fit(D, y[idx])
                finally:
                    mlpmod.train_test_split = original_split
                pred = model.predict(queries)
                result['predictions'][name] = [int(a) for a in pred]
            rec['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in seen]
            if name == 'MLP':
                est = model[-1]
                rec.update(n_iter=int(est.n_iter_), best_validation_score=float(est.best_validation_score_),
                           validation_scores=[float(v) for v in est.validation_scores_],
                           internal_splits=captured_splits)
            if name == 'Logistic Regression':
                rec['n_iter'] = model[-1].n_iter_.tolist()
        except Exception as exc:
            result['predictions'][name] = [None] * len(qis)
            result['fit_errors'].append({'method': name, 'type': type(exc).__name__,
                                         'message': str(exc), 'traceback': traceback.format_exc()})
        finally:
            mlpmod.train_test_split = original_split
            rec['stderr'] = stderr.getvalue()
            rec['fit_seconds'] = time.perf_counter() - mt
        result['models'][name] = rec
    result['wall_seconds'] = time.perf_counter() - t0
    return result


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def cache_valid(cached, task, binding):
    """Reuse a pool fit only for the same inputs, implementation and complete query set."""
    return (cached.get('key') == task['key'] and cached.get('binding') == binding
            and set(cached.get('query_indices', [])) >= {qi for _, qi in task['pairs']})


def run(scope, workers, output_dir=None, max_pools=None):
    global HERE
    if output_dir is not None:
        HERE = Path(output_dir).resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fps, X, y, qidx = load_data()
    qis = qidx if scope == 'main90' else list(range(540))
    protocols = MAIN_PROTOCOLS if scope == 'main90' else ALL_PROTOCOLS
    tasks = build_tasks(qis, protocols)
    expected_full_pools = len(tasks)
    if max_pools is not None:
        if max_pools < 1: raise ValueError('max_pools must be positive')
        tasks = tasks[:max_pools]
    tag = scope + ('_smoke' if max_pools is not None else '')
    cache = HERE / 'classical_pool_cache'
    cache.mkdir(exist_ok=True)
    metadata = source_metadata()
    metadata['scope'] = scope
    metadata['query_indices'] = qis
    metadata['protocols'] = protocols
    metadata['worker_count'] = workers
    metadata['task_count'] = len(tasks)
    metadata['expected_full_pools'] = expected_full_pools
    metadata['profile'] = 'smoke' if max_pools is not None else 'full'
    binding = digest_json({k:metadata[k] for k in ('source_sha256','python','numpy','scipy','sklearn','configs_by_seed')})
    metadata['cache_binding'] = binding
    write_json(HERE / ('classical_' + tag + '_manifest.json'), metadata)
    pending = []
    for task in tasks:
        task['binding'] = binding
        p = cache / (task['key'] + '.json')
        if p.exists():
            cached = json.loads(p.read_text())
            if cache_valid(cached, task, binding):
                continue
        pending.append(task)
    print(f'{scope}: {len(tasks)} total seed/pools; {len(pending)} pending; {workers} workers', flush=True)
    done = len(tasks) - len(pending)
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(eval_task, t): t for t in pending}
        for future in cf.as_completed(futures):
            ans = future.result()
            write_json(cache / (ans['key'] + '.json'), ans)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f'{scope}: {done}/{len(tasks)} pools, elapsed {time.perf_counter()-started:.1f}s', flush=True)
    rows = []
    fit_errors = []
    warning_count = 0
    for task in tasks:
        ans = json.loads((cache / (task['key'] + '.json')).read_text())
        pos = {qi:j for j,qi in enumerate(ans['query_indices'])}
        fit_errors.extend([dict(pool_key=task['key'], **e) for e in ans['fit_errors']])
        warning_count += sum(len(rec['warnings']) for rec in ans['models'].values())
        for proto, qi in task['pairs']:
            for name in METHODS:
                pred = ans['predictions'][name][pos[qi]]
                rows.append({'seed': task['seed'], 'protocol': proto, 'query_index': qi,
                             'noise_seed': noise_seed(task['seed'], qi),
                             'scenario_id': fps[qi]['scenario_id'], 'leak_node': fps[qi]['leak_node'],
                             'true_partition': int(y[qi]), 'method': name, 'predicted_partition': pred,
                             'correct': None if pred is None else int(pred == y[qi]),
                             'pool_key': task['key'], 'pool_count': len(task['pool_indices'])})
    rowpath = HERE / ('classical_' + tag + '_predictions.csv')
    with rowpath.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if max_pools is not None:
        report = dict(study='repeated_benchmarks',profile='smoke',scope=scope,completed=False,
                      evaluated_pools=len(tasks),expected_full_pools=expected_full_pools,
                      predictions=len(rows),fit_errors=fit_errors,metadata=metadata,
                      wall_seconds=time.perf_counter()-started)
        write_json(HERE / ('classical_' + tag + '_results.json'), report)
        if fit_errors: raise RuntimeError('Smoke fit errors exist; inspect the pool/result audit files.')
        return report
    summaries = []
    for proto in protocols:
        for name in METHODS:
            rates = []
            counts = []
            for seed in SEEDS:
                selected = [r for r in rows if r['protocol']==proto and r['method']==name and r['seed']==seed]
                assert len(selected) == len(qis)
                n_fail = sum(r['correct'] is None for r in selected)
                counts.append({'seed':seed, 'correct':sum(r['correct'] or 0 for r in selected),
                               'n':len(selected), 'fit_failures':n_fail})
                rates.append(None if n_fail else 100 * sum(r['correct'] for r in selected) / len(selected))
            summaries.append({'protocol':proto,'method':name,'rates_percent':rates,'counts':counts,
                              'mean_percent':None if None in rates else float(np.mean(rates)),
                              'sample_std_percent':None if None in rates else float(np.std(rates,ddof=1)),
                              'n_repeats':5,'n_queries_per_repeat':len(qis),'sd_ddof':1})
    output = {'metadata':metadata,'summary':summaries,'fit_errors':fit_errors,
              'warning_count':warning_count,'predictions_csv':str(rowpath),
              'wall_seconds':time.perf_counter()-started}
    write_json(HERE / ('classical_' + scope + '_results.json'), output)
    with (HERE / ('classical_' + scope + '_summary.tsv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['protocol','method','mean_pct','sample_sd_pct','n','repeats',*[f'seed{s}' for s in SEEDS]])
        for r in summaries:writer.writerow([r['protocol'],r['method'],r['mean_percent'],r['sample_std_percent'],len(qis),5,*r['rates_percent']])
    print(json.dumps({'scope':scope,'completed':not fit_errors,'fit_errors':len(fit_errors),
                      'warnings':warning_count,'wall_seconds':output['wall_seconds']}),flush=True)
    if fit_errors:
        raise RuntimeError('Fit errors exist; affected means are null. Inspect result and pool audit files.')
