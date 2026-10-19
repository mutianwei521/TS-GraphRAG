"""Developer/test node selection: feature rebuild, fixed search and frozen replay."""
from __future__ import annotations
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments.strict_engine.engine import (SEEDS, DEFAULT_DATA_DIR, DEFAULT_FROZEN_DIR, load_sources,
    load_observations, build_pools, FrozenEmbeddings)
from experiments.strict_engine.run import save_json, filehash

PROTOS = [f'H{i}' for i in range(9)]
CH = ['prior', 'centroid', 'rank', 'fingerprint', 'semantic', 'community', 'projection', 'cosine_sq_max']
DEFAULT = {'strong': [.25, .12, .08, .30, .10, .15, 0., 0.], 'weak': [.20, .20, .15, .18, .10, .17, 0., 0.],
           'micro': [.15, .25, .20, .12, .10, .18, 0., 0.]}
TAUS = [0., 1., 2., 3.]
SIGMA = .02
ROW_KEYS = {'X', 'A', 'MO', 'RMIN', 'truth', 'node', 'seed', 'qidx', 'dev', 'PRJ', 'COS'}
DEFAULT_SELECTION_FROZEN = DEFAULT_FROZEN_DIR.parent / 'node_selection'


def split_nodes(fps):
    bypart = {}
    for f in fps: bypart.setdefault(f['leak_partition'], set()).add(f['leak_node'])
    dev, test = set(), set()
    for part in sorted(bypart):
        nodes = sorted(bypart[part])
        if len(nodes) != 2: raise ValueError(f'Partition {part} requires exactly two leak nodes, got {nodes}')
        dev.add(nodes[0]); test.add(nodes[1])
    return dev, test


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def slice_rows(fp, rows):
    return {k: v[rows] if k in ROW_KEYS else v for k, v in fp.items()}


def selected_rows(fp, limit):
    if not limit: return np.arange(len(fp['seed']))
    rows = []
    for seed in np.unique(fp['seed']):
        for isdev in [True, False]:
            rows.extend(np.flatnonzero((fp['seed'] == seed) & (fp['dev'] == isdev))[:limit])
    return np.array(sorted(rows), dtype=int)


def subset_features(fp, limit):
    return slice_rows(fp, selected_rows(fp, limit))


def regime_idx(mo):
    return np.where(mo >= 1., 0, np.where(mo >= .1, 1, 2))


def scores_default(fp):
    W = np.array([DEFAULT['strong'], DEFAULT['weak'], DEFAULT['micro']])
    s = np.einsum('rpc,rpck->rpk', W[regime_idx(fp['MO'])], fp['X'])
    return .85*s + .15*np.einsum('rpkl,rpl->rpk', fp['A'], s)


def scores_single(fp, w, beta):
    s = np.einsum('c,rpck->rpk', np.asarray(w, float), fp['X'])
    return (1-beta)*s + beta*np.einsum('rpkl,rpl->rpk', fp['A'], s)


def correctness(S, truth):
    return ((S.max(axis=2) > 0) & (np.argmax(S, axis=2) == truth[:, None])).astype(int)


def dev_mean(S, truth):
    return correctness(S, truth).mean(axis=0)[:9].mean()


def select_fusion(fp, n_draws=400, seed_rng=20261011):
    """Only development arrays enter search, including its intermediate scores."""
    d = slice_rows(fp, np.flatnonzero(fp['dev']))
    if not len(d['truth']): raise ValueError('No development rows available.')
    rng = np.random.default_rng(seed_rng)
    d_def = dev_mean(scores_default(d), d['truth'])
    d_proj = dev_mean(scores_single(d, [0,0,0,0,0,0,1,0], 0.), d['truth'])
    best_score, best = d_proj, ([0,0,0,0,0,0,1,0], 0.)
    results = [('default_fusion', float(d_def)), ('projection_only', float(d_proj))]
    if d_def > best_score: best_score, best = d_def, None
    for i in range(n_draws):
        w = rng.dirichlet(np.ones(8)*.7); beta = float(rng.choice([0., .15, .3]))
        sc = dev_mean(scores_single(d, w, beta), d['truth'])
        results.append((f'draw{i}', float(sc), [round(float(x), 4) for x in w], beta))
        if sc > best_score: best_score, best = sc, (w.round(4).tolist(), beta)
    w, beta = (DEFAULT['weak'], .15) if best is None else best
    frozen = dict(weights_channels=dict(zip(CH, w)) if best is not None else 'regime default', beta=beta,
                  selection=f'max mean dev top-1 over H0-H8 among {n_draws} random Dirichlet draws + default + projection',
                  dev_mean_top1_selected=float(best_score), dev_mean_top1_default=float(d_def), dev_mean_top1_projection_only=float(d_proj))
    return frozen, results


def mcnemar_exact(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return (1. if n01+n10 == 0 else float(binomtest(n01, n01+n10, .5).pvalue)), n01, n10


def evaluate_frozen(fp, config):
    t = slice_rows(fp, np.flatnonzero(~fp['dev']))
    if not len(t['truth']): raise ValueError('No test rows available.')
    S_sel = scores_default(t) if config['weights_channels'] == 'regime default' else scores_single(t, [config['weights_channels'][c] for c in CH], config['beta'])
    sel = correctness(S_sel, t['truth']); dfn = correctness(scores_default(t), t['truth'])
    prj = correctness(scores_single(t, [0,0,0,0,0,0,1,0], 0.), t['truth'])
    rows = []
    for p, proto in enumerate(PROTOS):
        pv, n01, n10 = mcnemar_exact(sel[:, p], prj[:, p])
        rows.append(dict(protocol=proto, test_top1_selected=float(sel[:,p].mean()), test_top1_default_fusion=float(dfn[:,p].mean()),
                         test_top1_projection_recomputed=float(prj[:,p].mean()), mcnemar_selected_vs_projection_p=pv,
                         discordant_selected_only=n01, discordant_projection_only=n10))
    return rows


def mask_channel(ms, norms, kind, ti):
    if kind == 'proj':
        r = ms['PRJ'][:, :, ti, :]
        v = np.where(np.isfinite(r), 1-r/np.maximum(norms[:, None, None], 1e-18), 0.)
    else: v = np.maximum(ms['COS'][:, :, ti, :], 0.)**2
    v = np.clip(v, 0, None); mx = v.max(axis=2, keepdims=True)
    return np.where(mx > 0, v/np.maximum(mx, 1e-18), 0.)


def fused(fp, w6, wm, beta, ch):
    s = np.einsum('c,rpck->rpk', np.concatenate([w6, [wm]]), np.concatenate([fp['X'][:, :, :6, :], ch[:, :, None, :]], axis=2))
    return (1-beta)*s + beta*np.einsum('rpkl,rpl->rpk', fp['A'], s)


def score_masked_candidate(fp, ms, norms, spec):
    ti = int(np.flatnonzero(ms['taus'] == spec['tau'])[0])
    if spec['kind'] == 'fusion': return fused(fp, np.array(spec['w6']), spec['wm'], spec['beta'], mask_channel(ms, norms, 'proj', ti))
    return mask_channel(ms, norms, spec['kind'], ti)


def select_masked(fp, ms, norms, n_draws=400, seed_rng=20261011):
    rows = np.flatnonzero(fp['dev']); d = slice_rows(fp, rows); m = slice_rows(ms, rows); norms = norms[rows]
    rng = np.random.default_rng(seed_rng); cands = []
    for kind in ['proj', 'cos']:
        for ti, tau in enumerate(m['taus']):
            S = mask_channel(m, norms, kind, ti)
            cands.append((f'{kind}_tau{tau}', dev_mean(S, d['truth']), dict(kind=kind, tau=float(tau), beta=None)))
    for i in range(n_draws):
        ti = int(rng.integers(len(m['taus']))); w6 = rng.dirichlet(np.ones(6)*.7)
        wm = float(rng.dirichlet(np.ones(2)*.7)[0]); beta = float(rng.choice([0., .15, .3]))
        S = fused(d, w6, wm, beta, mask_channel(m, norms, 'proj', ti))
        cands.append((f'fusion_draw{i}', dev_mean(S, d['truth']), dict(kind='fusion', tau=float(m['taus'][ti]),
                      w6=w6.round(4).tolist(), wm=round(wm, 4), beta=beta)))
    cands.sort(key=lambda c: -c[1]); name, score, spec = cands[0]
    ref = dict(projection_tau0=float(dev_mean(mask_channel(m, norms, 'proj', 0), d['truth'])),
               cosine_tau0=float(dev_mean(mask_channel(m, norms, 'cos', 0), d['truth'])))
    frozen = dict(name=name, dev_mean_top1=float(score), spec=spec, dev_reference=ref,
                  selection_rule=f'max mean dev top-1 over H0-H8 among masked projection, masked cosine (tau 0-3) and {n_draws} fusion draws')
    return frozen, [(c[0], float(c[1])) for c in cands[:50]]


def evaluate_masked(fp, ms, norms, config):
    rows = np.flatnonzero(~fp['dev']); t = slice_rows(fp, rows); m = slice_rows(ms, rows); norms = norms[rows]
    best = correctness(score_masked_candidate(t, m, norms, config['spec']), t['truth'])
    proj = correctness(mask_channel(m, norms, 'proj', 0), t['truth']); cos = correctness(mask_channel(m, norms, 'cos', 0), t['truth'])
    out = []
    for p, proto in enumerate(PROTOS):
        pv_c, n01, n10 = mcnemar_exact(best[:,p], cos[:,p]); pv_p, _, _ = mcnemar_exact(best[:,p], proj[:,p])
        out.append(dict(protocol=proto, test_top1_frozen=float(best[:,p].mean()), test_top1_projection_tau0=float(proj[:,p].mean()),
                        test_top1_cosine_tau0=float(cos[:,p].mean()), mcnemar_frozen_vs_cosine_p=pv_c, mcnemar_frozen_vs_projection_p=pv_p,
                        discordant_frozen_only_vs_cos=n01, discordant_cos_only_vs_frozen=n10))
    return out

def channels(p, y, period, qe, sensor_parts):
    """Original eight stage-1 channels, before weight selection."""
    d = p.prepare(y, period, qe, sensor_parts); L = p.labels
    nrm = lambda v: v/v.max() if len(v) and v.max() > 0 else v
    def votes(order, scores, n):
        ix = order[:n]
        vals = np.bincount(p.partitions[ix], weights=np.maximum(scores[ix], 0)/np.arange(1, len(ix)+1), minlength=len(p.A_full))
        return vals[L]
    so = d['sem_order'][d['semcos'][d['sem_order']] >= -1.]
    cosv = (p.D @ y)/np.maximum(p.norms*np.linalg.norm(y), 1e-18)
    c2 = np.where(p.norms > 1e-12, cosv**2, 0.)
    cmax = np.array([c2[p.partitions == l].max() for l in L])
    res = np.linalg.norm(y[None,:]-(p.D@y/np.maximum(p.norms**2, 1e-18))[:,None]*p.D, axis=1)
    res[p.norms == 0] = np.inf
    rmin = np.array([res[p.partitions == l].min() for l in L])
    proj = np.clip(np.where(np.isfinite(rmin), 1-rmin/max(np.linalg.norm(y), 1e-18), 0.), 0, None)
    X = np.array([nrm(d['prior_full'][L]), nrm(d['centroid']), nrm(votes(d['rank_order'], d['rho'], 10)),
                  nrm(votes(d['fp_order'], d['fps'], 20)), nrm(votes(so, d['sem'], 20)),
                  nrm(d['community']), nrm(proj), nrm(cmax)])
    return X, p.A.copy(), d['max_obs'], rmin


def canonical_row_grid(data, fps):
    n = len(fps); total = len(SEEDS)*n
    if not n: raise ValueError('Empty canonical reference query grid.')
    for key in ['seed', 'qidx', 'truth']:
        if key not in data or np.asarray(data[key]).shape != (total,):
            raise ValueError(f'Canonical {key} grid shape must be ({total},).')
        if np.asarray(data[key]).dtype.kind not in 'iu':
            raise ValueError(f'Canonical {key} identity grid must use integers.')
    if not np.array_equal(data['seed'], np.repeat(SEEDS, n)) or not np.array_equal(data['qidx'], np.tile(np.arange(n), len(SEEDS))):
        raise ValueError('Canonical seed/query identity grid is incomplete, duplicated or reordered.')
    truth = np.tile([f['leak_partition'] for f in fps], len(SEEDS))
    if not np.array_equal(data['truth'], truth): raise ValueError('Canonical truth identity differs from reference scenarios.')
    labels = np.unique([f['leak_partition'] for f in fps])
    if not np.array_equal(labels, np.arange(len(labels))): raise ValueError('Reference labels must be contiguous partition integers.')
    if 'labels' not in data or np.asarray(data['labels']).dtype.kind not in 'iu' or not np.array_equal(data['labels'], labels):
        raise ValueError('Canonical partition label order is incompatible.')
    return total, len(labels)


def numeric_array(data, key, shape):
    a = np.asarray(data[key])
    if a.shape != shape: raise ValueError(f'Canonical {key} shape must be {shape}, got {a.shape}.')
    if a.dtype.kind not in 'fiu': raise ValueError(f'Canonical {key} must be a real numeric array.')
    return a


def validate_features(fp, fps):
    required = {'X','A','MO','RMIN','truth','node','seed','qidx','dev','protos','channels','labels'}
    if set(fp) != required: raise ValueError('Feature cache schema is missing or contains unsupported fields.')
    total, k = canonical_row_grid(fp, fps)
    if not np.array_equal(fp['channels'], CH): raise ValueError('Canonical channel order is incompatible.')
    if not np.array_equal(fp['protos'], PROTOS): raise ValueError('Canonical protocol grid is incompatible.')
    for key, shape in [('X',(total,len(PROTOS),len(CH),k)), ('A',(total,len(PROTOS),k,k)), ('MO',(total,len(PROTOS)))]:
        if not np.isfinite(numeric_array(fp,key,shape)).all(): raise ValueError(f'Canonical {key} must contain finite values.')
    residual = numeric_array(fp,'RMIN',(total,len(PROTOS),k))
    if np.isnan(residual).any() or np.isneginf(residual).any() or (residual < 0).any():
        raise ValueError('Projection residuals require nonnegative values or the +inf eligibility sentinel.')
    nodes = np.tile([f['leak_node'] for f in fps],len(SEEDS))
    if np.asarray(fp['node']).shape != (total,) or not np.array_equal(fp['node'],nodes):
        raise ValueError('Canonical node identity grid differs from reference scenarios.')
    dev,_ = split_nodes(fps)
    if np.asarray(fp['dev']).shape != (total,) or np.asarray(fp['dev']).dtype.kind != 'b' or not np.array_equal(fp['dev'],[n in dev for n in nodes]):
        raise ValueError('Canonical development/test grid is incompatible.')


def validate_masked(ms, fps):
    required = {'PRJ','COS','truth','seed','qidx','taus','labels'}
    if set(ms) != required: raise ValueError('Masked cache schema is missing or contains unsupported fields.')
    total,k = canonical_row_grid(ms,fps)
    if not np.array_equal(ms['taus'],TAUS): raise ValueError('Canonical masking threshold order is incompatible.')
    shape = (total,len(PROTOS),len(TAUS),k)
    residual = numeric_array(ms,'PRJ',shape); cosine = numeric_array(ms,'COS',shape)
    if np.isnan(residual).any() or np.isneginf(residual).any() or (residual < 0).any():
        raise ValueError('Masked projection residuals require nonnegative values or +inf eligibility sentinels.')
    if np.isnan(cosine).any() or np.isposinf(cosine).any() or np.any(np.abs(cosine[np.isfinite(cosine)]) > 1+1e-12):
        raise ValueError('Masked cosine requires finite values in [-1,1] or -inf eligibility sentinels.')
    if not np.array_equal(np.isposinf(residual),np.isneginf(cosine)):
        raise ValueError('Masked projection/cosine eligibility sentinels disagree.')


def feature_rankings_report(fp, expected_path):
    expected = {}
    with gzip.open(expected_path, 'rt', encoding='utf-8') as f:
        for r in map(json.loads, f):
            if r['method'] in ['fusion_strict', 'projection']:
                expected[r['seed'], r['query_index'], r['protocol'], r['method']] = r['ranking']
    mismatch = dict(fusion_strict=0, projection=0); scores = scores_default(fp); pairs = 0
    for k in range(len(fp['truth'])):
        for p, proto in enumerate(PROTOS):
            order = np.argsort(-scores[k,p], kind='stable'); ranking = fp['labels'][order[scores[k,p,order] > 0]].tolist()
            key = int(fp['seed'][k]), int(fp['qidx'][k]), proto
            mismatch['fusion_strict'] += ranking != expected[key + ('fusion_strict',)]
            mismatch['projection'] += int(np.argmin(fp['RMIN'][k,p])) != expected[key + ('projection',)][0]
            pairs += 1
    return dict(pairs_compared=pairs, mismatches=mismatch, dev_queries=int(fp['dev'].sum()), test_queries=int((~fp['dev']).sum()))


def rebuild_features(args, sources, arr):
    fpd, docs, vec, topo, adj, sp = sources; fps = fpd['fingerprints']; N = len(fps)
    dev_nodes, test_nodes = split_nodes(fps)
    qindices = np.arange(N)
    queries = [fps[i] for i in qindices]
    pools, mapping = build_pools(fps, queries, docs, vec, topo, adj, PROTOS)
    emb = FrozenEmbeddings.load(args.frozen_dir / 'retained_community_embeddings.npz')
    for p in pools.values(): p.set_embeddings(emb)
    n = len(queries); total = len(SEEDS)*n; K = len(np.unique([f['leak_partition'] for f in fps]))
    X = np.zeros((total, len(PROTOS), len(CH), K)); A = np.zeros((total, len(PROTOS), K, K))
    MO = np.zeros((total, len(PROTOS))); RMIN = np.zeros((total, len(PROTOS), K))
    for local, qi in enumerate(qindices):
        q = fps[int(qi)]
        for pj, proto in enumerate(PROTOS):
            p = pools[mapping[local, proto]]
            if len(p.labels) != K: raise ValueError(f'{proto} lost a partition label; frozen stage-1 contract changed.')
            for si, seed in enumerate(SEEDS):
                k = si*n+local; oi = si*N+int(qi)
                X[k,pj], A[k,pj], MO[k,pj], RMIN[k,pj] = channels(p, arr['observations'][oi], q['demand_period'], arr['embeddings'][oi], sp)
    qirows = np.tile(qindices, len(SEEDS))
    node = np.array([fps[i]['leak_node'] for i in qirows])
    fp = dict(X=X, A=A, MO=MO, RMIN=RMIN, truth=np.array([fps[i]['leak_partition'] for i in qirows]),
              node=node, seed=np.repeat(SEEDS, n), qidx=qirows, dev=np.array([n in dev_nodes for n in node]),
              protos=np.array(PROTOS), channels=np.array(CH), labels=np.arange(K))
    np.savez_compressed(args.output_dir / 'features.npz', **fp)
    save_json(args.output_dir / 'split.json', dict(dev_nodes=sorted(dev_nodes), test_nodes=sorted(test_nodes),
              rule='per partition: lexicographically smaller leak node -> dev, the other -> test',
              H0='No node exclusion: complete-corpus baseline; dev/test describes query allocation only.'))
    report = feature_rankings_report(fp, args.frozen_dir / 'predictions.jsonl.gz')
    save_json(args.output_dir / 'stage1_reproduction.json', report)
    if any(report['mismatches'].values()): raise AssertionError(f'Stage-1 ranking mismatch: {report}')
    return fp


def observations_for(fp, arr):
    index = {(int(s), int(q)): i for i, (s, q) in enumerate(zip(arr['seed'], arr['query_index']))}
    return arr['observations'][[index[int(s), int(q)] for s, q in zip(fp['seed'], fp['qidx'])]]


def rebuild_masked(args, fp, fps, observations):
    from experiments.strict_engine.engine import keep
    total = len(fp['truth']); K = len(fp['labels'])
    PRJ = np.full((total, len(PROTOS), len(TAUS), K), np.inf)
    COS = np.full_like(PRJ, -np.inf)
    D_all = np.array([f['sensor_fingerprint'] for f in fps], float)
    part = np.array([f['leak_partition'] for f in fps], int)
    for pj, proto in enumerate(PROTOS):
        for qi in np.unique(fp['qidx']):
            kidx = np.flatnonzero([keep(f, fps[int(qi)], proto) for f in fps]); Dk, pk = D_all[kidx], part[kidx]
            for k in np.flatnonzero(fp['qidx'] == qi):
                y = observations[k]
                for ti, tau in enumerate(TAUS):
                    m = np.abs(y) >= tau*SIGMA; ym = y[m]; Dm = Dk[:,m]
                    nrm2 = (Dm**2).sum(axis=1); dot = Dm@ym; yn = np.linalg.norm(ym)
                    res = np.linalg.norm(ym[None,:]-(dot/np.maximum(nrm2, 1e-18))[:,None]*Dm, axis=1)
                    res[nrm2 <= 1e-18] = np.inf
                    cos = dot/np.maximum(np.sqrt(nrm2)*max(yn, 1e-18), 1e-18); cos[nrm2 <= 1e-18] = -np.inf
                    for li, label in enumerate(fp['labels']):
                        sel = pk == label
                        if sel.any(): PRJ[k,pj,ti,li] = res[sel].min(); COS[k,pj,ti,li] = cos[sel].max()
        print(f'Masked scores {proto} complete', flush=True)
    ms = dict(PRJ=PRJ, COS=COS, truth=fp['truth'], seed=fp['seed'], qidx=fp['qidx'], taus=np.array(TAUS), labels=fp['labels'])
    np.savez_compressed(args.output_dir / 'masked_scores.npz', **ms)
    return ms


def align_masked(ms, fp):
    index = {(int(s), int(q)): i for i, (s, q) in enumerate(zip(ms['seed'], ms['qidx']))}
    if len(index) != len(ms['seed']): raise ValueError('Duplicate masked seed/query identity grid.')
    try: rows = [index[int(s), int(q)] for s, q in zip(fp['seed'], fp['qidx'])]
    except KeyError as exc: raise ValueError('Masked score rows do not match stage-1 features.') from exc
    ms = slice_rows(ms, rows)
    for key in ['seed', 'qidx', 'truth', 'labels']:
        if not np.array_equal(ms[key], fp[key]): raise ValueError(f'Masked and feature {key} mismatch.')
    return ms


def add_saved_cosine(rows, fp, expected_path):
    lookup = {(int(s), int(q)): i for i, (s, q) in enumerate(zip(fp['seed'], fp['qidx']))}
    correct = np.zeros((len(fp['truth']), len(PROTOS)), int)
    with gzip.open(expected_path, 'rt', encoding='utf-8') as f:
        for r in map(json.loads, f):
            key = r['seed'], r['query_index']
            if key in lookup and r['method'] == 'cosine' and r['protocol'] in PROTOS:
                correct[lookup[key], PROTOS.index(r['protocol'])] = r['top1']
    for p, row in enumerate(rows): row['test_top1_saved_cosine'] = float(correct[~fp['dev'], p].mean())


def compare_results(actual, expected):
    return all(all(k in e and (v == e[k] if isinstance(v, str) else np.isclose(v, e[k], rtol=0, atol=1e-12))
                   for k, v in a.items()) for a, e in zip(actual, expected)) and len(actual) == len(expected)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--frozen-dir', type=Path, default=DEFAULT_FROZEN_DIR, help='Strict scenario/query embedding assets.')
    p.add_argument('--selection-frozen-dir', type=Path, default=DEFAULT_SELECTION_FROZEN)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/node_selection'))
    p.add_argument('--stage', choices=['all', 'features', 'select', 'masked', 'masked-select', 'replay'], default='replay',
                   help='replay evaluates archived configs; all rebuilds features, repeats dev search, freezes and tests.')
    p.add_argument('--limit', type=int, default=0, help='N query cases per dev/test split per seed for smoke checks.')
    p.add_argument('--draws', type=int, default=400, help='Original search uses 400 draws; reduced draws are smoke only.')
    args = p.parse_args(argv)
    if args.limit < 0 or args.draws < 0: p.error('limits must be nonnegative')
    return args


def require_archive_parity(actual, expected, label):
    if not compare_results(actual, expected): raise ValueError(f'{label} archive parity mismatch.')
    return True


def main(argv=None):
    args = parse_args(argv); out = args.output_dir; out.mkdir(parents=True, exist_ok=True); start = time.time()
    sources = load_sources(args.data_dir, args.frozen_dir); fps = sources[0]['fingerprints']
    arr = load_observations(args.frozen_dir, fps, sources[0]['sensor_nodes'])
    report = dict(stage=args.stage, selection_mode='archived frozen test replay' if args.stage == 'replay' else 'development-only selection and/or artifact rebuild',
                  draws=args.draws, H0='Complete-corpus baseline; no node exclusion.', protocols=PROTOS,
                  fixed_search_seed=20261011, test_used_for_selection=False)
    if args.stage in ['all', 'features']: full_fp = rebuild_features(args, sources, arr)
    else:
        feature_path = args.selection_frozen_dir / 'features.npz' if args.stage == 'replay' else out / 'features.npz'
        if not feature_path.exists(): feature_path = args.selection_frozen_dir / 'features.npz'
        full_fp = load_npz(feature_path)
    # Validate every canonical row and all schema fields before applying a smoke limit.
    validate_features(full_fp, fps)
    fp = subset_features(full_fp, args.limit)
    full_observations = observations_for(full_fp, arr); full_norms = np.linalg.norm(full_observations, axis=1)
    selected = selected_rows(full_fp, args.limit); norms = full_norms[selected]
    if args.stage in ['all', 'select', 'replay']:
        if args.stage == 'replay': config = json.loads((args.selection_frozen_dir / 'frozen_config.json').read_text(encoding='utf-8'))
        else:
            config, search = select_fusion(fp, args.draws)
            save_json(out / 'dev_search_all.json', search)
        save_json(out / 'frozen_config.json', config)
        # Selection is frozen on disk before the test subset enters evaluation.
        results = evaluate_frozen(fp, config); add_saved_cosine(results, fp, args.frozen_dir / 'predictions.jsonl.gz')
        save_json(out / 'test_results.json', results)
        if args.stage == 'replay' or not args.limit:
            reference_results = results
            if args.limit:
                reference_results = evaluate_frozen(full_fp,config)
                add_saved_cosine(reference_results,full_fp,args.frozen_dir/'predictions.jsonl.gz')
            report['test_results_equal_archive'] = require_archive_parity(reference_results,
                json.loads((args.selection_frozen_dir/'test_results.json').read_text(encoding='utf-8')),'Frozen test')
    if args.stage in ['all', 'masked']:
        full_ms = rebuild_masked(args, full_fp, fps, full_observations)
        validate_masked(full_ms,fps)
    elif args.stage in ['masked-select', 'replay']:
        path = args.selection_frozen_dir / 'masked_scores.npz' if args.stage == 'replay' else out / 'masked_scores.npz'
        if not path.exists(): path = args.selection_frozen_dir / 'masked_scores.npz'
        full_ms = load_npz(path)
        validate_masked(full_ms,fps)
    if args.stage in ['all', 'masked-select', 'replay']:
        ms = align_masked(full_ms,fp)
        if args.stage == 'replay': config = json.loads((args.selection_frozen_dir / 'frozen_masked_config.json').read_text(encoding='utf-8'))
        else:
            config, search = select_masked(fp, ms, norms, args.draws)
            save_json(out / 'dev_masked_top50.json', search)
        save_json(out / 'frozen_masked_config.json', config)
        results = evaluate_masked(fp, ms, norms, config)
        save_json(out / 'test_masked_results.json', results)
        if args.stage == 'replay' or not args.limit:
            reference_results = evaluate_masked(full_fp,full_ms,full_norms,config) if args.limit else results
            report['masked_test_results_equal_archive'] = require_archive_parity(reference_results,
                json.loads((args.selection_frozen_dir/'test_masked_results.json').read_text(encoding='utf-8')),'Frozen masked test')
    dev_nodes, test_nodes = split_nodes(fps)
    report.update(canonical_rows=len(full_fp['truth']),dev_nodes=len(dev_nodes), test_nodes=len(test_nodes),
                  dev_rows=int(fp['dev'].sum()), test_rows=int((~fp['dev']).sum()), elapsed_seconds=time.time()-start)
    save_json(out / 'run_completion.json', dict(finished=True, **report))
    files = [args.data_dir / 'sensor_fingerprints.json', args.frozen_dir / 'query_observations_embeddings.npz', args.frozen_dir / 'retained_community_embeddings.npz']
    if args.stage == 'replay': files += list(args.selection_frozen_dir.glob('*'))
    save_json(out / 'source_manifest.json', dict(files={str(p.resolve()): filehash(p) for p in files if p.is_file()}, numpy_version=np.__version__, **report))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__': main()
