"""Portable original multileak sparse computation with explicit frozen inputs."""
from experiments.historical.exa7_support import INPUTS, FrozenReference, load_partition_map
import os, sys, json, time, copy, pickle, random, warnings, tempfile, uuid
import numpy as np
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
pass
import wntr
from scipy.optimize import nnls
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
_P = INPUTS['epanet_prefix']
INP = INPUTS['inp']
PART_PKL, K = ('partition_results_leiden/partitions.pkl', 15)
NOISE, SEED, Q_REF = (0.02, 11, 20.0)
COS_THR = 0.999
KS = [1, 2, 3, 5]
N_SCEN = INPUTS.get('N_SCEN', 60)
K_MAX = 8
DELTAS = [0.02, 0.05, 0.1, 0.2]
OUT = 'expA1_v2_results.json'

def build_classes(D, thr):
    nrm = np.linalg.norm(D, axis=1)
    ok = nrm > 1e-12
    U = np.zeros_like(D)
    U[ok] = D[ok] / nrm[ok, None]
    C = np.clip(U @ U.T, -1, 1)
    Dm = 1.0 - C
    np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2
    return fcluster(linkage(squareform(Dm, checks=False), method='complete'), t=1.0 - thr, criterion='distance')

def nnomp(y, D, Dn, k_max, delta=None, k_fixed=None):
    """Non-negative OMP over the class dictionary.
    k_fixed -> oracle mode; delta -> stop on relative residual improvement."""
    S, r = ([], y.copy())
    prev = np.linalg.norm(y)
    best = ([], np.array([]))
    for _ in range(k_max):
        corr = D @ r / np.maximum(Dn, 1e-18)
        if S:
            corr[S] = -np.inf
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        cand = S + [j]
        beta, _ = nnls(D[cand].T, y)
        rn = float(np.linalg.norm(y - D[cand].T @ beta))
        if k_fixed is None and delta is not None and (prev > 0) and ((prev - rn) / prev < delta):
            break
        S, r, prev = (cand, y - D[cand].T @ beta, rn)
        best = (list(S), beta.copy())
        if k_fixed is not None and len(S) >= k_fixed:
            break
    S, beta = best
    keep = beta > 1e-09
    return ([S[i] for i in range(len(S)) if keep[i]], beta[keep])

def main():
    pass
    pass
    pass
    loc = FrozenReference()
    sensors = sorted(loc.sensor_nodes, key=lambda x: int(x))
    m = len(sensors)
    wn0 = wntr.network.WaterNetworkModel(INP)
    juncs = list(wn0.junction_name_list)
    n2c = load_partition_map()
    xy = {n: np.array(wn0.get_node(n).coordinates) for n in juncs}
    z = np.load(INPUTS['sensitivity'], allow_pickle=True)
    S_full, nodes, meas = (z['S'], [str(x) for x in z['nodes']], [str(x) for x in z['meas']])
    midx = {n: i for i, n in enumerate(meas)}
    sens_cols = [midx[str(s)] for s in sensors]
    F = S_full[:, sens_cols]
    nidx = {n: i for i, n in enumerate(nodes)}
    lab = build_classes(F, COS_THR)
    n_cls = int(lab.max())
    node_class = {nodes[i]: int(lab[i]) for i in range(len(nodes))}
    D = np.zeros((n_cls, m))
    cls_nodes = {}
    for c in range(1, n_cls + 1):
        ii = np.where(lab == c)[0]
        cls_nodes[c] = [nodes[i] for i in ii]
        D[c - 1] = F[ii].mean(axis=0)
    Dn = np.linalg.norm(D, axis=1)
    Un = D / np.maximum(Dn[:, None], 1e-18)
    Gm = Un @ Un.T
    np.fill_diagonal(Gm, -1)
    pass
    pass
    cls_xy = {c: np.mean([xy[n] for n in cls_nodes[c]], axis=0) for c in cls_nodes}
    wn = copy.deepcopy(wn0)
    n_sims = 0
    bp = wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
    n_sims += 1

    def multi_dp(nds, qs):
        nonlocal n_sims
        w = copy.deepcopy(wn)
        for nd, q in zip(nds, qs):
            w.get_node(nd).add_demand(q / 1000.0, '881')
        lp = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
        n_sims += 1
        return np.array([float(lp[s] - bp[s]) for s in sensors])
    rng = random.Random(SEED)
    npr = np.random.default_rng(SEED)
    np.random.seed(SEED)
    pool = [n for n in juncs if n not in set(map(str, loc.sensor_nodes))]

    def draw(k, regime):
        while True:
            nds = rng.sample(pool, k)
            if len({node_class[n] for n in nds}) == k:
                break
        if regime == 'balanced':
            qs = [float(npr.uniform(5, 20)) for _ in range(k)]
        else:
            qs = [float(np.exp(npr.uniform(np.log(1), np.log(20)))) for _ in range(k)]
        return (nds, qs)
    results = {}
    pass
    pass
    pass
    for regime in ['balanced', 'skewed']:
        for k in KS:
            t0 = time.time()
            o_cls = o_part = e_cls = b_cls = 0
            tot = 0
            khat, spur, qerr, dists = ([], [], [], [])
            for _ in range(N_SCEN):
                nds, qs = draw(k, regime)
                y = multi_dp(nds, qs) + np.random.normal(0, NOISE, m)
                true_cls = [node_class[n] for n in nds]
                tot += k
                So, bo = nnomp(y, D, Dn, K_MAX, k_fixed=k)
                po = [c + 1 for c in So]
                o_cls += sum((1 for c in true_cls if c in po))
                o_part += sum((1 for n in nds if int(n2c[n]) in {int(n2c[cls_nodes[c][0]]) for c in po}))
                Se, be = nnomp(y, D, Dn, K_MAX, delta=0.05)
                pe = [c + 1 for c in Se]
                khat.append(len(pe))
                e_cls += sum((1 for c in true_cls if c in pe))
                spur.append(sum((1 for c in pe if c not in true_cls)))
                for c, q in zip(true_cls, qs):
                    if c in pe:
                        qerr.append(abs(be[pe.index(c)] * Q_REF - q) / q)
                for n in nds:
                    if pe:
                        dists.append(min((float(np.linalg.norm(cls_xy[c] - xy[n])) for c in pe)))
                beta_s = D @ y / np.maximum(Dn ** 2, 1e-18)
                res_s = np.linalg.norm(y[None, :] - beta_s[:, None] * D, axis=1)
                top = [int(j) + 1 for j in np.argsort(res_s)[:k]]
                b_cls += sum((1 for c in true_cls if c in top))
            r = dict(regime=regime, k=k, n=N_SCEN, oracle_class_recall=o_cls / tot, oracle_partition_recall=o_part / tot, est_class_recall=e_cls / tot, khat_mean=float(np.mean(khat)), khat_exact=float(np.mean([x == k for x in khat])), spurious_per_scen=float(np.mean(spur)), q_rel_err_median=float(np.median(qerr)) if qerr else float('nan'), dist_median=float(np.median(dists)) if dists else float('nan'), baseline_class_recall=b_cls / tot)
            results[f'{regime}_k{k}'] = r
            pass
    pass
    sens = {}
    for d in DELTAS:
        rng2 = random.Random(SEED)
        npr2 = np.random.default_rng(SEED)
        np.random.seed(SEED)
        kk, rc, sp, tot = ([], 0, [], 0)
        for _ in range(N_SCEN):
            while True:
                nds = rng2.sample(pool, 3)
                if len({node_class[n] for n in nds}) == 3:
                    break
            qs = [float(npr2.uniform(5, 20)) for _ in range(3)]
            y = multi_dp(nds, qs) + np.random.normal(0, NOISE, m)
            tc = [node_class[n] for n in nds]
            Se, _ = nnomp(y, D, Dn, K_MAX, delta=d)
            pe = [c + 1 for c in Se]
            kk.append(len(pe))
            rc += sum((1 for c in tc if c in pe))
            sp.append(sum((1 for c in pe if c not in tc)))
            tot += 3
        sens[str(d)] = dict(khat=float(np.mean(kk)), recall=rc / tot, spurious=float(np.mean(sp)))
        pass
    json.dump(dict(n_classes=n_cls, cos_threshold=COS_THR, results=results, stopping_sensitivity=sens, n_simulations=n_sims, config=dict(ks=KS, n_scen=N_SCEN, noise=NOISE, seed=SEED, k_max=K_MAX)), open(OUT, 'w'), indent=1)
    pass
if __name__ == '__main__':
    main()
