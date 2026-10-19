"""Portable original multileak exhaustive computation with explicit frozen inputs."""
from experiments.historical.exa7_support import INPUTS, FrozenReference, load_partition_map
import os, sys, json, copy, pickle, random, warnings, tempfile, uuid, itertools, time
import numpy as np
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
pass
pass
import wntr
from scipy.optimize import nnls
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
_P = INPUTS['epanet_prefix']
INP = INPUTS['inp']
NOISE = 0.02
SEED = 11
Q_REF = 20.0
COS = 0.999
N_SCEN = INPUTS.get('N_SCEN', 40)
z = np.load(INPUTS['sensitivity'], allow_pickle=True)
Sf = z['S']
nodes = [str(x) for x in z['nodes']]
meas = [str(x) for x in z['meas']]
midx = {n: i for i, n in enumerate(meas)}
loc = FrozenReference()
sensors = sorted(loc.sensor_nodes, key=lambda x: int(x))
cols = [midx[str(s)] for s in sensors]
m = len(cols)
wn0 = wntr.network.WaterNetworkModel(INP)
juncs = list(wn0.junction_name_list)
n2c = load_partition_map()

def classes(F, thr):
    nr = np.linalg.norm(F, axis=1)
    ok = nr > 1e-12
    U = np.zeros_like(F)
    U[ok] = F[ok] / nr[ok, None]
    C = np.clip(U @ U.T, -1, 1)
    D = 1 - C
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    return fcluster(linkage(squareform(D, checks=False), method='complete'), t=1 - thr, criterion='distance')
F = Sf[:, cols]
lab = classes(F, COS)
ncls = int(lab.max())
node_cls = {nodes[i]: int(lab[i]) for i in range(len(nodes))}
D = np.zeros((ncls, m))
cn = {}
for c in range(1, ncls + 1):
    ii = np.where(lab == c)[0]
    cn[c] = [nodes[i] for i in ii]
    D[c - 1] = F[ii].mean(axis=0)
Dn = np.linalg.norm(D, axis=1)
U = D / np.maximum(Dn[:, None], 1e-18)
Gm = U @ U.T
np.fill_diagonal(Gm, -1)
mu = float(Gm.max())
pass
pass
pass
pass
pass
wn = copy.deepcopy(wn0)
bp = wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
nsim = 1

def md(nds, qs):
    global nsim
    w = copy.deepcopy(wn)
    for nd, q in zip(nds, qs):
        w.get_node(nd).add_demand(q / 1000.0, '881')
    lp = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
    nsim += 1
    return np.array([float(lp[s] - bp[s]) for s in sensors])
rng = random.Random(SEED)
npr = np.random.default_rng(SEED)
np.random.seed(SEED)
pool = [n for n in juncs if n not in set(map(str, loc.sensor_nodes))]
pass
pass
pass
d1 = {}
for k in [2, 3]:
    combos = list(itertools.combinations(range(ncls), k))
    exh = grd = tot = 0
    resids = []
    for _ in range(N_SCEN):
        while True:
            nds = rng.sample(pool, k)
            if len({node_cls[n] for n in nds}) == k:
                break
        qs = [float(npr.uniform(5, 20)) for _ in range(k)]
        y = md(nds, qs) + np.random.normal(0, NOISE, m)
        tc = {node_cls[n] - 1 for n in nds}
        tot += k
        best = (np.inf, None)
        for cb in combos:
            b, _ = nnls(D[list(cb)].T, y)
            r = float(np.linalg.norm(y - D[list(cb)].T @ b))
            if r < best[0]:
                best = (r, cb)
        exh += len(tc & set(best[1]))
        resids.append(best[0])
        S = []
        rr = y.copy()
        for _ in range(k):
            c = D @ rr / np.maximum(Dn, 1e-18)
            if S:
                c[S] = -np.inf
            j = int(np.argmax(c))
            S.append(j)
            b, _ = nnls(D[S].T, y)
            rr = y - D[S].T @ b
        grd += len(tc & set(S))
    d1[k] = dict(exhaustive=exh / tot, greedy=grd / tot, median_resid=float(np.median(resids)))
    pass
pass
pass
pass
pass
d2 = {}
r2 = random.Random(3)
for b in [30, 45, 60, 90, 150, 250, 381]:
    cs = cols if b == 30 else r2.sample(range(len(meas)), b)
    Fb = Sf[:, cs]
    lb = classes(Fb, COS)
    nc = int(lb.max())
    Db = np.zeros((nc, len(cs)))
    for c in range(1, nc + 1):
        Db[c - 1] = Fb[np.where(lb == c)[0]].mean(axis=0)
    Un = Db / np.maximum(np.linalg.norm(Db, axis=1)[:, None], 1e-18)
    Gb = Un @ Un.T
    np.fill_diagonal(Gb, -1)
    mb = float(Gb.max())
    d2[str(b)] = dict(classes=nc, mu=mb, k_bound=float((1 + 1 / mb) / 2))
    pass
json.dump(dict(mu_deployed=mu, k_bound_deployed=float((1 + 1 / mu) / 2), D1=d1, D2=d2, n_sims=nsim), open('expA1b_results.json', 'w'), indent=1)
pass
