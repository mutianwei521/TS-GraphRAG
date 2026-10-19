"""Leakage-free retained-corpus evaluation of six-channel fusion + graph aggregation.

The candidate domain is the set of partition labels represented by retained
scenarios. All statistical/semantic artifacts are rebuilt from this corpus.
Static network topology/sensor assignments are allowed side information.
No held-out query metadata is passed to scoring beyond the observed demand period.
"""
from __future__ import annotations
import hashlib, json, os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO / 'data' / 'reference'
DEFAULT_FROZEN_DIR = REPO / 'data' / 'frozen' / 'strict_engine'
SEEDS = [41, 141, 241, 341, 441]
PROTOCOLS = [f'H{i}' for i in range(10)]
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
CHANNELS = ['prior', 'centroid', 'rank', 'fingerprint', 'semantic', 'community']

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def keep(f, q, protocol):
    sn, sq, st = [f[k] == q[k] for k in ['leak_node', 'leak_rate_Ls', 'demand_period']]
    return {'H0': True, 'H1': not(sn and sq and st), 'H2': not sn,
            'H3': not sq, 'H4': not st, 'H5': not(sn or sq),
            'H6': not(sn or st), 'H7': not(sq or st), 'H8': not(sn or sq or st),
            'H9': f['leak_partition'] != q['leak_partition']}[protocol]

def community_summary(pid, scenarios, topo):
    """Exact deterministic level-1 template used in graph_rag.py; retained only."""
    rates = [s['leak_rate_Ls'] for s in scenarios]
    drops = [s.get('max_pressure_drop', 0) for s in scenarios]
    periods = sorted(set(s.get('demand_period', '') for s in scenarios))
    affected = [s.get('num_affected_nodes', 0) for s in scenarios]
    nodes = sorted(set(s.get('leak_node', '') for s in scenarios))
    nc = topo['partitions'].get(str(pid), {}).get('node_count', 0)
    sev = defaultdict(int)
    for s in scenarios:
        sev[s.get('leak_severity', 'unknown')] += 1
    text = (f'Partition {pid} contains {nc} junction nodes. '
            f'This community has been characterized through {len(scenarios)} '
            f'leak simulation scenarios with leak rates ranging from '
            f'{min(rates):.0f} to {max(rates):.0f} L/s. '
            f'The hydraulic response shows maximum pressure drops between '
            f'{min(drops):.2f} and {max(drops):.2f} meters, '
            f'affecting {min(affected)} to {max(affected)} nodes. '
            f"Demand periods covered: {', '.join(periods)}. "
            f"Representative leak nodes in this community: {', '.join(nodes)}. "
            'Severity distribution: ' + ', '.join(f'{k}: {v}' for k,v in sorted(sev.items())) + '.')
    return text

@dataclass(frozen=True)
class Config:
    top_k: int = 10
    beta: float = .15
    steps: int = 1
    weak: float = .1
    strong: float = 1.
    semantic_multiplier: float = 1.
    semantic_threshold: float = -1.
    drop_channel: str = ''
    uniform_weights: bool = False

class Pool:
    def __init__(self, fingerprints, documents, vectors, topo, adjacency):
        # Only caller-provided retained objects are accessible in this class.
        self.fps = fingerprints
        self.ids = [f['scenario_id'] for f in fingerprints]
        self.key = digest(self.ids)
        assert self.ids and set(self.ids) == set(documents) == set(vectors)
        self.docs = [documents[s] for s in self.ids]
        self.E = np.array([vectors[s] for s in self.ids], dtype=np.float32)
        self.D = np.array([f['sensor_fingerprint'] for f in fingerprints], dtype=float)
        self.norms = np.linalg.norm(self.D, axis=1)
        self.partitions = np.array([f['leak_partition'] for f in fingerprints], int)
        self.labels = np.unique(self.partitions)
        self.periods = np.array([f['demand_period'] for f in fingerprints])
        self.ranks = self.D.argsort(axis=1).argsort(axis=1).astype(float)
        self.centroids = {int(p): self.D[self.partitions == p].mean(axis=0) for p in self.labels}
        self.period_centroids = {(int(p),t): self.D[(self.partitions==p)&(self.periods==t)].mean(axis=0)
                                 for p in self.labels for t in np.unique(self.periods)
                                 if np.any((self.partitions==p)&(self.periods==t))}
        self.summary_texts = {int(p): community_summary(int(p), [d for d in self.docs if d['leak_partition']==p], topo)
                              for p in self.labels}
        self.community_E = None
        self.A_full = np.array(adjacency, float)
        self.A = self.A_full[np.ix_(self.labels, self.labels)].copy()
        self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), 1.)
        self.artifact_digest = digest({'ids': self.ids, 'fps': fingerprints, 'docs': self.docs,
                                      'summaries': self.summary_texts,
                                      'vectors_sha': hashlib.sha256(self.E.tobytes()).hexdigest()})

    def set_embeddings(self, text_embeddings):
        self.community_E = np.array([text_embeddings[self.summary_texts[int(p)]] for p in self.labels])

    def provenance(self):
        return {'pool_id':self.key,'retained_ids':self.ids,'retained_labels':self.labels.tolist(),
                'artifact_sha256':self.artifact_digest,
                'semantic_document_ids':self.ids,'centroid_source_ids':self.ids,
                'community_source_ids':{str(p):[s for s,f in zip(self.ids,self.fps) if f['leak_partition']==p] for p in self.labels},
                'community_summaries':self.summary_texts,
                'retained_graph_adjacency_sha256':hashlib.sha256(self.A.tobytes()).hexdigest()}

    def prepare(self, y, period, qemb, sensor_partitions):
        """Query channels before top-k selection; computation sees retained rows only."""
        norm = np.linalg.norm(y)
        ix = np.flatnonzero(self.periods == period)
        if len(ix)==0:
            ix = np.arange(len(self.D))
        cos = self.D @ y / np.maximum(self.norms * norm, 1e-18)
        euc = np.linalg.norm(self.D-y[None,:],axis=1)
        fps = .5*np.maximum(cos,0)+.5/(1+euc/np.maximum(np.maximum(norm,self.norms),.1))
        valid = ix[self.norms[ix]>=1e-10]
        fp_order = valid[np.argsort(-fps[valid],kind='stable')]
        r = y.argsort().argsort().astype(float)
        rho = 1-6*((self.ranks-r)**2).sum(axis=1)/(len(y)*(len(y)**2-1))
        rank_order = ix[np.argsort(-rho[ix],kind='stable')]
        C = np.array([self.period_centroids.get((int(p),period),self.centroids[int(p)]) for p in self.labels])
        cn = np.linalg.norm(C,axis=1)
        cs = .5*np.maximum(C@y/np.maximum(cn*norm,1e-18),0)+.5/(1+np.linalg.norm(C-y,axis=1)/np.maximum(np.maximum(norm,cn),.1))
        cs[cn<1e-10]=0
        # Match LanceDB's default squared-L2 metric using original frozen document vectors.
        d2 = np.sum((self.E - qemb[None,:])**2,axis=1)
        sem = 1/(1+d2)
        semcos = self.E @ qemb / np.maximum(np.linalg.norm(self.E,axis=1)*np.linalg.norm(qemb),1e-18)
        sem_order = np.argsort(d2,kind='stable')
        comm = np.maximum(self.community_E @ qemb / np.maximum(np.linalg.norm(self.community_E,axis=1)*np.linalg.norm(qemb),1e-18),0)
        # A separate full-topology diagnostic retains all physical labels, without
        # restoring any deleted templates, documents, centroids or summaries.
        prior_full = np.zeros(len(self.A_full))
        totals = np.bincount(sensor_partitions,weights=np.abs(y),minlength=len(prior_full))
        peaks=np.array([np.abs(y)[sensor_partitions==p].max(initial=0) for p in range(len(prior_full))])
        affected=np.bincount(sensor_partitions,weights=(np.abs(y)>.05),minlength=len(prior_full))
        if totals.max()>1e-10:
            prior_full=.5*totals/totals.max()+.3*peaks/max(peaks.max(),1e-18)+.2*affected/max(affected.max(),1)
        projection=np.linalg.norm(y[None,:]-(self.D@y/np.maximum(self.norms**2,1e-18))[:,None]*self.D,axis=1)
        projection[self.norms==0]=np.inf
        cos[self.norms<=1e-12]=-np.inf
        projection_eligible=np.flatnonzero(self.norms>0)
        cosine_eligible=np.flatnonzero(self.norms>1e-12)
        return dict(prior_full=prior_full,centroid=cs,rank_order=rank_order,rho=rho,fp_order=fp_order,fps=fps,
                    sem_order=sem_order,sem=sem,semcos=semcos,community=comm,max_obs=float(np.abs(y).max()),
                    projection_order=projection_eligible[np.argsort(projection[projection_eligible])],
                    cosine_order=cosine_eligible[np.argsort(-cos[cosine_eligible])])

    def score(self, prepared, config=Config(), open_labels=False):
        d=prepared
        labels=np.arange(len(self.A_full)) if open_labels else self.labels
        def norm(v):
            return v/v.max() if len(v) and v.max()>0 else v
        def expand(v):
            z=np.zeros(len(self.A_full)); z[self.labels]=v
            return z[labels]
        def votes(order, scores, n):
            ix=order[:n]
            vals=np.bincount(self.partitions[ix],weights=np.maximum(scores[ix],0)/np.arange(1,len(ix)+1),minlength=len(self.A_full))
            return vals[labels]
        so=d['sem_order'][d['semcos'][d['sem_order']]>=config.semantic_threshold]
        X=np.array([norm(d['prior_full'][labels]),norm(expand(d['centroid'])),
                    norm(votes(d['rank_order'],d['rho'],config.top_k)),
                    norm(votes(d['fp_order'],d['fps'],config.top_k*2)),
                    norm(votes(so,d['sem'],config.top_k*2)),norm(expand(d['community']))])
        if d['max_obs']>=config.strong: w=np.array([.25,.12,.08,.30,.10,.15])
        elif d['max_obs']>=config.weak: w=np.array([.20,.20,.15,.18,.10,.17])
        else: w=np.array([.15,.25,.20,.12,.10,.18])
        if config.uniform_weights: w[:]=1/6
        w[4:6]*=config.semantic_multiplier
        if config.drop_channel: w[CHANNELS.index(config.drop_channel)]=0
        w/=w.sum()
        scores=w@X
        A=self.A_full.copy() if open_labels else self.A
        if open_labels: A/=np.maximum(A.sum(axis=1,keepdims=True),1.)
        for _ in range(config.steps): scores=(1-config.beta)*scores+config.beta*(A@scores)
        order=np.argsort(-scores,kind='stable')
        order=order[scores[order]>0]
        return labels[order].tolist(),scores[order].tolist(),X

    def baseline_rank(self, order):
        return list(dict.fromkeys(int(self.partitions[i]) for i in order))

def rank_metrics(ranking, truth):
    return [int(bool(ranking) and ranking[0]==truth), int(truth in ranking[:3]),
            1/(ranking.index(truth)+1) if truth in ranking else 0.]


class FrozenEmbeddings(dict):
    """Resolve an embedding only for an exact retained-summary SHA-256."""
    def __missing__(self, text):
        sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if sha not in self:
            raise ValueError(f'No frozen embedding for retained summary SHA-256 {sha}; rebuild embeddings explicitly.')
        return dict.__getitem__(self, sha)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({str(h): v for h, v in zip(z['text_sha256'], z['vectors'])})


def build_query_text(sensor_observations, demand_period, sensor_to_partition):
    """Original observation-only query formatter, with no locator or database import."""
    sorted_obs = sorted(sensor_observations.items(), key=lambda x: x[1])
    max_node, max_drop = sorted_obs[0]
    max_drop_abs = abs(max_drop)
    affected_parts = sorted(set(sensor_to_partition.get(n, -1) for n, dp in sorted_obs if dp < -0.1))
    severity = 'severe' if max_drop_abs >= 5 else 'moderate' if max_drop_abs >= 1 else 'small' if max_drop_abs >= .2 else 'micro'
    sensor_text = ', '.join(f"Sensor {n} (P#{sensor_to_partition.get(n, '?')}): {dp:.3f}m" for n, dp in sorted_obs[:8])
    return (f"A {severity} leak during {demand_period or 'unknown'}. "
            f"Max drop {max_drop_abs:.3f}m at node {max_node} (P#{sensor_to_partition.get(max_node, '?')}). "
            f"Sensors: {sensor_text}. Affected partitions: {affected_parts}.")


def load_sources(data_dir=DEFAULT_DATA_DIR, frozen_dir=DEFAULT_FROZEN_DIR):
    """Load explicit reference assets and immutable per-document vectors; no DB access."""
    data_dir, frozen_dir = Path(data_dir), Path(frozen_dir)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    fp = read(data_dir / 'sensor_fingerprints.json')
    raw_docs = read(data_dir / 'scenario_summaries.json')
    docs = {d['scenario_id']: d for d in raw_docs}
    frozen_docs = {d['scenario_id']: d for d in read(frozen_dir / 'frozen_scenario_documents.json')}
    ids = [f['scenario_id'] for f in fp['fingerprints']]
    with np.load(frozen_dir / 'frozen_scenario_vectors.npz', allow_pickle=False) as z:
        frozen_ids = z['ids'].tolist()
        if frozen_ids != ids:
            raise ValueError('Frozen scenario vector IDs/order differ from reference fingerprints.')
        vectors = dict(zip(frozen_ids, z['vectors']))
    for sid in ids:
        if sid not in docs or docs[sid] != frozen_docs.get(sid):
            raise ValueError(f'Frozen document incompatibility for {sid}; rebuild frozen vectors for changed text.')
    topo = read(data_dir / 'topology_index.json')
    adj = read(data_dir / 'partition_adjacency.json')['adjacency_matrix']
    sensor_summary = read(data_dir / 'sensor_summary.json')
    sp = {n: int(p) for p, d in sensor_summary['partition_details'].items() for n in d['sensor_nodes']}
    return fp, docs, vectors, topo, adj, np.array([sp[n] for n in fp['sensor_nodes']])


def validate_observations(arr, fingerprints, sensor_nodes):
    """Require the original complete seed/query grid and exact simulated observations."""
    n = len(fingerprints)
    expected_seed = np.repeat(SEEDS, n)
    expected_qi = np.tile(np.arange(n), len(SEEDS))
    if not np.array_equal(arr['seed'], expected_seed) or not np.array_equal(arr['query_index'], expected_qi):
        raise ValueError('Frozen observations have incompatible seed/query indexing.')
    D = np.asarray([f['sensor_fingerprint'] for f in fingerprints], float)
    expected = np.concatenate([D + np.random.default_rng(s).normal(0, .02, (n, len(sensor_nodes))) for s in SEEDS])
    if not np.array_equal(arr['observations'], expected):
        raise ValueError('Frozen observations do not match reference fingerprints and fixed noise seeds.')
    if len(arr['embeddings']) != len(expected) or not np.isfinite(arr['embeddings']).all():
        raise ValueError('Frozen query embeddings are incomplete or nonfinite.')


def load_observations(frozen_dir, fingerprints, sensor_nodes):
    with np.load(Path(frozen_dir) / 'query_observations_embeddings.npz', allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    validate_observations(arr, fingerprints, sensor_nodes)
    return arr


def build_pools(fingerprints, queries, documents, vectors, topo, adj, protocols=PROTOCOLS):
    pools, mapping = {}, {}
    for qi, q in enumerate(queries):
        for protocol in protocols:
            sub = [f for f in fingerprints if keep(f, q, protocol)]
            ids = [f['scenario_id'] for f in sub]
            key = digest(ids)
            if key not in pools:
                pools[key] = Pool(sub, {s: documents[s] for s in ids}, {s: vectors[s] for s in ids}, topo, adj)
            mapping[qi, protocol] = key
    return pools, mapping

def verify_model_checkpoint(model_dir, expected_files):
    """Require the audited fixed-revision model bytes, including for explicit local paths."""
    model_dir = Path(model_dir); actual = {}
    for relative, sha in expected_files.items():
        p = model_dir.joinpath(*relative.replace('\\', '/').split('/'))
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError(f'Model checkpoint differs from fixed revision {MODEL_REVISION}: {relative}')
        actual[relative] = sha
    return actual
