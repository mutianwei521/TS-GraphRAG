"""Run the fixed strict holdout/sensitivity benchmark using retained artifacts."""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .engine import (Pool, Config, CHANNELS, SEEDS, PROTOCOLS, MODEL_ID, MODEL_REVISION, DEFAULT_DATA_DIR,
                     DEFAULT_FROZEN_DIR, FrozenEmbeddings, build_query_text, build_pools, load_sources,
                     load_observations, rank_metrics, keep, verify_model_checkpoint, digest)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid():
    g = {'default': Config()}
    for b in [0, .05, .3, .5, .75]: g[f'beta_{b}'] = Config(beta=b)
    for n in [2, 3, 5]: g[f'graph_steps_{n}'] = Config(steps=n)
    for m in [0, .5, 1.5, 2]: g[f'semantic_weight_multiplier_{m}'] = Config(semantic_multiplier=m)
    for k in [3, 5, 20, 40]: g[f'top_k_{k}'] = Config(top_k=k)
    for lo, hi in [(.05, .5), (.2, 2), (.05, 1), (.2, 1), (.1, .5), (.1, 2)]:
        g[f'signal_thresholds_{lo}_{hi}'] = Config(weak=lo, strong=hi)
    for t in [.3, .5, .7, .85]: g[f'semantic_cosine_threshold_{t}'] = Config(semantic_threshold=t)
    for c in CHANNELS: g[f'leave_out_{c}'] = Config(drop_channel=c)
    g['uniform_channel_weights'] = Config(uniform_weights=True)
    return g


def write_tables(data, out, stem, groupname, nqueries):
    rows = []
    for (seed, proto, method), vals in data.items():
        a = np.array(vals)
        rows.append(dict(seed=seed, protocol=proto, **{groupname: method}, n=len(a),
                         top1_pct=100*a[:, 0].mean(), top3_pct=100*a[:, 1].mean(), mrr_pct=100*a[:, 2].mean()))
    if not rows: return []
    def write(name, values):
        with (out / name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    write(stem + '_by_seed.csv', rows)
    summary = []
    for proto, method in sorted({(r['protocol'], r[groupname]) for r in rows}):
        rr = [r for r in rows if r['protocol'] == proto and r[groupname] == method]
        row = dict(protocol=proto, **{groupname: method}, n_noise_seeds=len(rr), n_queries_each=nqueries)
        for m in ['top1_pct', 'top3_pct', 'mrr_pct']:
            vals = [r[m] for r in rr]
            row[m+'_mean'] = float(np.mean(vals)); row[m+'_sd'] = float(np.std(vals, ddof=1))
        summary.append(row)
    write(stem + '_summary.csv', summary)
    return summary


def initialize_embeddings(args, pools, arr, fingerprints, sensor_nodes, sensor_parts, out):
    if args.embedding_mode == 'replay':
        emb = FrozenEmbeddings.load(args.frozen_dir / 'retained_community_embeddings.npz')
        for pool in pools.values(): pool.set_embeddings(emb)
        return arr, emb, None
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(2)
    model_arg = args.model or MODEL_ID
    expected_files = json.loads((args.frozen_dir / 'source_manifest.json').read_text(encoding='utf-8'))['model_files_sha256']
    if Path(model_arg).is_dir():
        model_path = Path(model_arg)
    else:
        if model_arg != MODEL_ID: raise ValueError(f'Rebuild requires {MODEL_ID} at revision {MODEL_REVISION}.')
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=not args.allow_download,
                                           allow_patterns=[n.replace('\\', '/') for n in expected_files]))
    model_manifest = verify_model_checkpoint(model_path, expected_files)
    save_json(out / 'embedding_model_manifest.json', dict(revision=MODEL_REVISION, files_sha256=model_manifest))
    model = SentenceTransformer(str(model_path), local_files_only=True, device='cpu')
    documents = {d['scenario_id']: d for p in pools.values() for d in p.docs}
    ids = [f['scenario_id'] for f in fingerprints]
    scenario_E = model.encode([documents[s]['summary_text'] for s in ids], batch_size=32, show_progress_bar=True)
    scenario_vectors = dict(zip(ids, scenario_E))
    np.savez_compressed(out / 'frozen_scenario_vectors.npz', ids=np.array(ids), vectors=scenario_E)
    save_json(out / 'frozen_scenario_documents.json', [documents[s] for s in ids])
    for p in pools.values():
        p.E = np.array([scenario_vectors[s] for s in p.ids], dtype=np.float32)
        p.artifact_digest = digest(dict(ids=p.ids, fps=p.fps, docs=p.docs, summaries=p.summary_texts,
                                       vectors_sha=hashlib.sha256(p.E.tobytes()).hexdigest()))
    texts = sorted({t for p in pools.values() for t in p.summary_texts.values()})
    vectors = model.encode(texts, batch_size=128, show_progress_bar=True)
    emb = FrozenEmbeddings({hashlib.sha256(t.encode()).hexdigest(): v for t, v in zip(texts, vectors)})
    for p in pools.values(): p.set_embeddings(emb)
    np.savez_compressed(out / 'retained_community_embeddings.npz', text_sha256=np.array(list(emb)), vectors=np.array(list(emb.values())))
    sp = dict(zip(sensor_nodes, sensor_parts.tolist()))
    query_texts = [build_query_text(dict(zip(sensor_nodes, y)), fingerprints[int(qi)]['demand_period'], sp)
                   for qi, y in zip(arr['query_index'], arr['observations'])]
    arr = dict(arr, embeddings=model.encode(query_texts, batch_size=128, show_progress_bar=True))
    np.savez_compressed(out / 'query_observations_embeddings.npz', **arr)
    return arr, emb, scenario_vectors


def counterfactual_checks(fps, queries, docs, vectors, topo, adj, pools, mapping, emb, arr, sensor_parts):
    """Poison all excluded records before filtering, then check artifacts and predictions."""
    qi = min(31, len(queries)-1); q = queries[qi]
    y, qe = arr['observations'][qi], arr['embeddings'][qi]
    checks = []
    for proto in PROTOCOLS[1:]:
        mutated, md, mv = copy.deepcopy((fps, docs, vectors))
        excluded = 0
        for f in mutated:
            if not keep(f, q, proto):
                excluded += 1; sid = f['scenario_id']; new = 'EXCLUDED_POISON_' + sid
                f.update(sensor_fingerprint=[1e6]*len(y), max_pressure_drop=9e9, num_affected_nodes=999999, scenario_id=new)
                md[new] = md.pop(sid); md[new].update(scenario_id=new, summary_text='POISON '*100, max_pressure_drop=9e9, num_affected_nodes=999999)
                mv[new] = np.full_like(mv.pop(sid), 9999.)
        sub = [f for f in mutated if keep(f, q, proto)]; ids = [f['scenario_id'] for f in sub]
        test = Pool(sub, {s: md[s] for s in ids}, {s: mv[s] for s in ids}, topo, adj); test.set_embeddings(emb)
        p = pools[mapping[qi, proto]]
        ra, sa, _ = p.score(p.prepare(y, q['demand_period'], qe, sensor_parts))
        rb, sb, _ = test.score(test.prepare(y, q['demand_period'], qe, sensor_parts))
        if p.artifact_digest != test.artifact_digest or ra != rb or not np.array_equal(sa, sb):
            raise AssertionError(f'Excluded-record poisoning affected {proto}')
        checks.append(dict(protocol=proto, excluded_records=excluded, artifact_identical=True, ranking_and_scores_identical=True))
    return checks


def compare_expected(prediction_path, expected_path):
    """Compare recomputed complete rankings to archived query-level outputs."""
    expected = {}
    with gzip.open(expected_path, 'rt', encoding='utf-8') as f:
        for r in map(json.loads, f): expected[r['seed'], r['query_index'], r['protocol'], r['method']] = r['ranking']
    mismatches = []; count = 0
    with gzip.open(prediction_path, 'rt', encoding='utf-8') as f:
        for r in map(json.loads, f):
            key = r['seed'], r['query_index'], r['protocol'], r['method']; count += 1
            if expected.get(key) != r['ranking']:
                if len(mismatches) < 20: mismatches.append(dict(key=key, actual=r['ranking'], expected=expected.get(key)))
    return dict(rankings_compared=count, rankings_equal=not mismatches, first_mismatches=mismatches)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--frozen-dir', type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/strict_engine'))
    parser.add_argument('--study', choices=['holdout', 'sensitivity', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=0, help='First N queries per seed; retained corpus is never truncated.')
    parser.add_argument('--embedding-mode', choices=['replay', 'rebuild'], default='replay')
    parser.add_argument('--model', help='Optional fixed-checkpoint local path or Hugging Face model ID for rebuild.')
    parser.add_argument('--allow-download', action='store_true', help='Allow explicit checkpoint download during embedding rebuild.')
    parser.add_argument('--legacy-attribution', action='store_true', help='Also run the explicitly unfiltered-semantic attribution diagnostic.')
    args = parser.parse_args(argv)
    if args.limit < 0: parser.error('--limit must be nonnegative')
    if args.embedding_mode != 'rebuild' and (args.model or args.allow_download): parser.error('--model/--allow-download require --embedding-mode rebuild')
    return args


def main(argv=None):
    args = parse_args(argv); out = args.output_dir; out.mkdir(parents=True, exist_ok=True); start = time.time()
    fpd, docs, vectors, topo, adj, sp = load_sources(args.data_dir, args.frozen_dir)
    fps = fpd['fingerprints']; queries = fps[:args.limit] if args.limit else fps
    cfgs = grid(); arr = load_observations(args.frozen_dir, fps, fpd['sensor_nodes'])
    pools, mapping = build_pools(fps, queries, docs, vectors, topo, adj)
    arr, emb, rebuilt_vectors = initialize_embeddings(args, pools, arr, fps, fpd['sensor_nodes'], sp, out)
    if rebuilt_vectors is not None: vectors = rebuilt_vectors
    design = dict(seeds=SEEDS, noise_std_m=.02, n_queries_per_seed=len(queries), protocols=PROTOCOLS,
                  sensitivity_protocols=['H0', 'H2', 'H4', 'H8'], sensitivity_grid={n: asdict(c) for n, c in cfgs.items()},
                  selection_policy='Prespecified one-factor sensitivity; default preserved; no test selection.',
                  embedding_mode=args.embedding_mode, embedding_checkpoint=MODEL_ID, embedding_revision=MODEL_REVISION,
                  H9_interpretation='Structural retained-label integrity control; not unseen-partition generalization.',
                  query_input='Sensor observations and observed demand period only.', LLM_calls=0,
                  replication='Five independent noise realizations on the same fixed scenarios; one network.', study=args.study,
                  legacy_attribution=args.legacy_attribution)
    save_json(out / 'frozen_design.json', design)
    with gzip.open(out / 'retained_provenance.jsonl.gz', 'wt', encoding='utf-8') as f:
        for pool in pools.values(): f.write(json.dumps(pool.provenance(), ensure_ascii=False)+'\n')
    h0 = pools[mapping[0, 'H0']]
    oldcomm = json.loads((args.data_dir / 'community_summaries.json').read_text(encoding='utf-8'))['level_1_communities']
    equal = {str(p): text == oldcomm[str(p)]['summary'] for p, text in h0.summary_texts.items()}
    if not all(equal.values()): raise ValueError('H0 deterministic community text differs from reference.')
    checks = dict(H0_summary_text_exact_match=equal,
                  excluded_counterfactual_tests=counterfactual_checks(fps, queries, docs, vectors, topo, adj, pools, mapping, emb, arr, sp))
    raw, sens, legacy = defaultdict(list), defaultdict(list), defaultdict(list)
    holdout = args.study in ['holdout', 'all']; sensitivity = args.study in ['sensitivity', 'all']
    predictions = gzip.open(out / 'predictions.jsonl.gz', 'wt', encoding='utf-8') if holdout else None
    sensitivities = gzip.open(out / 'sensitivity_predictions.jsonl.gz', 'wt', encoding='utf-8') if sensitivity else None
    legacy_file = gzip.open(out / 'legacy_attribution_predictions.jsonl.gz', 'wt', encoding='utf-8') if args.legacy_attribution else None
    nobs = 0; h9pass = 0; retained_violations = 0
    try:
        for seed, qi, y, qe in zip(arr['seed'], arr['query_index'], arr['observations'], arr['embeddings']):
            qi = int(qi); seed = int(seed)
            if qi >= len(queries): continue
            q = queries[qi]; truth = q['leak_partition']; nobs += 1
            if legacy_file:
                hd = h0.prepare(y, q['demand_period'], qe, sp); _, _, fullX = h0.score(hd, open_labels=True)
            for proto in PROTOCOLS:
                p = pools[mapping[qi, proto]]; d = p.prepare(y, q['demand_period'], qe, sp)
                if holdout:
                    rank, score, _ = p.score(d)
                    ranks = dict(fusion_strict=rank, projection=p.baseline_rank(d['projection_order']), cosine=p.baseline_rank(d['cosine_order']))
                    if proto == 'H9': ranks['fusion_open_topology'] = p.score(d, open_labels=True)[0]
                    for method, ranking in ranks.items():
                        met = rank_metrics(ranking, truth); raw[seed, proto, method].append(met)
                        if method != 'fusion_open_topology':
                            retained_violations += int(not set(ranking).issubset(set(p.labels)))
                            if proto == 'H9': assert met == [0, 0, 0.]; h9pass += 1
                        rec = dict(seed=seed, query_index=qi, query_id=q['scenario_id'], truth=truth, protocol=proto,
                                   method=method, pool_id=p.key, n_retained=len(p.ids), ranking=ranking,
                                   top1=met[0], top3=met[1], rr=met[2], rr_top10=rank_metrics(ranking[:10], truth)[2])
                        if method == 'fusion_strict': rec['scores'] = score
                        predictions.write(json.dumps(rec)+'\n')
                if sensitivity and proto in design['sensitivity_protocols']:
                    for name, cfg in cfgs.items():
                        rank = p.score(d, cfg)[0]; met = rank_metrics(rank, truth); sens[seed, proto, name].append(met)
                        sensitivities.write(json.dumps(dict(seed=seed, query_id=q['scenario_id'], protocol=proto, setting=name,
                                                             pool_id=p.key, ranking=rank, top1=met[0], top3=met[1], rr=met[2]))+'\n')
                if legacy_file:
                    openrank, _, X = p.score(d, open_labels=True); X[4:6] = fullX[4:6]
                    w = np.array([.25, .12, .08, .30, .10, .15] if d['max_obs'] >= 1 else
                                 [.20, .20, .15, .18, .10, .17] if d['max_obs'] >= .1 else [.15, .25, .20, .12, .10, .18])
                    pre = w @ X; A = p.A_full / np.maximum(p.A_full.sum(axis=1, keepdims=True), 1.)
                    oldrank = np.argsort(-(.85*pre + .15*(A@pre)), kind='stable').tolist()
                    for method, ranking in [('fusion_legacy_unfiltered', oldrank), ('fusion_filtered_open_topology', openrank)]:
                        met = rank_metrics(ranking, truth); legacy[seed, proto, method].append(met)
                        legacy_file.write(json.dumps(dict(seed=seed, query_id=q['scenario_id'], truth=truth, protocol=proto, method=method,
                                                          pool_id=p.key, ranking=ranking, top1=met[0], top3=met[1], rr=met[2],
                                                          rr_top10=rank_metrics(ranking[:10], truth)[2]))+'\n')
            if nobs % 270 == 0: print(f'{nobs} observations evaluated in {time.time()-start:.1f}s', flush=True)
    finally:
        for f in [predictions, sensitivities, legacy_file]:
            if f: f.close()
    summary = write_tables(raw, out, 'holdout', 'method', len(queries))
    write_tables(sens, out, 'sensitivity', 'setting', len(queries))
    write_tables(legacy, out, 'legacy_attribution', 'method', len(queries))
    checks.update(H9_zero_metric_checks=h9pass, retained_label_violations=retained_violations, total_query_protocol_pairs=nobs*10)
    if holdout and (args.frozen_dir / 'predictions.jsonl.gz').exists():
        checks['archived_ranking_parity'] = compare_expected(out / 'predictions.jsonl.gz', args.frozen_dir / 'predictions.jsonl.gz')
    save_json(out / 'integrity_checks.json', checks)
    sources = {str(p.resolve()): filehash(p) for d in [args.data_dir, args.frozen_dir] for p in d.iterdir() if p.is_file()}
    save_json(out / 'source_manifest.json', dict(files=sources, numpy_version=np.__version__, embedding_revision=MODEL_REVISION))
    save_json(out / 'run_completion.json', dict(finished=True, elapsed_seconds=time.time()-start, observations=nobs,
              query_protocol_pairs=nobs*10 if holdout else 0, sensitivity_query_settings=nobs*4*len(cfgs) if sensitivity else 0,
              output_sha256={p.name: filehash(p) for p in out.iterdir() if p.is_file() and p.name != 'run_completion.json'}))
    print(json.dumps(dict(completed=True, observations=nobs, holdout_rows=len(raw), sensitivity_settings=len(cfgs) if sensitivity else 0,
                         archived_ranking_parity=checks.get('archived_ranking_parity')), indent=2))
    return summary


if __name__ == '__main__':
    main()
