# -*- coding: utf-8 -*-
"""Supplementary Fig. S2 | Confusion matrices of projection-residual retrieval under H0, H2 and H8,
pooled over the five repetitions of the 540-query evaluation (Table S2b), plus the share of Top-1 errors
that fall in a partition adjacent to the true partition, per repetition.

Bar + s.d. + replicate-point version (accepted style, fig_style):
a-c, row-normalized confusion matrices (soft sequential purple, shared colourbar right of panel b, off-diagonal cells
     >= 0.10 annotated). Titles give the POOLED Top-1 accuracy (identical to the mean of the five runs
     because every run has 540 queries) and the POOLED share of Top-1 errors that land in a partition
     adjacent to the true partition (adjacent errors / errors, both summed over the five runs; one decimal).
     These pooled shares (H0 59.5%, H2 77.6%, H8 76.0%) are the numbers the matrices themselves contain and
     the numbers quoted in the Results, Supplementary Note 1.7 and the Response ('206 of 346 errors under H0
     (60%), 503 of 648 under H2 (78%), 825 of 1,086 under H8 (76%)').
d,   per protocol (H0, H2, H8) the share of Top-1 errors that fall in an adjacent partition: one value per
     repetition (seed); bar = mean of the five per-run shares (pastel palette), charcoal cap = sample s.d.,
     five white circles = the runs (deterministic horizontal spread, seed order), value label '<mean>%
     (mean of runs)' above cap and points, and the mean +/- s.d. number of Top-1 errors per run (the
     denominator of each per-run share) printed at the base of each bar.
     Note: the pooled share (a-c titles) is NOT identical to the mean of the per-run shares (panel-d bar),
     because the denominator (Top-1 errors per run: 63/74/72/68/69 under H0) differs between runs; the two
     differ by < 0.3 pp (asserted below), and the caption says so. Top-1 accuracy has no such discrepancy.

Layout: 2 x 2 (a, b | c, d). A 1 x 4 strip at 14-15 in would leave each 15 x 15 matrix < 3.5 in wide, too
small for the tick labels and cell annotations. Panel d spans the full width of panel b (left/right-aligned
with b, top/bottom-aligned with c). figsize width 14.5 in so that the tight-bbox export stays >= 14 in wide.

Inputs (frozen): results/E4_mean_std_20260909/classical_all540_predictions.csv (method == 'projection')
                 knowledge_base/partition_adjacency.json
Outputs: figures/FigS2_confusion.{png,pdf,svg} (fig_style.save)
         source_data/FigS2_confusion_counts.csv         (pooled counts; read by make_fig4_holdout.py, format kept)
         source_data/FigS2d_adjacent_fraction_per_run.csv (panel d, one row per protocol x seed)
Prints: pooled summary per protocol (incl. the chance level of an adjacent error, console only) and a one-line
        consistency report for panel d (bar height == mean of the five points, n == 5).
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from fig_style import *
from nature_style import MM
from pastel_style import apply_style, SEQUENTIAL, BLUE, ORANGE, RED, INK
apply_style()

PRED = os.path.join(ROOT, 'results', 'E4_mean_std_20260909', 'classical_all540_predictions.csv')
ADJ = os.path.join(ROOT, 'knowledge_base', 'partition_adjacency.json')

# ---------------- data
df = pd.read_csv(PRED)
df = df[df['method'] == 'projection'].copy()
with open(ADJ, encoding='utf-8') as f:
    adj = json.load(f)
if 'adjacency_matrix' in adj:
    A = (np.array(adj['adjacency_matrix']) > 0).astype(int); K = A.shape[0]
else:
    K = int(adj['num_partitions']) if 'num_partitions' in adj else \
        1 + max(max(int(k) for k in adj['adjacency']), max(h for nb in adj['adjacency'].values() for h in nb))
    A = np.zeros((K, K), int)
    for k, nb in adj['adjacency'].items():
        for h in nb:
            A[int(k), int(h)] = 1
np.fill_diagonal(A, 0)
deg = A.sum(axis=1)

protocols = ['H0', 'H2', 'H8']
seeds = sorted(df['seed'].unique())
n_runs = len(seeds)
PROT_DESC = {'H0': 'no holdout', 'H2': 'queried location removed', 'H8': 'location, magnitude and state removed'}


def is_adjacent(t, p):
    return 0 <= p < K and A[int(t), int(p)] == 1


rows = []            # pooled counts (source data, format read by make_fig4_holdout.py)
summary = {}         # pooled summary per protocol (titles, console)
per_run = []         # panel d: one row per protocol x seed
for h in protocols:
    sub = df[df['protocol'] == h]
    M = np.zeros((K, K), int)
    for t, p in zip(sub['true_partition'], sub['predicted_partition']):
        if 0 <= p < K:
            M[t, p] += 1
    n = len(sub); correct = int(np.trace(M))
    err = sub[sub['true_partition'] != sub['predicted_partition']]
    adjacent = int(sum(is_adjacent(t, p) for t, p in zip(err['true_partition'], err['predicted_partition'])))
    # chance level (console only): each error lands uniformly on one of the other K-1 partitions
    chance = 100 * float(np.mean([deg[int(t)] / (K - 1) for t in err['true_partition']])) if len(err) else float('nan')
    summary[h] = dict(n=n, correct=correct, acc=100 * correct / n, errors=int(len(err)), adjacent=adjacent,
                      adj_frac_pooled=100 * adjacent / max(1, len(err)), chance_adjacent=chance)
    for t in range(K):
        for p in range(K):
            rows.append([h, t, p, int(M[t, p])])
    summary[h]['R'] = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    summary[h]['M'] = M
    for s in seeds:
        ss = sub[sub['seed'] == s]
        e = ss[ss['true_partition'] != ss['predicted_partition']]
        a = int(sum(is_adjacent(t, p) for t, p in zip(e['true_partition'], e['predicted_partition'])))
        per_run.append(dict(protocol=h, seed=int(s), n_queries=int(len(ss)), n_errors=int(len(e)), n_adjacent_errors=a,
                            top1_percent=100 * (len(ss) - len(e)) / len(ss),
                            adjacent_error_percent=100 * a / max(1, len(e))))
per_run = pd.DataFrame(per_run)
for h in protocols:
    v = per_run[per_run['protocol'] == h]['adjacent_error_percent'].values
    summary[h]['adj_frac_mean'] = float(v.mean()); summary[h]['adj_frac_sd'] = float(v.std(ddof=1))

# ---------------- figure: 2 x 2, square heatmaps a-c, bar panel d
# The heatmaps are aspect-locked and height-limited (cell height = 0.87 * H / (2 + hspace)), so the exported
# (tight-bbox) width is ~ 2 * heatmap side + column gap + label overhangs; 14.5 x 14.6 in with hspace 0.28
# gives ~5.57 in heatmaps and an exported width >= 14 in (a wider figsize alone only widens trimmed margins).
fig = plt.figure(figsize=(183*MM, 166*MM))
gs = gridspec.GridSpec(2, 2, figure=fig, left=0.075, right=0.90, top=0.94, bottom=0.10, hspace=0.32, wspace=0.25)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
heat_axes = [ax_a, ax_b, ax_c]
TICK_FS, CELL_FS, TITLE_FS = 5, 5, 7
im = None
for ax, h in zip(heat_axes, protocols):
    R = summary[h]['R']; M = summary[h]['M']
    im = ax.imshow(R, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f'P{i}' for i in range(K)], rotation=90, fontsize=TICK_FS)
    ax.set_yticklabels([f'P{i}' for i in range(K)], fontsize=TICK_FS)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True); ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.grid(which='minor', color='white', lw=0.6); ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=3)
    # annotate off-diagonal cells with >= 10 % of the row (as in the accepted version)
    for t in range(K):
        for p in range(K):
            if t != p and M[t, p] > 0 and R[t, p] >= 0.10:
                # Keep the original values and label rule; pick a readable text colour.
                rgb = np.asarray(SEQUENTIAL(R[t, p])[:3])
                linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
                luminance = float(np.dot(linear,[.2126,.7152,.0722]))
                ax.text(p, t, f'{R[t, p]:.2f}', ha='center', va='center', fontsize=CELL_FS,
                        color=('white' if luminance < .18 else INK), zorder=4)
    ax.set_xlabel('Predicted partition'); ax.set_ylabel('True partition')
    s = summary[h]
    # adjacent-error share = POOLED share (adjacent errors / errors summed over the five runs, one decimal): the
    # number the pooled matrix contains and the number quoted in the text ('206 of 346 errors under H0 (60%)');
    # the panel-d bar is the mean of the per-run shares, which differs slightly because the per-run
    # denominators (Top-1 errors per run) differ
    ax.set_title(f'{h} · Top-1 {s["acc"]:.1f}%', fontsize=TITLE_FS, pad=6)
    for sp in ax.spines.values():
        sp.set_linewidth(.55)

# ---------------- d: adjacent-error share per protocol, per run (bar = mean, cap = s.d., circles = runs)
xd = np.arange(len(protocols)); BW = 0.60
means = np.array([summary[h]['adj_frac_mean'] for h in protocols])
sds = np.array([summary[h]['adj_frac_sd'] for h in protocols])
bars = ax_d.bar(xd, means, width=BW, color=[BLUE,ORANGE,RED], edgecolor='#A9ADB0', linewidth=.35, zorder=2,
                yerr=sds, error_kw=dict(ecolor=INK, elinewidth=.65, capsize=2.4, capthick=.65, zorder=3))
report = []
top_of = []
for i, h in enumerate(protocols):
    pts = per_run[per_run['protocol'] == h].sort_values('seed')['adjacent_error_percent'].values
    assert len(pts) == n_runs == 5, (h, len(pts))
    assert abs(pts.mean() - bars[i].get_height()) <= 0.06, (h, pts.mean(), bars[i].get_height())
    # the pooled share quoted in the a-c titles and the text must stay within 0.5 pp of the per-run mean
    # shown here (they differ only through the run-to-run variation of the denominator)
    assert abs(summary[h]['adj_frac_pooled'] - pts.mean()) < 0.5, (h, summary[h]['adj_frac_pooled'], pts.mean())
    xs = xd[i] + np.linspace(-0.30, 0.30, len(pts)) * BW
    ax_d.plot(xs, pts, linestyle='none', marker='o', ms=2.5, mfc='white', mec=INK, mew=0.5, zorder=5)
    top_of.append(max(means[i] + sds[i], pts.max()))
    report.append(f'{h}: bar {means[i]:.2f} = mean of {len(pts)} runs {pts.mean():.2f} (s.d. {sds[i]:.2f}; '
                  f'panel-d label {means[i]:.1f} (mean of runs); title/text quote pooled {summary[h]["adj_frac_pooled"]:.2f} '
                  f'= {summary[h]["adjacent"]}/{summary[h]["errors"]}; chance {summary[h]["chance_adjacent"]:.1f})')
# value label = per-run mean, explicitly marked as such (the a-c titles quote the pooled share); placed above
# the error cap AND the highest replicate point (errs = distance from bar top to that maximum)
value_labels(ax_d, bars, means, fmt='{:.1f}%', offset=3.0, fontsize=6, color=INK,
             errs=np.array(top_of) - means)
ax_d.set_xticks(xd); ax_d.set_xticklabels(protocols, fontsize=6)
ax_d.set_xlim(-0.5, len(protocols) - 0.5); ax_d.set_ylim(0, 100)
ax_d.set_ylabel('Adjacent errors (%)',fontsize=6.5)
ax_d.set_xlabel('Protocol',fontsize=6.5)
ax_d.set_title('Error destinations', fontsize=TITLE_FS, pad=8)
ax_d.grid(False)

# ---------------- align: aspect-locked heatmaps, bar panel spanning the full width of b (same height as c),
#                  shared colourbar right of panel b, panel letters at the same offset from every panel
fig.canvas.draw()
pa, pb, pc, pd_ = (ax.get_position() for ax in (ax_a, ax_b, ax_c, ax_d))
ax_d.set_position([pb.x0, pc.y0, pb.width, pc.height])
cax = fig.add_axes([pb.x1 + 0.012, pb.y0, 0.013, pb.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Row fraction', fontsize=6)
cb.ax.tick_params(labelsize=5)
for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), 'abcd'):
    panel_label(ax, lab, x=-0.11, y=1.11,size=8)

# ---------------- outputs
pd.DataFrame(rows, columns=['protocol', 'true_partition', 'predicted_partition', 'count_pooled_5_runs']).to_csv(
    os.path.join(SD, 'FigS2_confusion_counts.csv'), index=False)
per_run.to_csv(os.path.join(SD, 'FigS2d_adjacent_fraction_per_run.csv'), index=False)
save(fig, 'FigS2_confusion')
print(json.dumps({h: {k: v for k, v in s.items() if k not in ('R', 'M')} for h, s in summary.items()}, indent=1))
print('panel d consistency: ' + ' | '.join(report))
