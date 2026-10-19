"""Redraw the complete prespecified fusion sensitivity experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _paths import INPUT, OUTPUT
SOURCE = INPUT / 'sensitivity_standalone'
OUT = OUTPUT / 'figS6_standalone_20261017'
BASE = OUT
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
WIDTH_MM, HEIGHT_MM = 183, 167
PROTOCOLS = ['H0', 'H2', 'H4', 'H8']
COLORS = ['#4C88B0', '#58A38E', '#D88F68', '#9B7DB5']
MARKERS = ['o', 's', '^', 'D']
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5,
    'xtick.labelsize': 6.1, 'ytick.labelsize': 6.1,
    'axes.linewidth': .55, 'axes.edgecolor': '#7E858B',
    'axes.labelcolor': '#363C42', 'text.color': '#28313B',
    'xtick.color': '#50565D', 'ytick.color': '#50565D',
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'mathtext.fontset': 'dejavusans',
})

# Pair every setting with its own seed/protocol baseline before estimating SD.
raw = pd.read_csv(SOURCE / 'strict_engine_sensitivity_by_seed.csv')
published = pd.read_csv(SOURCE / 'strict_engine_sensitivity_summary.csv')
assert len(raw) == 680 and len(published) == 136
assert not raw.duplicated(['seed', 'protocol', 'setting']).any()
assert set(raw.seed) == {41, 141, 241, 341, 441}
assert set(raw.protocol) == set(PROTOCOLS) and (raw.n == 540).all()
defaults = raw[raw.setting.eq('default')][['seed', 'protocol', 'top1_pct']].rename(columns={'top1_pct': 'default_top1_pct'})
paired = raw.merge(defaults, on=['seed', 'protocol'], validate='many_to_one')
paired['paired_delta_pp'] = paired.top1_pct - paired.default_top1_pct
agg = paired.groupby(['protocol', 'setting'], sort=False).agg(
    mean_delta_pp=('paired_delta_pp', 'mean'),
    sd_delta_pp=('paired_delta_pp', 'std'),
    n_noise_seeds=('paired_delta_pp', 'size'),
    absolute_mean_top1_pct=('top1_pct', 'mean'),
    absolute_sd_top1_pct=('top1_pct', 'std'),
).reset_index()
assert len(agg) == 136 and (agg.n_noise_seeds == 5).all()
check = agg.merge(published, on=['protocol', 'setting'], validate='one_to_one')
np.testing.assert_allclose(check.absolute_mean_top1_pct, check.top1_pct_mean, atol=1e-11)
np.testing.assert_allclose(check.absolute_sd_top1_pct, check.top1_pct_sd, atol=1e-11)
stats = agg.set_index(['protocol', 'setting'])
base_accuracy = {p: float(stats.loc[(p, 'default'), 'absolute_mean_top1_pct']) for p in PROTOCOLS}
paired.to_csv(OUT / 'FigS6_paired_seed_data.csv', index=False)
agg.to_csv(OUT / 'FigS6_plot_source_data.csv', index=False)

fig = plt.figure(figsize=(7.20472440945, 6.57480314961))  # 183 × 167 mm.
def figtext(x_mm, y_mm, text, **kw):
    return fig.text(x_mm / WIDTH_MM, y_mm / HEIGHT_MM, text, **kw)

handles = [Line2D([], [], color=c, marker=m, markersize=3.7, linewidth=1.1,
                  label=f'{p} ({base_accuracy[p]:.2f}%)') for p, c, m in zip(PROTOCOLS, COLORS, MARKERS)]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .943), ncol=4,
           frameon=False, fontsize=6.8, columnspacing=1.9, handletextpad=.65,
           handlelength=1.65)

seen_settings = set()
data_extents = []
def axes_at(col, row):
    left = 14 + col * 58.0
    bottom = [113, 64][row]
    return fig.add_axes([left / WIDTH_MM, bottom / HEIGHT_MM, 46.2 / WIDTH_MM, 34 / HEIGHT_MM]), left, bottom

def setup(ax, title, letter, left, bottom, ylim, yticks):
    figtext(left - 6.5, bottom + 36.5, letter, fontsize=8, fontweight='bold', va='baseline')
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel('Δ Top-1 (pp)', labelpad=2)
    ax.axhline(0, color='#68727D', linewidth=.7, linestyle=(0, (3, 2)), zorder=1)
    ax.yaxis.grid(False)
    ax.tick_params(length=2, width=.55, pad=2)
    ax.set_axisbelow(True)

def points(ax, xs, settings, joined=True, offsets=None):
    seen_settings.update(settings)
    for j, (p, color, marker) in enumerate(zip(PROTOCOLS, COLORS, MARKERS)):
        y = np.array([stats.loc[(p, s), 'mean_delta_pp'] for s in settings], dtype=float)
        sd = np.array([stats.loc[(p, s), 'sd_delta_pp'] for s in settings], dtype=float)
        x = np.array(xs, dtype=float) + (offsets[j] if offsets is not None else 0)
        ax.errorbar(x, y, yerr=sd, fmt=marker, linestyle='-' if joined else 'none',
                    color=color, linewidth=1.05, elinewidth=.55, capsize=1.2,
                    capthick=.55, markersize=3.1, markeredgewidth=.35,
                    markeredgecolor='white', zorder=3+j*.1)
        low, high = float((y-sd).min()), float((y+sd).max())
        assert low >= ax.get_ylim()[0] and high <= ax.get_ylim()[1], (p, settings, low, high)
        data_extents.append({'protocol': p, 'settings': settings, 'min_lower_whisker': low, 'max_upper_whisker': high})

def default_band(ax, x, width):
    ax.axvspan(x-width, x+width, color='#E8ECF0', alpha=.75, lw=0, zorder=0)

# a. Full range retains the large adverse response at beta 0.75.
ax, left, bottom = axes_at(0, 0)
setup(ax, 'Propagation strength', 'a', left, bottom, (-56, 5), [-50, -25, 0])
xs = [0, .05, .15, .3, .5, .75]
default_band(ax, .15, .018)
points(ax, xs, ['beta_0', 'beta_0.05', 'default', 'beta_0.3', 'beta_0.5', 'beta_0.75'])
ax.set_xlim(-.035, .79)
ax.set_xticks(xs, ['0', '.05', '.15', '.3', '.5', '.75'])
ax.set_xlabel('Propagation weight β', labelpad=3)

ax, left, bottom = axes_at(1, 0)
setup(ax, 'Propagation depth', 'b', left, bottom, (-2.4, 1.7), [-2, -1, 0, 1])
default_band(ax, 1, .15)
points(ax, [1, 2, 3, 5], ['default', 'graph_steps_2', 'graph_steps_3', 'graph_steps_5'])
ax.set_xlim(.65, 5.3)
ax.set_xticks([1, 2, 3, 5])
ax.set_xlabel('Aggregation steps', labelpad=3)

ax, left, bottom = axes_at(2, 0)
setup(ax, 'Semantic / community weights', 'c', left, bottom, (-5, 3), [-4, -2, 0, 2])
default_band(ax, 1, .08)
points(ax, [0, .5, 1, 1.5, 2], ['semantic_weight_multiplier_0', 'semantic_weight_multiplier_0.5', 'default', 'semantic_weight_multiplier_1.5', 'semantic_weight_multiplier_2'])
ax.set_xlim(-.15, 2.15)
ax.set_xticks([0, .5, 1, 1.5, 2], ['0', '0.5', '1', '1.5', '2'])
ax.set_xlabel('Joint weight multiplier', labelpad=3)

ax, left, bottom = axes_at(0, 1)
setup(ax, 'Retrieval depth', 'd', left, bottom, (-3.2, 2), [-3, -2, -1, 0, 1, 2])
default_band(ax, 10, 1.25)
points(ax, [3, 5, 10, 20, 40], ['top_k_3', 'top_k_5', 'default', 'top_k_20', 'top_k_40'])
ax.set_xlim(1, 42)
ax.set_xticks([3, 5, 10, 20, 40])
ax.set_xlabel('Rank-vote depth k', labelpad=3)

ax, left, bottom = axes_at(1, 1)
setup(ax, 'Signal-amplitude boundaries', 'e', left, bottom, (-6, 5), [-4, -2, 0, 2, 4])
settings = ['signal_thresholds_0.05_0.5', 'signal_thresholds_0.1_0.5', 'signal_thresholds_0.05_1', 'default', 'signal_thresholds_0.2_1', 'signal_thresholds_0.1_2', 'signal_thresholds_0.2_2']
default_band(ax, 3, .40)
points(ax, np.arange(7), settings, joined=False, offsets=[-.24, -.08, .08, .24])
ax.set_xlim(-.6, 6.6)
ax.set_xticks(range(7), ['.05\n.5', '.1\n.5', '.05\n1', '.1\n1', '.2\n1', '.1\n2', '.2\n2'])
ax.tick_params(axis='x', labelsize=5.8)
ax.set_xlabel('Weak / strong boundaries (m)', labelpad=4)

ax, left, bottom = axes_at(2, 1)
setup(ax, 'Semantic similarity cutoff', 'f', left, bottom, (-1, 3), [-1, 0, 1, 2, 3])
points(ax, [.3, .5, .7, .85], ['semantic_cosine_threshold_0.3', 'semantic_cosine_threshold_0.5', 'semantic_cosine_threshold_0.7', 'semantic_cosine_threshold_0.85'])
ax.set_xlim(.25, .9)
ax.set_xticks([.3, .5, .7, .85], ['0.3', '0.5', '0.7', '0.85'])
ax.set_xlabel('Minimum semantic cosine', labelpad=3)
ax.text(.03, .94, 'Default: no cutoff', transform=ax.transAxes, fontsize=5.8,
        va='top', color='#6D7379', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .8})

# g. Categorical changes: horizontal effects and paired-noise SD, no false trend line.
ax = fig.add_axes([38 / WIDTH_MM, 10 / HEIGHT_MM, 140 / WIDTH_MM, 38 / HEIGHT_MM])
figtext(7.5, 51, 'g', fontsize=8, fontweight='bold')
settings = ['leave_out_prior', 'leave_out_centroid', 'leave_out_rank', 'leave_out_fingerprint', 'leave_out_semantic', 'leave_out_community', 'uniform_channel_weights']
labels = ['Remove sensor prior', 'Remove centroid matching', 'Remove rank matching', 'Remove fingerprints', 'Remove semantic retrieval', 'Remove community retrieval', 'Uniform channel weights']
seen_settings.update(settings)
ax.axvline(0, color='#68727D', linewidth=.8, linestyle=(0, (3, 2)), zorder=1)
for j, (p, color, marker) in enumerate(zip(PROTOCOLS, COLORS, MARKERS)):
    y = np.arange(len(settings)) + [-.24, -.08, .08, .24][j]
    means = np.array([stats.loc[(p, s), 'mean_delta_pp'] for s in settings], dtype=float)
    sds = np.array([stats.loc[(p, s), 'sd_delta_pp'] for s in settings], dtype=float)
    ax.hlines(y, 0, means, colors=color, linewidth=.55, alpha=.32, zorder=2)
    ax.errorbar(means, y, xerr=sds, fmt=marker, linestyle='none', color=color,
                markersize=3.2, markeredgecolor='white', markeredgewidth=.35,
                elinewidth=.65, capsize=1.3, capthick=.6, zorder=3+j*.1)
    assert (means-sds).min() >= -25 and (means+sds).max() <= 14
ax.set_ylim(6.65, -.65)
ax.set_xlim(-25, 14)
ax.set_xticks([-20, -10, 0, 10])
ax.set_yticks(range(len(settings)), labels)
ax.tick_params(axis='y', length=0, pad=4, labelsize=6.1)
ax.tick_params(axis='x', length=2, pad=2, width=.55)
ax.spines['left'].set_visible(False)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.set_xlabel('Change in Top-1 accuracy (percentage points)', labelpad=4)

assert seen_settings == set(raw.setting), (seen_settings ^ set(raw.setting))
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside_text = []
for label in fig.findobj(mpl.text.Text):
    if label.get_visible() and label.get_text().strip():
        bounds = label.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
            outside_text.append(label.get_text())
assert not outside_text, outside_text
stem = OUT / 'FigS6_engine_sensitivity_redrawn'
fig.savefig(str(stem) + '.pdf')
fig.savefig(str(stem) + '.svg')
fig.savefig(str(stem) + '.png', dpi=600)
fig.savefig(BASE / 'FigS6_preview.png', dpi=300)

caption = (
    'Fig. S6 | Parameter and channel sensitivity of the strict fusion component. '
    'a–d, Changes in Top-1 partition accuracy with propagation weight β (a), aggregation depth (b), '
    'the joint semantic/community weight multiplier (c) and rank-vote retrieval depth k (d). '
    'e, Boundary pairs for the maximum absolute query pressure change; the upper and lower tick numbers denote '
    'the weak and strong boundaries in metres, respectively. f, Minimum cosine similarity for semantic '
    'retrieval candidates; the default applies no cutoff. g, Individual channel removals and uniform '
    'channel weighting. Colours and marker shapes identify H0 (no exclusion), H2 (location exclusion), '
    'H4 (demand-state exclusion) and H8 (joint location, magnitude and demand-state exclusions). '
    'Points show mean within-seed differences from the unchanged default, and whiskers show ±1 sample '
    'standard deviation of these paired differences across five Gaussian measurement-noise realizations '
    '(s.d. 0.02 m; seeds 41, 141, 241, 341 and 441), each using the same 540 fixed simulated queries. '
    'Default Top-1 accuracies are given in the legend. Dashed zero lines mark no change; grey bands '
    'in a–e mark the default parameter settings. Connecting segments in a–d and f join tested values '
    'only; points are not joined across categories in e and g. Axis ranges differ between panels, and '
    'no values or whiskers are clipped. All 33 perturbations and the fixed default are retained; no '
    'configuration is selected using these results. Weight multipliers and channel removals are followed '
    'by weight normalization; fingerprint and semantic retrieval use 2k candidates. The semantic cutoff '
    'differs from the inactive scenario-graph construction threshold. This one-factor analysis evaluates '
    'the fusion component rather than the subsequently frozen projection-only estimator and does not '
    'test all parameter interactions or LLM confidence gating. Source data are provided as a Source Data file.'
)
(OUT / 'FigS6_caption.txt').write_text(caption+'\n', encoding='utf-8')
(OUT / 'FigS6_caption.md').write_text(caption+'\n', encoding='utf-8')
report = {
    'source_records': len(raw), 'settings_including_default': len(seen_settings),
    'perturbations': len(seen_settings)-1, 'protocols': PROTOCOLS,
    'seed_count': 5, 'queries_per_seed': 540,
    'paired_difference_before_aggregation': True,
    'all_settings_displayed': True, 'no_data_or_uncertainty_clipping': True,
    'canvas_mm': [WIDTH_MM, HEIGHT_MM], 'png_dpi': 600,
    'caption_word_count': len(caption.split()),
    'source_sha256': {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['strict_engine_sensitivity_by_seed.csv', 'strict_engine_sensitivity_summary.csv']},
}
(BASE / 'plot_integrity_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
plt.close(fig)
