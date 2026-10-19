# -*- coding: utf-8 -*-
"""Shared style for the revised main figures.

Reproduces the typography and palette conventions of the previously accepted
figure set (Times New Roman serif, black-edged bars with value labels, warm-to-cool
RdYlBu gradient, hatched 'ours' bar, dashed light grid, bold serif panel letters).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PKGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from _paths import INPUT, OUTPUT, FIGDIR as _FIGDIR, SD as _SD
ROOT = str(INPUT)
PACKAGE = str(OUTPUT)
FIGDIR = str(_FIGDIR)
SD = str(_SD)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

C = {
    'ours': '#d73027',      # deep red (projection residual / ours)
    'orange': '#fc8d59',
    'yellow': '#fee090',
    'paleblue': '#e0f3f8',
    'lightblue': '#91bfdb',
    'blue': '#4575b4',
    'teal': '#2a9d8f',
    'purple': '#8e6bb3',
    'grey': '#999999',
    'dark': '#333333',
    'grid': '#e0e0e0',
    'green': '#27AE60',
    'amber': '#E67E22',
}
# Method family colours used in every comparison panel
METHOD_COLORS = {
    'Projection residual': '#d73027',
    'Cosine 1-NN': '#f46d43',
    'Cosine k-NN (k=5)': '#fdae61',
    'Random Forest (100 trees)': '#fee090',
    'Logistic Regression': '#e0f3f8',
    'Nearest Centroid': '#c7dfe8',
    'MLP (128,64)': '#abd9e9',
    'SVM (RBF)': '#74add1',
    'GraphSAGE+': '#4575b4',
    'GAT+': '#313695',
    'GCN+': '#6a51a3',
}
SHORT = {
    'Projection residual': 'Projection residual (ours)',
    'Cosine 1-NN': 'Cosine 1-NN',
    'Cosine k-NN (k=5)': 'Cosine k-NN (k = 5)',
    'Random Forest (100 trees)': 'Random Forest',
    'Logistic Regression': 'Logistic Regression',
    'Nearest Centroid': 'Nearest Centroid',
    'MLP (128,64)': 'MLP (128-64)',
    'SVM (RBF)': 'SVM (RBF)',
    'GraphSAGE+': 'GraphSAGE+',
    'GAT+': 'GAT+',
    'GCN+': 'GCN+',
}
PROT_LABEL = {'H0': 'H0\nnone', 'H1': 'H1\nscenario', 'H2': 'H2\nlocation', 'H3': 'H3\nmagnitude', 'H4': 'H4\nstate',
              'H5': 'H5\nloc. $\\cup$ mag.', 'H6': 'H6\nloc. $\\cup$ state', 'H7': 'H7\nmag. $\\cup$ state', 'H8': 'H8\nall three', 'H9': 'H9\npartition'}


PROT_NARROW = {'H0': 'H0\nnone', 'H1': 'H1\nscen.', 'H2': 'H2\nloc.', 'H3': 'H3\nmag.', 'H4': 'H4\nstate', 'H5': 'H5\nL$\\,\\cup\\,$M', 'H6': 'H6\nL$\\,\\cup\\,$S', 'H7': 'H7\nM$\\,\\cup\\,$S', 'H8': 'H8\nL$\\,\\cup\\,$M$\\,\\cup\\,$S'}


def gradient_colors(values, vmin=0, vmax=95, cmap='RdYlBu_r'):
    norm = plt.Normalize(vmin, vmax)
    return [getattr(cm, cmap)(norm(v)) for v in values]


def style_ax(ax, axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis=axis, linestyle='--', color=C['grid'], alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.12, y=1.05, size=16):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight='bold', va='top', ha='right')


def value_labels(ax, bars, values, fmt='{:.1f}', offset=1.0, fontsize=7.5, color='#333333', horizontal=False, errs=None):
    for i, (bar, v) in enumerate(zip(bars, values)):
        e = errs[i] if errs is not None else 0
        if horizontal:
            ax.text(bar.get_width() + e + offset, bar.get_y() + bar.get_height() / 2, fmt.format(v),
                    va='center', ha='left', fontsize=fontsize, color=color)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + offset, fmt.format(v),
                    ha='center', va='bottom', fontsize=fontsize, color=color)


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        dpi = 600 if ext == 'png' else 300
        path = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(path, dpi=dpi, bbox_inches=None, facecolor='white')
    plt.close(fig)
    print('saved', name)


def parse_pm(cell):
    """'87.2 ± 0.8' -> (87.2, 0.8)"""
    import re
    m = re.match(r'\s*([\d.]+)\s*±\s*([\d.]+)', str(cell))
    return float(m.group(1)), float(m.group(2))
