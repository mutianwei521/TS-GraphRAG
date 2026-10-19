"""Shared quantitative-figure colors sampled from the user-provided reference."""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap

PALETTE = {
    'purple': '#BAACDC', 'blue': '#A5C1E3', 'light_blue': '#B4CDE5',
    'mint': '#94DACE', 'yellow': '#FFE19B', 'coral': '#FFB7B7',
    'peach': '#FFC184', 'violet': '#A79FCD', 'grey': '#CBD0D5',
}
INK = '#3F3F3F'
BLUE = PALETTE['blue']
PURPLE = PALETTE['purple']
TEAL = PALETTE['mint']
RED = PALETTE['coral']
ORANGE = PALETTE['peach']
YELLOW = PALETTE['yellow']
GREY = PALETTE['grey']
GREEN = PALETTE['mint']
PALE = '#F4F2F8'

METHOD_COLORS = {
    'Projection residual': PURPLE,
    'Cosine 1-NN': BLUE,
    'Cosine k-NN (k=5)': PALETTE['light_blue'],
    'Random Forest (100 trees)': TEAL,
    'SVM (RBF)': ORANGE,
    'MLP (128,64)': RED,
    'Nearest Centroid': YELLOW,
    'Logistic Regression': '#D9D3EC',
    'GCN+': '#C0DCD8',
    'GAT+': '#E3CCE4',
    'GraphSAGE+': PALETTE['violet'],
}
PHYSICS_COLORS = {
    'projection': PURPLE, 'cosine 1-NN': BLUE, 'WLS/NNLS': YELLOW,
    'Mahalanobis': ORANGE, 'Bayes-marginal': RED, 'KDE': TEAL,
    'graph-reg 1-NN': PALETTE['violet'], 'label-prop': '#C0DCD8',
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'reference_purple', ['#FAF9FD', '#E2DDF0', PURPLE, '#7771B3'])
DIVERGING = LinearSegmentedColormap.from_list(
    'reference_coral_white_purple', [RED, '#FFFDFC', '#AAA5E9'])


def darker(color, factor=.70):
    """Related, readable stroke color for fine lines on a white background."""
    return tuple(v * factor for v in to_rgb(color))


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.edgecolor': '#666666', 'axes.linewidth': .55,
        'xtick.major.width': .55, 'ytick.major.width': .55,
        'hatch.linewidth': .4, 'grid.color': '#ECECEC',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
