"""Shared publication-size style and auditable export for the visual revision."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

from _paths import INPUT as ROOT, OUTPUT as PKG, FIGDIR, SD, QA
MM = 1 / 25.4
INK = '#171717'
TEAL = '#269D90'
RED = '#D73027'
BLUE = '#4575B4'
PURPLE = '#9C7CB5'
GREY = '#A8ADB2'
PALE = '#EDF7F4'
ORANGE = '#FC8D59'
YELLOW = '#FEE090'
GREEN = '#76B83F'
CYAN = '#02B0DB'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'axes.titleweight': 'bold', 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'legend.frameon': False, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'stix', 'figure.dpi': 160, 'savefig.dpi': 600,
    'savefig.bbox': None,
})

def figure(height_mm=145):
    return plt.figure(figsize=(183*MM, height_mm*MM))

def letter(ax, value, x=-.13, y=1.08):
    ax.text(x, y, value, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='left', va='top', color=INK)

def clean(ax, grid=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(pad=2)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='y', color='#E9EDF0', lw=.4)
    else:
        ax.grid(False)

def export(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for obj in fig.findobj(match=Text):
        s = obj.get_text().strip()
        if not s or not obj.get_visible():
            continue
        box = obj.get_window_extent(renderer)
        items.append({'text': s, 'size_pt': obj.get_fontsize(),
                      'bbox_px': [box.x0, box.y0, box.x1, box.y1]})
    width, height = fig.canvas.get_width_height()
    overflow = [item['text'] for item in items if
                item['bbox_px'][0] < -1 or item['bbox_px'][1] < -1 or
                item['bbox_px'][2] > width+1 or item['bbox_px'][3] > height+1]
    report = {'figure': name, 'width_mm': float(fig.get_figwidth()*25.4),
              'height_mm': float(fig.get_figheight()*25.4),
              'text_word_count': sum(len(re.findall(r'\b[\w]+\b', t['text'])) for t in items),
              'min_text_pt': min((t['size_pt'] for t in items), default=None),
              'overflow': overflow, 'text': items}
    (QA / (name+'.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    fig.savefig(FIGDIR / f'{name}.png', dpi=600, facecolor='white', bbox_inches=None)
    fig.savefig(FIGDIR / f'{name}.pdf', facecolor='white', bbox_inches=None)
    fig.savefig(FIGDIR / f'{name}.svg', facecolor='white', bbox_inches=None)
    print(f'{name}: {report["text_word_count"]} words; {report["width_mm"]:.1f} x {report["height_mm"]:.1f} mm; overflow={overflow}')
    plt.close(fig)
