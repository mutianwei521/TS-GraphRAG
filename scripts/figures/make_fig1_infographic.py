# -*- coding: utf-8 -*-
"""Fig. 1 | Conceptual infographic for the revised manuscript (final version).

Refined from fig1_design_A.py after the three-judge review.  Layout (portrait, 180 mm wide):
  title band     - full paper title (dark strip)
  a Challenge    (green)  - water losses, the inverse problem
  b Approach     (blue)   - TS-GraphRAG pipeline as numbered boxes + real EXA7 mini-map,
                            highlighted projection-residual formula box, optional layers
  c Evidence 1   (red)    - holdout protocol family: corpus-deletion schematic (H0-H9),
                            protocol-coloured circle markers with ± s.d. lines on a 0-100 axis
  d Evidence 2   (amber)  - supervised baselines under the same protocols (single-hue heat map)
  e Evidence 3   (teal)   - corpus allocation at a fixed budget (circle markers), A vs C90 callout
  message band   - take-home message + simulation-only caveat (dark strip)

Diagrams and typography use vector primitives; the full 540 x 30 matrix is a raster image.  The layout
is expressed in design units of a 7.2 x 8.70 in canvas and exported at exactly 180 mm width
(SCALE) with zero padding, so every point size in the script is the printed point size.
Inputs (frozen, read-only):
  dataset/Exa7.inp                                   hydraulic model (coordinates, links)
  partition_results_leiden/partition_summary.json    K = 15 node assignments
  knowledge_base/sensor_fingerprints.json            30 sensor junctions + template fingerprints
Output: figure1_dense_20261003/figures/Fig1_infographic.{png,pdf,svg}
"""
import os
import json
import numpy as np
from fig_style import *            # rcParams (Times New Roman), palette C, style_ax, save(), ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle, Circle,
                                Polygon, Ellipse, Arc)
import wntr

# --------------------------------------------------------------------------- canvas
W, H = 7.2, 8.70                                 # design units (the layout was drawn on a 7.2 x 8.70 in canvas)
SCALE = (180 / 25.4) / W                                # export exactly 180 mm (7.087 in) wide so 7 pt prints as 7 pt
OUTPUT_WIDTH_MM = 180
# Publication-safe export contract: editable SVG/PDF text and a 600-dpi PNG preview.
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.dpi': 600})
OUTPUT_FILES = ('figures/Fig1_infographic.png', 'figures/Fig1_infographic.pdf',
                'figures/Fig1_infographic.svg')
fig = plt.figure(figsize=(W * SCALE, H * SCALE), facecolor='white')
plt.rcParams['savefig.pad_inches'] = 0           # no tight-bbox padding: exported size = canvas size
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')

INK = C['dark']
GREY = '#6a6a6a'
SEP = '#d0d0d0'
GREEN, BLUE, RED, AMBER, TEAL = C['green'], C['blue'], C['ours'], C['amber'], C['teal']
# Semantic accents: cyan = network, yellow = library, violet = residual, coral = holdout.
CYAN, YELLOW, VIOLET, CORAL, MINT = '#16B6D4', '#F4C64E', '#7B4DB3', '#EF6C63', '#42B9A8'
FS = 7.5            # initial body default (overridden by dense layout)
FS7 = 7.5           # initial dense-text default (actual final floor: 6.5 pt)
FST = 8.5           # box / panel titles (pt)
LH = 0.115          # line pitch for 7.5 pt text (inches)
LH7 = 0.112         # line pitch for dense 7.5 pt text (inches)
MS = 8.0            # circle-marker diameter (pt) in the two small charts
MEW = 0.8           # circle-marker edge width (pt)

# protocol colour code (blue -> red ramp for H0-H8, black for the H9 audit); reused for bars
PROT_COL = {'H0': '#313695', 'H1': '#4575b4', 'H2': '#74add1', 'H3': '#abd9e9', 'H4': '#e0f3f8',
            'H5–H7': '#fee090', 'H8': '#d73027', 'H9': '#000000'}

TEXTS = []          # (Text artist, containing box or None) for the layout audit


def T(x, y, s, size=FS, box=None, **kw):
    kw.setdefault('ha', 'left')
    kw.setdefault('va', 'top')
    kw.setdefault('color', INK)
    kw.setdefault('zorder', 8)
    t = ax.text(x, y, s, fontsize=size, **kw)
    TEXTS.append((t, box))
    return t


def lines(x, y, rows, size=FS, lh=LH, box=None, **kw):
    """Stack of left-aligned lines; each row is a string or (string, dict-of-overrides)."""
    for i, row in enumerate(rows):
        if isinstance(row, tuple):
            s, extra = row
        else:
            s, extra = row, {}
        T(x, y - i * lh, s, size=size, box=box, **{**kw, **extra})
    return y - len(rows) * lh


def luminance(col):
    r, g, b = plt.matplotlib.colors.to_rgb(col)
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------------------- primitives
def rbox(x0, y0, w, h, fc='white', ec=INK, lw=1.0, r=0.08, ls='-', z=1, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def band(x0, y0, w, h, color, title, hh=0.30, r=0.10, size=12, lw=1.6, letter=None, toff=0.0):
    """Rounded panel with a filled header strip, a white bold title and a bold panel letter."""
    rbox(x0, y0, w, h, fc='white', ec=color, lw=lw, r=r, z=1)
    rbox(x0, y0 + h - hh, w, hh, fc=color, ec=color, lw=lw, r=r, z=2)
    # squaring rectangle for the lower header corners: unstroked so it stays inside the outline
    ax.add_patch(Rectangle((x0, y0 + h - hh), w, hh * 0.5, fc=color, ec='none', lw=0, zorder=2))
    T(x0 + w / 2 + toff, y0 + h - hh / 2, title, size=size, color='white', fontweight='bold',
      ha='center', va='center', zorder=9)
    if letter is not None:
        T(x0 + 0.11, y0 + h - hh / 2, letter, size=12, color='white', fontweight='bold',
          ha='left', va='center', zorder=9)
    return (x0, y0, x0 + w, y0 + h - hh)


def block_arrow(xc, ytop, ybot, color=RED, w=0.16, hw=0.36):
    hl = min(0.13, (ytop - ybot) * 0.65)
    ys = ybot + hl
    pts = [[xc - w / 2, ytop], [xc + w / 2, ytop], [xc + w / 2, ys], [xc + hw / 2, ys],
           [xc, ybot], [xc - hw / 2, ys], [xc - w / 2, ys]]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec='none', zorder=3))


def arrow(p0, p1, color=INK, lw=1.0, ls='-', z=4, ms=9, style='-|>'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        ls=ls, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def badge(x, y, n, color, r=0.095):
    ax.add_patch(Circle((x, y), r, fc=color, ec='white', lw=1.0, zorder=10))
    T(x, y, str(n), size=8, color='white', fontweight='bold', ha='center', va='center', zorder=11)


def stage(x0, y0, w, h, n, title, color=BLUE, ls='-', fc='white', tcolor=INK, lw=1.1):
    rbox(x0, y0, w, h, fc=fc, ec=color, lw=lw, r=0.07, ls=ls, z=2)
    box = (x0, y0, x0 + w, y0 + h)
    if n is not None:
        badge(x0 + 0.02, y0 + h - 0.02, n, color)
    T(x0 + (0.17 if n is not None else 0.1), y0 + h - 0.06, title, size=FST,
      fontweight='bold', color=tcolor, box=box)
    return box


def chip(xc, yc, label, w=0.25, h=0.10):
    col = PROT_COL[label]
    rbox(xc - w / 2, yc - h / 2, w, h, fc=col, ec='black', lw=0.5, r=0.03, z=6)
    T(xc, yc, label, size=FS7, fontweight='bold', ha='center', va='center',
      color='white' if luminance(col) < 0.55 else INK, zorder=9)


def metric(x, y, label, color, w=0.62, h=0.18, size=7.0):
    """Compact colour-coded metric chip used in place of explanatory sentences."""
    rbox(x, y, w, h, fc=color, ec='white', lw=0.6, r=0.07, z=7)
    T(x + w / 2, y + h / 2, label, size=size, color='white', fontweight='bold',
      ha='center', va='center', zorder=9)


def tiny_key(x, y, color, label, size=6.6):
    ax.add_patch(Circle((x, y), 0.035, fc=color, ec='white', lw=0.4, zorder=7))
    T(x + 0.08, y + 0.045, label, size=size, va='center', zorder=8)


# --------------------------------------------------------------------------- glyphs
def droplet(x, y, r, color):
    ax.add_patch(Circle((x, y), r, fc=color, ec='none', zorder=5))
    ax.add_patch(Polygon([[x - r * 0.93, y + r * 0.37], [x + r * 0.93, y + r * 0.37],
                          [x, y + r * 2.0]], closed=True, fc=color, ec='none', zorder=5))
    ax.add_patch(Ellipse((x - r * 0.38, y + r * 0.1), r * 0.32, r * 0.6, angle=25,
                         fc='white', ec='none', alpha=0.75, zorder=6))


def sensor(x, y, r=0.05, color=RED):
    ax.add_patch(Circle((x, y), r, fc=color, ec='black', lw=0.6, zorder=6))
    for k in (1.9, 2.7):
        ax.add_patch(Arc((x, y), 2 * r * k, 2 * r * k, theta1=15, theta2=75,
                         color=color, lw=0.9, zorder=6))


def wave(x, y, w, h, color):
    t = np.linspace(0, 1, 80)
    ax.plot(x + t * w, y + 0.5 * h * np.sin(2 * np.pi * 2 * t), color=color, lw=1.3, zorder=6,
            solid_capstyle='round')


def zone_glyph(x, y, s, color):
    pts = np.array([[-1.0, 0.2], [-0.55, 0.9], [0.5, 1.0], [1.0, 0.25], [0.7, -0.8], [-0.4, -1.0]]) * s
    ax.add_patch(Polygon(pts + [x, y], closed=True, fc=color, ec=INK, lw=0.6, alpha=0.55, zorder=5))
    ax.add_patch(Polygon([[x - 0.3 * s, y + 0.25 * s], [x + 0.3 * s, y + 0.25 * s], [x, y - 0.45 * s]],
                         closed=True, fc=RED, ec='none', zorder=6))
    ax.add_patch(Circle((x, y + 0.35 * s), 0.33 * s, fc=RED, ec='black', lw=0.5, zorder=7))


def cards(x, y, w, h, n=3, color=C['yellow']):
    """Stack of scenario cards (library glyph); (x, y) = lower-left of the front card."""
    for i in range(n - 1, -1, -1):
        d = i * 0.035
        rbox(x + d, y + d, w, h, fc=color if i == 0 else '#fff6d5', ec=INK, lw=0.6, r=0.025, z=5 + (n - i))
    for k in range(3):
        ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.06 - 0.055 * k] * 2, color=INK, lw=0.6, zorder=9)


def strip(x, y, w, h, vals, cmap='Blues', vmin=None, vmax=None):
    n = len(vals)
    cw = w / n
    norm = plt.Normalize(vmin if vmin is not None else min(vals), vmax if vmax is not None else max(vals))
    cmap_ = plt.get_cmap(cmap)
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, fc=cmap_(norm(v)), ec='white', lw=0.3, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=INK, lw=0.6, zorder=6))


def magnifier(x, y, r=0.085, color=BLUE):
    ax.add_patch(Circle((x, y), r, fc='white', ec=color, lw=2.0, zorder=6))
    ax.plot([x + r * 0.75, x + r * 1.9], [y - r * 0.75, y - r * 1.9], color=color, lw=3.0,
            solid_capstyle='round', zorder=6)


def sheet_stack(cx, ybot, deleted, n=3, nrow=6, ncol=3, cw=0.07, ch=0.04, off=0.028):
    """Reference-corpus glyph: one sheet per leak location (n shown), rows = magnitudes,
    columns = demand states.  deleted(s, r, c) -> True paints the cell red (s = 0 is the front
    sheet, r = 0 the top row, c = 0 the left column)."""
    sw, sh = ncol * cw, nrow * ch
    fx = cx - (sw + (n - 1) * off) / 2
    for s in range(n - 1, -1, -1):
        x0, y0 = fx + s * off, ybot + s * off
        z = 5 + 2 * (n - s)
        ax.add_patch(Rectangle((x0, y0), sw, sh, fc='white', ec='none', zorder=z))
        for r in range(nrow):
            for c in range(ncol):
                dead = deleted(s, r, c)
                ax.add_patch(Rectangle((x0 + c * cw, y0 + sh - (r + 1) * ch), cw, ch,
                                       fc=RED if dead else '#ececec', ec='white', lw=0.3, zorder=z))
        ax.add_patch(Rectangle((x0, y0), sw, sh, fc='none', ec=INK, lw=0.5, zorder=z + 1))
    return (fx, ybot, fx + sw + (n - 1) * off, ybot + sh + (n - 1) * off)


def heat_cell(x, y, w, h, v, vmin=0, vmax=100, cmap='Oranges'):
    col = plt.get_cmap(cmap)((v - vmin) / (vmax - vmin))
    ax.add_patch(Rectangle((x, y), w, h, fc=col, ec='black', lw=0.5, zorder=5))
    T(x + w / 2, y + h / 2, f'{v:.1f}', size=8, ha='center', va='center',
      color='white' if luminance(col) < 0.6 else INK, fontweight='bold', zorder=7)


def dot_axes(x0, y0, w, h, vals, errs, ticklabels, colors, ytop=126, ylabel='Top-1 (%)'):
    """Point chart with a 0-100 y-axis and dashed light grid: the +/- s.d. error line of each
    point is drawn first (capsize 4, elinewidth 1.2) and a filled circle (MS pt, black edge)
    is drawn on top of it; nothing is connected.  The area above 100 (up to ytop) is reserved
    for annotations."""
    a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    xs = np.arange(len(vals))
    a.errorbar(xs, vals, yerr=errs, fmt='none', ecolor='black', elinewidth=1.2, capsize=4,
               capthick=1.2, zorder=3)
    a.scatter(xs, vals, s=MS ** 2, c=colors, edgecolors='black', linewidths=MEW, zorder=4)
    a.set_xlim(-0.55, len(vals) - 0.45)
    a.set_ylim(0, ytop)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_xticks(xs)
    a.set_xticklabels(ticklabels, fontsize=FS7)
    a.tick_params(axis='y', labelsize=FS7, length=2, pad=1.5, width=0.6)
    a.tick_params(axis='x', length=0, pad=2.5)
    a.set_ylabel(ylabel, fontsize=FS, labelpad=2)
    style_ax(a)
    a.spines['left'].set_bounds(0, 100)
    a.spines['left'].set_linewidth(0.7)
    a.spines['bottom'].set_linewidth(0.7)
    a.set_facecolor('none')
    return a


def marker_radius(ytop, h):
    """Radius of the circle marker (incl. its edge) in data units of a chart of height h
    (design units; points are converted through the export SCALE)."""
    return (MS / 2 + MEW / 2) / 72 / SCALE * ytop / h


def dot_labels(a, vals, errs, ytop, h, below=False, y=None, gap=6.0, bbox=None, two_line=False):
    """'v ± e' label for every point: at a common height y (va='bottom'), or just above /
    below the circle or its s.d. cap, whichever reaches further.  two_line=True stacks the
    label as 'v' over '± e' so neighbouring labels stay apart in a narrow chart."""
    r = marker_radius(ytop, h)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if y is not None:
            yy, va = y, 'bottom'
        elif below:
            yy, va = v - max(e, r) - gap, 'top'
        else:
            yy, va = v + max(e, r) + gap, 'bottom'
        s = f'{v:.1f}\n± {e:.1f}' if two_line else f'{v:.1f} ± {e:.1f}'
        a.text(i, yy, s, ha='center', va=va, fontsize=FS7, color=INK, zorder=6, bbox=bbox,
               linespacing=1.05)


# --------------------------------------------------------------------------- data for the mini-map
wn = wntr.network.WaterNetworkModel(os.path.join(ROOT, 'dataset', 'Exa7.inp'))
assign = json.load(open(os.path.join(ROOT, 'partition_results_leiden', 'partition_summary.json')))['15']['node_assignments']
SF = json.load(open(os.path.join(ROOT, 'knowledge_base', 'sensor_fingerprints.json')))
sensors = SF['sensor_nodes']
assert len(sensors) == 30 and len(set(sensors)) == 30 and wn.num_junctions == 381
assert len(set(assign[n] for n in wn.junction_name_list)) == 15
# one real template fingerprint (30 pressure changes) for the fingerprint glyph
_tpl = next(t for t in SF['fingerprints'] if abs(t['leak_rate_Ls'] - 20) < 1e-6 and abs(t['demand_multiplier'] - 1.0) < 1e-6)
F_J = np.asarray(_tpl['sensor_fingerprint'], float)
rng = np.random.default_rng(3)
Y_OBS = F_J * 0.8 + rng.normal(0, 0.02, F_J.size)      # illustrative query: scaled template + 0.02 m noise

PALETTE15 = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1',
             '#ff9da7', '#9c755f', '#bab0ac', '#1f77b4', '#2ca02c', '#8c564b', '#17becf', '#7f7f7f']


def draw_minimap(x0, y0, w, h):
    axm = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    axm.set_aspect('equal')
    axm.axis('off')
    xy = {n: np.array(wn.get_node(n).coordinates, dtype=float) for n in wn.node_name_list}
    for lname in wn.link_name_list:
        lk = wn.get_link(lname)
        a, b = xy[lk.start_node_name], xy[lk.end_node_name]
        axm.plot([a[0], b[0]], [a[1], b[1]], color='#a3a3a3', lw=0.35, zorder=1, solid_capstyle='round')
    juncs = wn.junction_name_list
    pts = np.array([xy[n] for n in juncs])
    cols = [PALETTE15[assign[n]] for n in juncs]
    axm.scatter(pts[:, 0], pts[:, 1], s=2.6, c=cols, edgecolors='none', zorder=2)
    sx = np.array([xy[s] for s in sensors])
    assert sx.shape == (30, 2)
    # white halo + black ring so neighbouring sensors separate on any partition colour
    axm.scatter(sx[:, 0], sx[:, 1], s=26, facecolors='white', edgecolors='none', zorder=5)
    axm.scatter(sx[:, 0], sx[:, 1], s=11, facecolors=RED, edgecolors='black', linewidths=0.45, zorder=6)
    # reservoir / tank symbols are deliberately not drawn: the figure carries fact-sheet items only
    axm.margins(0.03)
    return axm


"""Layout segment appended to the preserved original vector primitives.
All coordinates use the original six-stage figure's design-unit canvas.
"""
from pathlib import Path
import csv
import re
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.text import Text

from _paths import OUTPUT as PKG
QA = PKG / 'qa'
SD = PKG / 'source_data'
for folder in (QA, SD):
    folder.mkdir(exist_ok=True)

# Follow the reference's serif typography; graphic labels use the same family.
FS = FS7 = 7.0
FST = 8.0
MS = 6.4
MEW = .65
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})

def subaxis(x, y, w, h):
    a = fig.add_axes([x/W, y/H, w/W, h/H])
    a.set_facecolor('none')
    return a

def label(x, y, s, size=7, **kw):
    return T(x, y, s, size=size, va='center', **kw)

def pin(x, y, r=.04, color=CORAL):
    ax.add_patch(Polygon([[x-r*.8,y],[x+r*.8,y],[x,y-r*1.75]],
                         closed=True, fc=color, ec='none', zorder=6))
    ax.add_patch(Circle((x,y),r,fc=color,ec='white',lw=.45,zorder=7))
    ax.add_patch(Circle((x,y),r*.27,fc='white',ec='none',zorder=8))

def gauge(x, y, s=.17, color=CYAN):
    ax.add_patch(Circle((x,y),s,fc='white',ec=color,lw=1.6,zorder=5))
    ax.add_patch(Arc((x,y),s*1.48,s*1.48,theta1=25,theta2=155,color=color,lw=1.4,zorder=6))
    for theta in np.linspace(25,155,5):
        a=np.deg2rad(theta)
        ax.plot([x+s*.59*np.cos(a),x+s*.74*np.cos(a)],
                [y+s*.59*np.sin(a),y+s*.74*np.sin(a)],c=color,lw=.6,zorder=6)
    arrow((x,y),(x+s*.47,y+s*.49),color=CORAL,lw=1.2,ms=5,z=6)
    ax.add_patch(Circle((x,y),s*.10,fc=color,ec='none',zorder=7))
    ax.plot([x,x],[y-s,y-s*1.27],color=color,lw=2.0,zorder=5)

def tiny_network(x,y,w,h,color=TEAL):
    pts=np.array([[.06,.50],[.22,.82],[.40,.59],[.53,.94],[.65,.48],
                  [.93,.72],[.80,.13],[.42,.12],[.15,.20]])* [w,h]+[x,y]
    for i,j in [(0,1),(1,2),(2,3),(2,4),(4,5),(4,6),(6,7),(7,8),(8,0),(2,7)]:
        ax.plot(pts[[i,j],0],pts[[i,j],1],color=color,lw=.7,zorder=4)
    for i,p in enumerate(pts):
        ax.add_patch(Circle(p,.019,fc=CORAL if i in (1,5) else color,ec='white',lw=.35,zorder=6))

def urban_leak(x,y,w,h):
    """Schematic city / pipe context only, no quantitative scale."""
    rbox(x,y,w,h,fc='#EFF9FC',ec='none',lw=0,r=.045,z=2)
    base=y+h*.42
    for i,(xx,ww,hh,col) in enumerate([(0.04,.17,.36,'#77CBCD'),(.26,.19,.55,'#5E9FCC'),
                                      (.50,.17,.43,'#9E8EC6'),(.72,.23,.32,'#E7B761')]):
        bx=x+xx*w; by=base; bw=ww*w; bh=hh*h
        ax.add_patch(Rectangle((bx,by),bw,bh,fc=col,ec='none',zorder=4))
        for row in range(3):
            for coln in range(2):
                ax.add_patch(Rectangle((bx+bw*(.17+.40*coln),by+bh*(.16+.24*row)),
                                      bw*.18,bh*.12,fc='white',ec='none',zorder=5))
        ax.plot([bx+bw*.5,bx+bw*.5],[base, y+h*.21],color=BLUE,lw=1.3,zorder=4)
    ax.plot([x+.04*w,x+.97*w],[y+h*.21]*2,color='#8BD7DB',lw=6,zorder=4,solid_capstyle='round')
    ax.plot([x+.04*w,x+.97*w],[y+h*.21]*2,color=BLUE,lw=1.2,zorder=5)
    bx=x+w*.64; by=y+h*.21
    ax.plot([bx-.03,bx,bx-.016,bx+.033],[by+.035,by+.007,by-.015,by-.033],c=CORAL,lw=1.4,zorder=7)
    for dx in (-.065,0,.066):
        droplet(bx+dx,by-.07-abs(dx)*.2,.014,CYAN)

def small_chart_frame(a):
    a.spines[['top','right']].set_visible(False)
    for sp in a.spines.values(): sp.set_linewidth(.6)
    a.tick_params(length=2,width=.6,pad=2)
    a.grid(axis='y',color='#DEE5EC',lw=.4,ls=(0,(3,3)))
    a.set_axisbelow(True)

# Full real corpus, no aesthetic sampling. A shared norm is used for all matrix rows.
records=sorted(SF['fingerprints'],key=lambda r:(r['leak_partition'],r['leak_node'],r['leak_rate_Ls'],r['demand_multiplier']))
F=np.asarray([r['sensor_fingerprint'] for r in records],float)
norm2=np.einsum('ij,ij->i',F,F)
norm=np.sqrt(norm2)
nonzero=norm>1e-12
matrix=np.divide(np.abs(F),norm[:,None],out=np.zeros_like(F),where=nonzero[:,None])
cm=LinearSegmentedColormap.from_list('fingerprint_teal',['#FAF0BC','#8DD8DB','#1D719F','#17486F'])
cn=PowerNorm(.5,vmin=0,vmax=float(matrix.max()))
coeff=np.divide(F@Y_OBS,norm2,out=np.zeros(len(F)),where=nonzero)
residual=np.linalg.norm(Y_OBS[None,:]-coeff[:,None]*F,axis=1)
residual[~nonzero]=np.inf
rank=np.argsort(residual,kind='stable')
winner=records[int(rank[0])]['leak_partition']
assert F.shape==(540,30) and np.isfinite(matrix).all()

def highlighted_network(x,y,w,h,part):
    a=subaxis(x,y,w,h)
    xy={n:np.asarray(v.coordinates) for n,v in wn.nodes()}
    for _,link in wn.links():
        start,end=link.start_node_name,link.end_node_name
        selected=assign.get(start)==part and assign.get(end)==part
        a.plot(*np.array([xy[start],xy[end]]).T,c=TEAL if selected else '#C5D1D9',
               lw=.95 if selected else .40,zorder=3 if selected else 1)
    nodes=wn.junction_name_list
    p=np.array([xy[n] for n in nodes])
    a.scatter(p[:,0],p[:,1],s=[7 if assign[n]==part else 1.3 for n in nodes],
              c=[TEAL if assign[n]==part else '#D0D8DE' for n in nodes],lw=0,zorder=4)
    a.set_aspect('equal');a.margins(.025);a.axis('off')
    return a

# Keep reference geometry: title / challenge / six-stage approach / three evidence cards.
rbox(.15,8.26,6.90,.32,fc=INK,ec=INK,r=.08)
label(3.60,8.42,'TS-GraphRAG  |  simulation-grounded leak localization',9.8,
      color='white',ha='center',fontweight='bold')

band(.15,7.08,6.90,1.10,GREEN,'Challenge',hh=.28,letter='a')
ax.plot([3.15,3.15],[7.18,7.81],c=SEP,lw=.7)
label(.30,7.77,'Water losses',8.1,fontweight='bold')
urban_leak(.30,7.18,1.31,.46)
label(1.77,7.73,'NRW',6.7,color=BLUE,fontweight='bold')
metric(1.73,7.49,'346 Mm³ d⁻¹ [1]',BLUE,w=1.26,h=.19,size=6.8)
label(1.77,7.41,'Physical losses',6.7,color=CORAL,fontweight='bold')
metric(1.73,7.17,'>32 B m³ yr⁻¹ [2]',CORAL,w=1.26,h=.19,size=6.7)

# Three filled visual motifs express sensing, nuisance factors, and output resolution.
for x,w,fc in [(3.30,1.02,'#F0F7FE'),(4.40,1.22,'#FFF7E8'),(5.70,1.20,'#EFF9F6')]:
    rbox(x,7.17,w,.65,fc=fc,ec='none',lw=0,r=.045,z=2)
gauge(3.56,7.52,.14,BLUE)
tiny_network(3.78,7.35,.44,.31,BLUE)
label(3.80,7.72,'Sparse sensing',6.7,ha='center',fontweight='bold')
label(3.81,7.24,'30 / 381',7.4,ha='center',fontweight='bold',color=BLUE)
label(5.01,7.72,'Discharge + demand',6.7,ha='center',fontweight='bold')
for i,s in enumerate((.025,.039,.055)):
    droplet(4.62+i*.18,7.42,s,CYAN)
wave(5.17,7.49,.29,.12,AMBER)
wave(5.17,7.34,.29,.07,VIOLET)
label(5.02,7.23,'vary',6.7,ha='center',color=GREY)
label(6.30,7.72,'Partition output',6.7,ha='center',fontweight='bold')
tiny_network(5.83,7.29,.62,.27,TEAL)
zone_glyph(6.63,7.45,.12,TEAL)

block_arrow(3.60,7.08,6.90)
band(.15,3.78,6.90,3.12,BLUE,'Approach  |  TS-GraphRAG',letter='b')

# 1. Existing model with all sensors and partitions, filling the tall left column.
stage(.30,3.90,1.80,2.60,1,'Network model',fc='#F5FAFE')
label(.42,6.23,'EXA7 · EPANET / WNTR',6.7,color=GREY)
draw_minimap(.42,4.28,1.55,1.84)
for x,c,s in [(.47,CYAN,'381 J'),(1.01,RED,'30 S'),(1.53,VIOLET,'15 Z')]:
    metric(x,4.03,s,c,w=.44,h=.18,size=6.5)

# 2. Actual factor counts are visual arrays, not paragraphs.
stage(2.30,5.51,2.25,.99,2,'Template library',fc='#FFFCF3')
for i in range(30):
    pin(2.47+(i%6)*.071,5.95+(i//6)*.046,.011,CYAN)
for i,rate in enumerate((2,5,10,20,35,50)):
    ax.add_patch(Rectangle((3.18+i*.056,5.93),.033,.05+.21*rate/50,
                           fc=AMBER,ec='none',zorder=6))
for i,(amp,col) in enumerate(((.05,TEAL),(.08,BLUE),(.11,VIOLET))):
    wave(3.93,5.94+i*.069,.37,amp*.45,col)
label(2.65,5.81,'30 locations',6.8,ha='center')
label(3.35,5.81,'6 rates',6.8,ha='center')
label(4.07,5.81,'3 states',6.8,ha='center')
label(3.01,6.04,'×',11,color=GREY,ha='center')
label(3.69,6.04,'×',11,color=GREY,ha='center')
ax.plot([2.48,4.38],[5.73,5.73],c='#DEC97A',lw=.65)
label(3.43,5.62,'540 templates',8.1,color=BLUE,fontweight='bold',ha='center')

# 3. The complete matrix replaces the largely vacant single-strip card.
stage(4.75,5.51,2.25,.99,3,'Fingerprint matrix',fc='#F1F9F9')
am=subaxis(4.88,5.76,1.93,.47)
am.imshow(matrix,aspect='auto',cmap=cm,norm=cn,interpolation='nearest',rasterized=True)
am.set(xticks=[],yticks=[])
for sp in am.spines.values():sp.set_visible(False)
label(4.89,5.62,'540 × 30',7.5,color=BLUE,fontweight='bold')
for i in range(30):
    ax.add_patch(Rectangle((6.00+i*.019,5.60),.020,.043,fc=cm(i/29),ec='none',zorder=5))
label(5.88,5.62,'0',6.5,color=GREY)
label(6.62,5.62,'max',6.5,color=GREY)

# 4. Same illustrative query as the source figure, explicit and data-linked.
stage(2.30,4.42,1.00,.93,4,'Query',fc='#F3FAFE')
qa=subaxis(2.41,4.69,.76,.39)
qa.plot(range(30),Y_OBS,color=BLUE,lw=1.0)
qa.fill_between(range(30),Y_OBS,0,color=CYAN,alpha=.22,lw=0)
qa.set_xlim(0,29);qa.axis('off')
label(2.81,4.55,r'Illustrative $\mathbf{y}$',6.8,ha='center',color=BLUE)

# 5. Geometry + actual top-three ranks make fit/rank visible, with short labels.
stage(3.45,4.42,2.00,.93,5,'Residual rank',fc='#F5F0FC',color=VIOLET,lw=1.4)
O=np.array([3.60,4.62]);Y=np.array([3.88,5.10]);V=np.array([4.27,4.87])
u=V-O;v=Y-O;P=O+u*np.dot(u,v)/np.dot(u,u)
arrow(O,Y,color=INK,lw=.9,ms=6)
arrow(O,P,color=BLUE,lw=1.2,ms=6)
ax.plot([Y[0],P[0]],[Y[1],P[1]],c=CORAL,lw=1.1,ls=(0,(2,1)),zorder=5)
# The scaled-template arrow must end at the orthogonal projection, not beyond it.
along=(O-P)/np.linalg.norm(O-P);normal=(Y-P)/np.linalg.norm(Y-P)
corner=np.array([P+along*.035,P+(along+normal)*.035,P+normal*.035])
ax.plot(corner[:,0],corner[:,1],c=GREY,lw=.55,zorder=6)
label(3.70,5.03,r'$\mathbf{y}$',7.2)
label(4.08,4.69,r'$a_j\mathbf{f}_j$',6.7,color=BLUE,ha='center')
label(4.01,5.04,r'$r_j$',7,color=CORAL)
rankvals=residual[rank[:3]]
for i,value in enumerate(rankvals):
    yy=5.06-i*.14
    ax.add_patch(Rectangle((4.66,yy-.035),.48*value/rankvals.max(),.066,
                           fc=TEAL if i==0 else '#C6B5DE',ec='none',zorder=5))
    label(4.57,yy,str(i+1),6.5,ha='center',color=GREY)
label(4.90,4.64,'min residual',6.6,ha='center',color=VIOLET)

# 6. All network geometry retained; highlight the computed winning partition.
stage(5.60,4.42,1.40,.93,6,'Zone output',fc='#F1FAF5',color=TEAL)
highlighted_network(5.73,4.53,.71,.63,winner)
zone_glyph(6.64,4.89,.15,TEAL)
label(6.63,4.59,f'P{winner}',8,color=TEAL,fontweight='bold',ha='center')

# Compact auxiliary branch. Explicit comparator keeps the result interpretable.
rbox(2.30,3.90,4.70,.36,fc='#F2F3F4',ec='#ABB0B5',lw=.7,r=.045,ls=(0,(3,2)))
label(2.41,4.15,'Optional · single realization',6.5,fontweight='bold',color=GREY)
label(4.42,4.15,'Graph vs 1-NN',6.5,color=GREY)
label(5.61,4.15,'LM re-ranking (est.)',6.5,color=GREY)
# The approved revision updates this H8 annotation from the strict five-seed means.
import pandas as pd
_strict = pd.read_csv(Path(ROOT)/'plot_tables/holdout_summary.csv')
_h8 = _strict[_strict.protocol.eq('H8')].set_index('method').top1_pct_mean
label(2.41,3.995,f'Fusion {_h8["fusion_strict"]:.1f}% < residual {_h8["projection"]:.1f}%',6.5,color=GREY)
label(4.42,3.995,'−0.2 to −1.9 pp',6.5,color=GREY)
label(5.61,3.995,'−0.9 to −37.4 pp',6.5,color=GREY)
for p,q,c in [((2.10,6.04),(2.30,6.04),BLUE),((4.55,6.04),(4.75,6.04),BLUE),
              ((5.17,5.51),(5.17,5.35),VIOLET),((3.30,4.88),(3.45,4.88),BLUE),
              ((5.45,4.88),(5.60,4.88),VIOLET)]:arrow(p,q,color=c,lw=1.2,ms=8)
arrow((4.48,4.42),(4.48,4.26),color=GREY,lw=.8,ls=(0,(3,2)),ms=6)

# Frozen quantitative inputs; no re-estimation or selection of favourable observations.
def read_tsv(name):
    with (Path(ROOT)/'results/E4_mean_std_20260909'/name).open(encoding='utf-8-sig') as f:
        return {row['Method']:row for row in csv.DictReader(f,delimiter='\t')}
all540=read_tsv('Table_S3b_all540.tsv')
matched=read_tsv('Table_S3a_matched90.tsv')
protocols=['H0','H1','H2','H8']
vals1,errs1=map(list,zip(*(parse_pm(all540['Projection residual'][p]) for p in protocols)))
assert vals1==[87.2,86.8,76.0,59.8] and errs1==[.8,.6,1.1,1.2]
with (Path(ROOT)/'results/E5_design_curve_20260909/e5_h0_summary.csv').open(encoding='utf-8-sig') as f:
    allocation=list(csv.DictReader(f))
designs=['A','C60','C90','B']
allocrows=[next(r for r in allocation if r['group']=='G2' and r['protocol']=='H0' and r['metric']=='top1' and r['design']==d) for d in designs]
# Preserve reference display precision; full values remain in the source-data table.
vals3=[round(float(r['mean']),1) for r in allocrows]
errs3=[round(float(r['sample_sd']),1) for r in allocrows]
assert vals3==[69.9,79.0,82.0,79.2] and errs3==[8.6,8.4,7.4,6.9]
with (Path(ROOT)/'results/E5_design_curve_20260909/e5_g2_top1_pairwise.csv').open(encoding='utf-8-sig') as f:
    pair=next(r for r in csv.DictReader(f) if r['first']=='C90' and r['second']=='A')
assert (int(pair['positive']),int(pair['negative']),int(pair['ties']))==(15,0,0)

EY0,EH=.62,2.98
CX0,CW=.15,2.60
DX0,DW=2.85,1.86
EX0,EW=4.81,2.24
for xc in (CX0+CW/2,DX0+DW/2,EX0+EW/2):block_arrow(xc,3.78,3.60)

# c: extend the chart vertically through the former blank band.
band(CX0,EY0,CW,EH,CORAL,'Holdout protocols',hh=.28,size=9.3,lw=1.3,letter='c',toff=.08)
label(.28,3.17,'Red = deleted',6.7,color=CORAL,fontweight='bold')
label(2.60,3.17,'H9: audit',6.7,ha='right',color=GREY)
DEL={'H0':lambda s,r,c:False,'H1':lambda s,r,c:s==0 and r==0 and c==2,
     'H2':lambda s,r,c:s==0,'H3':lambda s,r,c:r==0,'H4':lambda s,r,c:c==2,
     'H8':lambda s,r,c:s==0 or r==0 or c==2,'H9':lambda s,r,c:s<=1}
for i,key in enumerate(['H0','H1','H2','H3','H4','H8','H9']):
    cx=.39+i*.354
    sheet_stack(cx,2.79,DEL[key],cw=.07,ch=.040,off=.022)
    chip(cx,2.69,key,w=.25,h=.14)
axc=dot_axes(.58,1.04,2.00,1.48,vals1,errs1,protocols,
             [PROT_COL[p] for p in protocols],ytop=123)
axc.bar(range(4),vals1,width=.50,color=[PROT_COL[p] for p in protocols],alpha=.13,zorder=1)
dot_labels(axc,vals1,errs1,123,1.48,below=True,gap=5.0,two_line=True)
small_chart_frame(axc)
axc.plot([-.2,-.2,1.2,1.2],[102,106,106,102],c=GREY,lw=.6)
axc.text(.5,110,'H1 ≈ H0',ha='center',fontsize=7)
axc.text(2.72,110,'H2 → H8',ha='center',fontsize=7,color=RED)
label(1.45,.79,'540 queries · 5 repeats',6.8,ha='center',color=GREY)

# d: a taller complete matrix, plus a compact deletion-factor diagram.
band(DX0,EY0,DW,EH,AMBER,'Baselines',hh=.28,size=9.3,lw=1.3,letter='d',toff=.08)
label(3.00,3.18,'Top-1 (%) · 90 queries',6.8,color=GREY)
methods=['GraphSAGE+','Random Forest (100 trees)','Projection residual']
heat=np.array([[parse_pm(matched[m][p])[0] for p in ['H0','H2','H8']] for m in methods])
hx=3.73; cw=.29; ch=.46; gap=.025
for j,p in enumerate(['H0','H2','H8']):label(hx+j*(cw+gap)+cw/2,2.99,p,7.3,ha='center',fontweight='bold')
for i,name in enumerate(['GraphSAGE+','Random forest','Projection']):
    y0=2.82-ch-i*(ch+.055); yc=y0+ch/2
    label(3.00,yc+.075,name,6.6,fontweight='bold' if i==2 else 'normal')
    # Method structure glyphs are conceptual icons, not extra evaluation series.
    if i==0:tiny_network(3.06,yc-.17,.43,.14,VIOLET)
    elif i==1:
        for xx in (3.10,3.28,3.46):
            ax.plot([xx,xx],[yc-.19,yc-.10],c=TEAL,lw=.7)
            ax.add_patch(Polygon([[xx-.06,yc-.13],[xx,yc-.035],[xx+.06,yc-.13]],fc=TEAL,ec='none',zorder=5))
    else:
        arrow((3.06,yc-.16),(3.22,yc-.045),color=VIOLET,lw=.8,ms=4)
        arrow((3.06,yc-.16),(3.51,yc-.10),color=BLUE,lw=.8,ms=4)
    for j,value in enumerate(heat[i]):heat_cell(hx+j*(cw+gap),y0,cw,ch,value)
ky=1.13; keyw=1.49
for i in range(100):ax.add_patch(Rectangle((3.06+i*keyw/100,ky),keyw/100,.07,fc=plt.get_cmap('Oranges')(i/99),ec='none',zorder=4))
label(3.06,1.04,'0',6.5,color=GREY)
label(4.55,1.04,'100',6.5,ha='right',color=GREY)
label(3.00,.80,'H8',7,color=RED,fontweight='bold')
for xx,col,ss in [(3.44,CYAN,'L'),(3.82,AMBER,'Q'),(4.20,VIOLET,'D')]:
    ax.add_patch(Circle((xx,.81),.095,fc=col,ec='none',zorder=5))
    label(xx,.81,ss,7,color=INK if col==AMBER else 'white',ha='center',fontweight='bold')
label(3.63,.81,r'$\cup$',8,ha='center',color=GREY);label(4.01,.81,r'$\cup$',8,ha='center',color=GREY)

# e: use the chart and allocation glyphs to occupy the card's full height.
band(EX0,EY0,EW,EH,TEAL,'Corpus allocation',hh=.28,size=9.3,lw=1.3,letter='e',toff=.08)
label(4.95,3.18,'540 templates · unseen locations',6.8,color=GREY)
axe=dot_axes(5.24,1.82,1.65,1.24,vals3,errs3,['30×6','60×3','90×2','180×1'],
              [BLUE,AMBER,TEAL,VIOLET],ytop=123)
axe.bar(range(4),vals3,width=.45,color=[BLUE,AMBER,TEAL,VIOLET],alpha=.12,zorder=1)
dot_labels(axe,vals3,errs3,123,1.24,gap=4,two_line=True)
small_chart_frame(axe)
label(6.06,1.53,'Locations × rates · 15 draws',6.7,ha='center',color=GREY)
# Two complementary corpus shapes, with the correct selection scope.
rbox(4.94,1.04,.91,.35,fc='#EDF5FC',ec='#96B7D7',lw=.6,r=.04)
rbox(5.97,1.04,.94,.35,fc='#EAF7F3',ec='#8BC5BA',lw=.6,r=.04)
for i in range(6):
    for j in range(3):ax.add_patch(Rectangle((5.01+j*.034,1.10+i*.032),.027,.025,fc=BLUE,ec='none',zorder=4))
for i in range(2):
    for j in range(9):ax.add_patch(Rectangle((6.02+j*.022,1.14+i*.065),.017,.043,fc=TEAL,ec='none',zorder=4))
label(5.20,1.28,'Represented',6.5,color=BLUE)
label(5.20,1.13,'A: 30 × 6',6.8,fontweight='bold',color=BLUE)
label(6.29,1.28,'Unseen',6.5,color=TEAL)
label(6.29,1.13,'C90: 90 × 2',6.7,fontweight='bold',color=TEAL)
arrow((5.86,1.21),(5.96,1.21),color=CORAL,lw=.9,ms=5)
for i in range(15):
    ax.add_patch(Circle((5.01+i*.052,.85),.017,fc=TEAL,ec='none',zorder=5))
label(5.88,.85,'C90 > A: 15/15',6.8,color=TEAL,fontweight='bold')

rbox(.15,.10,6.90,.42,fc=INK,ec=INK,r=.08)
label(3.60,.38,'Evaluation protocol changes the corpus-design decision',8.4,
      ha='center',color='white',fontweight='bold')
label(3.60,.20,'Simulation evidence · field validation remains open',7,
      ha='center',color='#E4E4E4',style='italic')

# Data and rendering audit written from the actual artist tree, not static guesses.
def audit_layout():
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    items=[]
    for a in fig.axes:
        arts=list(a.texts)
        if a.axison:
            arts+=a.get_xticklabels()+a.get_yticklabels()+[a.xaxis.label,a.yaxis.label]
        for t in arts:
            if not t.get_visible() or not t.get_text().strip():continue
            bb=t.get_window_extent(renderer)
            items.append({'text':t.get_text(),'font_pt':t.get_fontsize(),
                          'bbox':[float(x) for x in bb.extents]})
    fw,fh=fig.canvas.get_width_height()
    overflow=[i['text'] for i in items if i['bbox'][0]<0 or i['bbox'][1]<0 or i['bbox'][2]>fw+1 or i['bbox'][3]>fh+1]
    overlaps=[]
    for i,a in enumerate(items):
        for b in items[i+1:]:
            aa,bb=a['bbox'],b['bbox']
            if min(aa[2],bb[2])-max(aa[0],bb[0])>1.5 and min(aa[3],bb[3])-max(aa[1],bb[1])>1.5:
                overlaps.append([a['text'],b['text']])
    report={'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,
            'word_count':sum(len(re.findall(r'\b\w+\b',i['text'])) for i in items),
            'minimum_font_pt':min(i['font_pt'] for i in items),'overflow':overflow,
            'text_overlaps':overlaps,'text':items,
            'chart_heights_before_after':{'c':[.8,1.48],'d_heatmap':[1.00,1.49],'e':[1,1.24]}}
    (QA/'layout_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='text'},ensure_ascii=False))

mapping={'query_origin':_tpl['scenario_id'],'query_illustrative':True,'query_random_seed':3,
         'query_scale':.8,'query_noise_sd_m':.02,'rank_total':540,'rank_eligible':int(nonzero.sum()),'zero_norm_templates':int((~nonzero).sum()),
         'top3':[{'scenario_id':records[int(i)]['scenario_id'],'residual':float(residual[i])} for i in rank[:3]],
         'predicted_partition':winner,'matrix_shape':list(F.shape),
         'holdout':{'protocols':protocols,'means':vals1,'sd':errs1},
         'baseline':{'methods':methods,'values':heat.tolist()},
         'allocation':{'designs':designs,'mean':vals3,'sd':errs3},'C90_vs_A_wins':15}
(QA/'data_audit.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
np.savetxt(SD/'matrix_raw.csv',F,delimiter=',',header=','.join(map(str,sensors)),comments='')
with (SD/'matrix_rows.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['scenario_id','partition','leak_node','rate_Ls','demand'])
    for r in records:writer.writerow([r['scenario_id'],r['leak_partition'],r['leak_node'],r['leak_rate_Ls'],r['demand_multiplier']])
np.savetxt(SD/'illustrative_query.csv',np.column_stack([F_J,Y_OBS]),delimiter=',',header='template,illustrative_query',comments='')
audit_layout()
save(fig,'Fig1_conceptual')
