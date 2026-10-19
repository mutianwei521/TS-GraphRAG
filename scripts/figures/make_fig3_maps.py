# -*- coding: utf-8 -*-
"""Fig. 3 | Partition maps and sensor configurations (Voronoi-zone style of the previous figure set).

a, EXA7 (K = 15, 30 sensors)   b, City D (K = 15, 28 sensors)   c, Haining (K = 15, 62 sensors)
Zones are Voronoi cells of the junctions merged by partition label: a visual aid, not a geometric partition.
"""
import os, json, pickle
import numpy as np
from collections import defaultdict
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, box, GeometryCollection
from shapely.ops import unary_union
import wntr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fig_style import *
from nature_style import MM, TEAL, RED, GREY

import argparse
from pathlib import Path
from _paths import INPUT
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--municipal-data-dir', type=Path, required=True,
                    help='Directory containing dagang.inp, haining.inp, dagang_partitions.pkl and haining_partitions.pkl')
municipal = parser.parse_args().municipal_data_dir.resolve()
for filename in ['dagang.inp', 'haining.inp', 'dagang_partitions.pkl', 'haining_partitions.pkl']:
    if not (municipal/filename).is_file():
        raise FileNotFoundError(f'Provide separately obtained municipal input: {municipal/filename}')
sensor_lists = json.loads((INPUT/'fig3_sensor_assignments.json').read_text(encoding='utf-8'))

ZONE_FILLS = ['#D6EAF8', '#D5F5E3', '#FCF3CF', '#E8DAEF', '#D1F2EB', '#FDEBD0', '#D4E6F1', '#D0ECE7', '#FEF9E7',
              '#D7BDE2', '#A9DFBF', '#AED6F1', '#F9E79F', '#D2B4DE', '#A3E4D7']
ZONE_BORDERS = ['#2980B9', '#27AE60', '#D4AC0D', '#7D3C98', '#1ABC9C', '#CA6F1E', '#2471A3', '#17A589', '#B7950B',
                '#6C3483', '#229954', '#2E86C1', '#D68910', '#884EA0', '#138D75']
FILL_ALPHA = 0.57


def voronoi_zones(pos, partition, margin_frac=0.06):
    nids = [n for n in partition if n in pos]
    pts = np.array([pos[n] for n in nids]); pid = [partition[n] for n in nids]
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max(); ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    dx, dy = (xmax - xmin) * margin_frac, (ymax - ymin) * margin_frac
    bbox = box(xmin - dx, ymin - dy, xmax + dx, ymax + dy)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2; span = max(xmax - xmin, ymax - ymin) * 4
    mirror = np.array([[cx + i * span, cy + j * span] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
    vor = Voronoi(np.vstack([pts, mirror]))
    cells = {}
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        try:
            poly = Polygon([vor.vertices[v] for v in reg])
            if not poly.is_valid:
                poly = poly.buffer(0)
            c = poly.intersection(bbox)
            if not c.is_empty:
                cells[i] = c
        except Exception:
            continue
    part_cells = defaultdict(list)
    for i, poly in cells.items():
        part_cells[pid[i]].append(poly)
    zones, orphans = {}, []
    for p, polys in part_cells.items():
        m = unary_union(polys)
        if isinstance(m, MultiPolygon):
            comps = sorted(m.geoms, key=lambda g: g.area, reverse=True)
            zones[p] = comps[0]; orphans += comps[1:]
        else:
            zones[p] = m
    for frag in orphans:
        best = min(zones, key=lambda p: zones[p].distance(frag.centroid))
        zones[best] = unary_union([zones[best], frag])
    for _ in range(5):
        gap = bbox.difference(unary_union(list(zones.values())))
        if gap.is_empty or gap.area < 1e-9:
            break
        pieces = [g for g in getattr(gap, 'geoms', [gap]) if isinstance(g, Polygon) and g.area > 0]
        if not pieces:
            break
        for piece in pieces:
            best = min(zones, key=lambda p: zones[p].distance(piece.centroid))
            zones[best] = unary_union([zones[best], piece])
    return zones


def plot_poly(ax, geom, fc, ec, lw):
    if geom.is_empty:
        return
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            plot_poly(ax, g, fc, ec, lw)
        return
    r, g, b = int(fc[1:3], 16) / 255, int(fc[3:5], 16) / 255, int(fc[5:7], 16) / 255
    xs, ys = geom.exterior.xy
    ax.add_patch(plt.Polygon(list(zip(xs, ys)), closed=True, facecolor=(r, g, b, FILL_ALPHA), edgecolor=ec, linewidth=lw, zorder=1))
    for hole in geom.interiors:
        xs, ys = hole.xy
        ax.add_patch(plt.Polygon(list(zip(xs, ys)), closed=True, facecolor='white', edgecolor=ec, linewidth=lw * 0.5, zorder=1.1))


def load_network(inp, drop_dummies=False):
    wn = wntr.network.WaterNetworkModel(inp)
    pos = {n: tuple(map(float, wn.get_node(n).coordinates)) for n in wn.node_name_list}
    juncs = [j for j in wn.junction_name_list if not (drop_dummies and (j.endswith('-A') or j.endswith('-B')))]
    links = [(wn.get_link(l).start_node_name, wn.get_link(l).end_node_name) for l in wn.link_name_list]
    sources = list(wn.reservoir_name_list) + list(wn.tank_name_list)
    return wn, pos, juncs, links, sources


def draw(ax, pos, juncs, links, sources, partition, sensors, title, cfg):
    part = {n: partition[n] for n in juncs if n in partition}
    zones = voronoi_zones(pos, part)
    for p in sorted(zones):
        plot_poly(ax, zones[p], ZONE_FILLS[p % 15], ZONE_BORDERS[p % 15], cfg['zone_lw'])
    for a, b in links:
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], '-', color='#555555', lw=cfg['pipe_lw'], alpha=0.48, zorder=2)
    ss = set(sensors)
    jx = [pos[n][0] for n in juncs if n not in ss]; jy = [pos[n][1] for n in juncs if n not in ss]
    ax.plot(jx, jy, 'o', color='#555555', ms=cfg['node_ms'], mec='none', alpha=0.6, zorder=3)
    sx = [pos[s][0] for s in sensors if s in pos]; sy = [pos[s][1] for s in sensors if s in pos]
    ax.plot(sx, sy, 'o', color='#E64B35', ms=cfg['sensor_ms'], mec='black', mew=0.5, zorder=6)
    # reservoirs / tanks: open black squares drawn ABOVE the sensors (zorder 6.5, below the zone labels). On EXA7,
    # reservoir 407 lies 8 coordinate units from sensor junction 34 and was hidden under its circle when drawn below it.
    for s in sources:
        ax.plot(pos[s][0], pos[s][1], 's', mfc='none', mec='black', mew=1.2, ms=cfg['res_ms'] + 3, zorder=6.5)
    for p, geom in zones.items():
        c = geom.representative_point()
        ax.text(c.x, c.y, f'P{p}', ha='center', va='center', fontsize=cfg['label_fs'], fontweight='bold', color='black', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.6))
    xs = [pos[n][0] for n in juncs]; ys = [pos[n][1] for n in juncs]
    mx, my = (max(xs) - min(xs)) * 0.06, (max(ys) - min(ys)) * 0.06
    ax.set_xlim(min(xs) - mx, max(xs) + mx); ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect('equal'); ax.set_anchor('N'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#777777'); sp.set_visible(True)
    ax.set_title(title, fontsize=8, fontweight='bold', loc='left', pad=6)


def main():
    # EXA7
    wn7, pos7, j7, l7, s7 = load_network(os.path.join(ROOT, 'dataset', 'Exa7.inp'))
    part7 = {k: int(v) for k, v in json.load(open(os.path.join(ROOT, 'partition_results_leiden', 'partition_summary.json')))['15']['node_assignments'].items()}
    sens7 = [str(s) for s in json.load(open(os.path.join(ROOT, 'knowledge_base', 'sensor_fingerprints.json')))['sensor_nodes']]
    # City D
    wnD, posD, jD, lD, sD = load_network(str(municipal/'dagang.inp'), drop_dummies=True)
    partD = {str(k): int(v) for k, v in pickle.load(open(municipal/'dagang_partitions.pkl', 'rb'))[15]['node_to_community'].items()}
    sensD = [str(s) for s in sensor_lists['City D']]
    # Haining
    wnH, posH, jH, lH, sH = load_network(str(municipal/'haining.inp'))
    partH = {str(k): int(v) for k, v in pickle.load(open(municipal/'haining_partitions.pkl', 'rb'))[15]['node_to_community'].items()}
    sensH = [str(s) for s in sensor_lists['Haining']]

    fig = plt.figure(figsize=(183*MM,155*MM))
    # Larger maps follow the actual geometry: tall EXA7 left, City D upper
    # right, Haining lower right, and City D sensor detail below its map.
    axes=[fig.add_axes([.033,.13,.34,.76]),
          fig.add_axes([.42,.55,.56,.34]),
          fig.add_axes([.62,.115,.36,.395])]
    cfg7 = dict(sensor_ms=3.4, node_ms=.75, res_ms=.8, label_fs=6.5, zone_lw=.7, pipe_lw=.35)
    cfgD = dict(sensor_ms=3.1, node_ms=.75, res_ms=.8, label_fs=6.0, zone_lw=.7, pipe_lw=.35)
    cfgH = dict(sensor_ms=2.6, node_ms=.55, res_ms=.8, label_fs=6.0, zone_lw=.7, pipe_lw=.3)
    draw(axes[0], pos7, j7, l7, s7, part7, sens7, 'EXA7', cfg7)
    draw(axes[1], posD, jD, lD, sD, partD, sensD, 'City D', cfgD)
    draw(axes[2], posH, jH, lH, sH, partH, sensH, 'Haining', cfgH)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.045, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
    legend = [Patch(facecolor='#D4F1F9', edgecolor='#2980B9', lw=.6, label='Partition'),
              Line2D([0], [0], color='#555555', lw=1.2, label='Pipe'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=2, label='Junction'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#E64B35', markersize=3, markeredgecolor='black', markeredgewidth=0.4, label='Sensor'),
              Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markersize=3, markeredgecolor='black', markeredgewidth=.6, label='Reservoir / tank')]
    fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=7, frameon=True, edgecolor='#BBBBBB', bbox_to_anchor=(0.5, .025))
    # A second view makes the dense sensor neighbourhood legible without
    # distorting the network coordinates or removing nodes from the full map.
    core=[n for n in jD if partD.get(n) in (4,6,10,11)]
    pxy=np.array([posD[n] for n in core])
    lo=pxy.min(axis=0); hi=pxy.max(axis=0); pad=(hi-lo)*.08
    lo-=pad; hi+=pad
    inset=fig.add_axes([.415,.175,.175,.25])
    for u,v in lD:
        if u in posD and v in posD:
            inset.plot([posD[u][0],posD[v][0]],[posD[u][1],posD[v][1]],color='#777777',lw=.55)
    pts=np.array([posD[n] for n in jD]); inset.scatter(pts[:,0],pts[:,1],s=2.5,color='#555555')
    pts=np.array([posD[n] for n in sensD]); inset.scatter(pts[:,0],pts[:,1],s=14,fc=RED,ec='black',lw=.45,zorder=3)
    inset.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xticks=[],yticks=[])
    inset.set_aspect('equal'); inset.set_title('Sensor detail',fontsize=7,pad=4)
    for sp in inset.spines.values(): sp.set_visible(True); sp.set_linewidth(.8); sp.set_color('#777777')
    from matplotlib.patches import Rectangle, ConnectionPatch
    axes[1].add_patch(Rectangle(lo,*(hi-lo),fill=False,ec=TEAL,lw=.65,zorder=8))
    fig.add_artist(ConnectionPatch(xyA=(.23,0),coordsA=axes[1].transAxes,xyB=(.5,1),coordsB=inset.transAxes,color='#777777',lw=.75,ls='--'))
    from nature_style import QA
    (QA/'Fig3_data_checks.json').write_text(json.dumps(dict(
        networks=['EXA7','City D','Haining'],junctions=[len(j7),len(jD),len(jH)],
        sensors=[len(sens7),len(sensD),len(sensH)],full_network_views=True,
        coordinate_aspect='equal',inset='City D duplicate view of selected sensor neighbourhood',
        data_exclusions='none beyond the existing City D dummy-junction definition',
    ),indent=2),encoding='utf-8')
    save(fig, 'Fig3_partition_maps')
    print('sensors placed:', sum(s in pos7 for s in sens7), sum(s in posD for s in sensD), sum(s in posH for s in sensH))


if __name__ == '__main__':
    main()
