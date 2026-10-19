"""Draw the README overview from the released EXA7 data and inference engine.

Run from any working directory: python scripts/readme/build_teaser.py
Only the geometry in panel 3 is schematic. Display normalization never changes
the data passed to the inference engine. Output is an original PNG/PDF drawing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.phase3_query.numerical_retrieval import NumericalRetriever, read_observations

NAVY = "#172B4D"
MUTED = "#65758D"
STROKE = "#E1E7F0"
ACCENTS = ["#3274D9", "#20A58F", "#D99C32", "#8766C5"]
PARTITION_COLORS = [
    "#3274D9", "#67A6D9", "#287F9B", "#27A58E", "#6CBD99",
    "#9AC8AA", "#C8AF55", "#DB913F", "#EDB665", "#D97162",
    "#BD6C85", "#906AC0", "#B499D1", "#6E79B7", "#75B9C0",
]


def network_geometry(path):
    """Read physical links, coordinates and intermediate vertices from EPANET."""
    section = ""
    coordinates, links, vertices = {}, [], {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.upper()
            continue
        fields = line.split()
        if section == "[COORDINATES]" and len(fields) >= 3:
            coordinates[fields[0]] = [float(fields[1]), float(fields[2])]
        elif section in {"[PIPES]", "[PUMPS]", "[VALVES]"} and len(fields) >= 3:
            links.append((fields[0], fields[1], fields[2]))
        elif section == "[VERTICES]" and len(fields) >= 3:
            vertices.setdefault(fields[0], []).append([float(fields[1]), float(fields[2])])
    missing = {node for _, a, b in links for node in (a, b)} - coordinates.keys()
    if missing:
        raise ValueError(f"Missing physical node coordinates: {sorted(missing)}")
    return coordinates, links, vertices


def draw_network(ax, coordinates, links, vertices, assignments, sensors=(), selected=None):
    edge_segments, edge_colors, edge_widths = [], [], []
    for link_id, a, b in links:
        edge_segments.append([coordinates[a], *vertices.get(link_id, []), coordinates[b]])
        pa, pb = assignments[a], assignments[b]
        if selected is None:
            edge_colors.append(PARTITION_COLORS[pa] if pa == pb else "#B8C2D1")
            edge_widths.append(1.25 if pa == pb else 0.9)
        else:
            included = pa == selected and pb == selected
            edge_colors.append(PARTITION_COLORS[selected] if included else "#C5CEDB")
            edge_widths.append(2.0 if included else 1.0)
    ax.add_collection(LineCollection(edge_segments, colors=edge_colors, linewidths=edge_widths,
                                     capstyle="round", joinstyle="round", zorder=1))
    ids = list(coordinates)
    xy = np.asarray([coordinates[node] for node in ids])
    colors = [PARTITION_COLORS[assignments[node]] if selected is None or assignments[node] == selected
              else "#B3C0D2" for node in ids]
    sizes = [8 if selected is None or assignments[node] == selected else 4 for node in ids]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, edgecolors="white", linewidths=.25, zorder=2)
    if sensors:
        sensor_xy = np.asarray([coordinates[node] for node in sensors])
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=29, c="#E6545D", edgecolors="white",
                   linewidths=.8, zorder=4)
    if selected is not None:
        region_xy = np.asarray([coordinates[node] for node in ids if assignments[node] == selected])
        center = region_xy.mean(axis=0)
        ax.annotate(f"P{selected}", center, xytext=(9, 10), textcoords="offset points",
                    color=PARTITION_COLORS[selected], weight="bold", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#BBD2F2", lw=.8), zorder=5)
    all_xy = np.asarray([p for segment in edge_segments for p in segment])
    margin = np.ptp(all_xy, axis=0) * .045
    ax.set_xlim(all_xy[:, 0].min() - margin[0], all_xy[:, 0].max() + margin[0])
    ax.set_ylim(all_xy[:, 1].min() - margin[1], all_xy[:, 1].max() + margin[1])
    ax.set_aspect("equal")
    ax.axis("off")


def add_axes(fig, box):
    ax = fig.add_axes(box)
    ax.set_axis_off()
    return ax


def build(output):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 12, "text.color": NAVY, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    paths = {"network": ROOT / "data/Exa7.inp",
             "partitions": ROOT / "data/reference/partition_summary.json",
             "fingerprints": ROOT / "data/reference/sensor_fingerprints.json",
             "observation": ROOT / "data/example_observation.csv"}
    corpus = NumericalRetriever.from_file(paths["fingerprints"])
    observation = read_observations(paths["observation"], corpus.sensor_nodes)
    ranking = corpus.rank(observation, method="projection")
    prepared = corpus.prepare(observation)
    assignments = json.loads(paths["partitions"].read_text(encoding="utf-8"))["15"]["node_assignments"]
    coordinates, links, vertices = network_geometry(paths["network"])
    if corpus.D.shape != (540, 30) or len(set(assignments.values())) != 15:
        raise ValueError("Teaser expects the released 540 x 30 / 15-partition corpus")
    fig = plt.figure(figsize=(20, 6.5), facecolor="white")
    canvas = add_axes(fig, [0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    card_lefts = [.022, .267, .512, .757]
    width = .221
    y0, y1 = .062, .878
    for i, x in enumerate(card_lefts):
        canvas.add_patch(FancyBboxPatch((x, y0), width, y1-y0,
            boxstyle="round,pad=0.008,rounding_size=0.017", facecolor="#FBFCFE",
            edgecolor=STROKE, linewidth=1))
        canvas.add_patch(FancyBboxPatch((x+.010, .780), .029, .058,
            boxstyle="round,pad=0.002,rounding_size=0.012", facecolor=ACCENTS[i], edgecolor="none"))
        canvas.text(x+.0245, .809, str(i+1), ha="center", va="center", color="white", weight="bold", fontsize=14)
        canvas.text(x+.051, .810, ["Hydraulic graph", "Pressure library", "Projection match", "Partition ranking"][i],
                    va="center", fontsize=16, weight="bold")
        if i < 3:
            canvas.add_patch(FancyArrowPatch((x+width+.006, .470), (card_lefts[i+1]-.010, .470),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#94A4BA"))
    # Phase bars pair the two structural operations and the two online operations.
    for x0, x1, label, color in [(card_lefts[0], card_lefts[1]+width, "OFFLINE  /  MODEL + LIBRARY", ACCENTS[0]),
                               (card_lefts[2], card_lefts[3]+width, "ONLINE  /  OBSERVE + RETRIEVE", ACCENTS[3])]:
        canvas.plot([x0, x1], [.932, .932], color=color, lw=2, solid_capstyle="round")
        canvas.text(x0, .962, label, fontsize=11, weight="bold", color=MUTED,
                    fontfamily="DejaVu Sans Mono", va="center")
    # 1: Every physical node and edge uses its actual published coordinates.
    net = add_axes(fig, [.035, .242, .194, .496])
    draw_network(net, coordinates, links, vertices, assignments, corpus.sensor_nodes)
    canvas.text(card_lefts[0]+width/2, .164, "EXA7  ·  15 hydraulic partitions", fontsize=12, ha="center")
    canvas.scatter([card_lefts[0]+.047], [.113], s=28, c="#E6545D", edgecolor="white", linewidth=.6)
    canvas.text(card_lefts[0]+.057, .113, "30 pressure sensors", fontsize=11, va="center", color=MUTED)
    # 2: All released rows and channels, in original corpus/sensor order.
    maxima = np.max(np.abs(corpus.D), axis=1, keepdims=True)
    display_matrix = -corpus.D / np.where(maxima > 0, maxima, 1)
    cmap = LinearSegmentedColormap.from_list("pressure_signature", ["#F4FAFB", "#A8DCCE", "#31A78E", "#176A70", "#17375E"])
    heat = fig.add_axes([.291, .247, .173, .469])
    heat.imshow(display_matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap, rasterized=True)
    heat.set_xticks([])
    heat.set_yticks([])
    for spine in heat.spines.values():
        spine.set_visible(True); spine.set_edgecolor("#C7DBDE"); spine.set_linewidth(.7)
    canvas.text(card_lefts[1]+width/2, .164, "540 templates × 30 channels", fontsize=12, ha="center")
    canvas.text(card_lefts[1]+width/2, .113, "Row-normalized for display", fontsize=10.5, ha="center", color=MUTED)
    # 3: A deliberately declared schematic, with the exact projection rule below.
    geo = fig.add_axes([.531, .330, .178, .385])
    geo.set_xlim(-.07, 1.02); geo.set_ylim(-.08, 1.03); geo.set_aspect("equal"); geo.axis("off")
    y = np.array([.55, .83]); f = np.array([1., .35]); q = f * (f @ y) / (f @ f)
    geo.plot([0, .99], [0, .99*.35], color="#D7E2EF", lw=3, solid_capstyle="round")
    geo.add_patch(FancyArrowPatch((0, 0), y, arrowstyle="-|>", mutation_scale=18, lw=2.5, color=ACCENTS[2]))
    geo.add_patch(FancyArrowPatch((0, 0), q, arrowstyle="-|>", mutation_scale=18, lw=2.5, color=ACCENTS[3]))
    geo.plot([q[0], y[0]], [q[1], y[1]], "--", color="#65758D", lw=1.7, dashes=(3, 2.2))
    unit = (y-q) / np.linalg.norm(y-q)
    along = f / np.linalg.norm(f)
    square = np.asarray([q+.055*along, q+.055*along+.055*unit, q+.055*unit])
    geo.plot(square[:,0], square[:,1], color="#95A2B4", lw=1)
    geo.scatter([0], [0], s=18, c=NAVY, zorder=5)
    geo.text(y[0]+.025, y[1]+.025, "$\\mathbf{y}$", color=ACCENTS[2], fontsize=18)
    geo.text(q[0]+.015, q[1]-.09, "$a_j\\mathbf{f}_j$", color=ACCENTS[3], fontsize=16)
    geo.text(.85, .37, "$\\mathbf{f}_j$", color="#8191A7", fontsize=14)
    geo.text(.66, .64, "$r_j$", fontsize=16, color=MUTED)
    canvas.text(card_lefts[2]+width/2, .283, "Schematic · 30D → 2D", fontsize=10, ha="center", color=MUTED)
    canvas.text(card_lefts[2]+width/2, .206, r"$a_j=\frac{\mathbf{f}_j^\top\mathbf{y}}{\mathbf{f}_j^\top\mathbf{f}_j}$",
                ha="center", fontsize=16)
    canvas.text(card_lefts[2]+width/2, .116, r"$r_j=\|\mathbf{y}-a_j\mathbf{f}_j\|_2$", ha="center", fontsize=14)
    # 4: Actual prediction on the shipped example; separate partition ranks.
    result_net = add_axes(fig, [.774, .337, .190, .400])
    draw_network(result_net, coordinates, links, vertices, assignments, selected=ranking["predicted_partition"])
    top3_records = []
    for idx, partition in enumerate(ranking["top3_partitions"]):
        candidates = np.flatnonzero(corpus.partitions == partition)
        best = candidates[np.argmin(prepared["projection_residuals"][candidates])]
        record = {"rank": idx+1, "partition": partition, "best_scenario_id": corpus.ids[best],
                  "projection_residual": float(prepared["projection_residuals"][best])}
        top3_records.append(record)
        yy = .289-idx*.065
        rowcolor = "#E9F0FC" if idx == 0 else "#F0F3F8"
        canvas.add_patch(FancyBboxPatch((card_lefts[3]+.016, yy-.024), width-.032, .048,
            boxstyle="round,pad=.002,rounding_size=.008", fc=rowcolor, ec="none"))
        canvas.text(card_lefts[3]+.030, yy, f"{idx+1:02d}", va="center", fontsize=10.5,
                    color=MUTED, fontfamily="DejaVu Sans Mono")
        canvas.text(card_lefts[3]+.067, yy, f"Partition {partition}", va="center", fontsize=12,
                    color=PARTITION_COLORS[partition] if idx == 0 else NAVY,
                    weight="bold" if idx == 0 else "normal")
        if idx == 0:
            canvas.text(card_lefts[3]+width-.028, yy, "●", va="center", ha="right", fontsize=10,
                        color=PARTITION_COLORS[partition])
    canvas.text(card_lefts[3]+width/2, .101, "Released example observation", fontsize=10.5, ha="center", color=MUTED)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "teaser.png", dpi=100, facecolor="white")
    fig.savefig(output / "teaser.pdf", facecolor="white", metadata={"Title": "TS-GraphRAG: structure to retrieval",
                   "Author": "TS-GraphRAG", "Subject": "Data-derived README method overview",
                   "Creator": "scripts/readme/build_teaser.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    provenance = {
        "builder": "scripts/readme/build_teaser.py",
        "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths.values()},
        "canvas_pixels": [2000, 650], "partition_key": "15", "physical_nodes": len(coordinates),
        "physical_links": len(links), "sensors": corpus.sensor_nodes,
        "matrix_shape": list(corpus.D.shape),
        "matrix_display": "All rows/channels in original order; -D divided by row-wise max(abs(D)); zero rows displayed as zero. Display normalization is never used for inference.",
        "geometry": {"schematic": True, "illustrated_vectors": "A two-dimensional orthogonal projection illustrates the released unconstrained least-squares amplitude in the original 30-channel sensor space.",
                     "formula": "a_j = (f_j.T @ y) / (f_j.T @ f_j); r_j = ||y - a_j f_j||_2"},
        "actual_example": {"method": "projection", "predicted_partition": ranking["predicted_partition"],
                           "n_retained": ranking["n_retained"], "top3_distinct_partitions": top3_records},
        "fonts": ["Arial (DejaVu Sans fallback)", "DejaVu Sans Mono", "DejaVu mathtext"],
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    (output / "teaser_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"png": str(output / "teaser.png"), "pdf": str(output / "teaser.pdf"),
                      "top3_partitions": ranking["top3_partitions"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    build(parser.parse_args().output_dir)
