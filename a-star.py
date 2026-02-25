import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import heapq
from matplotlib.collections import LineCollection
import torch
import os
import io
from PIL import Image
from typing import Dict, List

# ─────────────────────────────────────────────
#  Palette — clean white / warm off-white theme
# ─────────────────────────────────────────────
P = {
    "bg":                  "#FAFAF8",
    "bg2":                 "#F3F2EF",
    "border":              "#E2E0DB",
    "text":                "#1A1A18",
    "subtext":             "#6B6860",
    "edge":                "#D4D0C8",
    "edge_weight":         "#A09C94",
    "path_edge":           "#2563EB",
    "node_default":        "#FFFFFF",
    "node_default_border": "#C8C4BC",
    "node_open":           "#F59E0B",
    "node_open_border":    "#D97706",
    "node_exp":            "#3B82F6",
    "node_exp_border":     "#1D4ED8",
    "node_path":           "#10B981",
    "node_path_border":    "#059669",
    "node_cur":            "#EF4444",
    "node_cur_border":     "#B91C1C",
}


# ─────────────────────────────────────────────
#  A* Core
# ─────────────────────────────────────────────
class AStarVisualizer:
    def __init__(self):
        self.steps: List[Dict] = []

    def reset(self):
        self.steps = []

    def _heuristic(self, positions, goal, kind, scale=1.0):
        gx, gy = positions[goal]
        h = {}
        for n, (x, y) in positions.items():
            dx, dy = abs(x - gx), abs(y - gy)
            if kind == "euclidean":
                h[n] = np.sqrt(dx**2 + dy**2) * scale
            elif kind == "manhattan":
                h[n] = (dx + dy) * scale
            else:
                h[n] = 0.0
        return h

    def _consistent(self, graph, h):
        for node in graph:
            for nbr, cost in graph[node].items():
                if h[node] > h[nbr] + cost + 1e-9:
                    return False
        return True

    def _admissible(self, graph, positions, goal, kind):
        scale = 1.0
        for _ in range(12):
            h = self._heuristic(positions, goal, kind, scale)
            if self._consistent(graph, h):
                return h
            scale *= 0.5
        return h

    def run(self, graph, positions, start, goal, kind):
        self.reset()
        h = self._admissible(graph, positions, goal, kind)
        heap, ctr = [], 0
        heapq.heappush(heap, (h[start], ctr, start))
        open_hash = {start}
        came_from = {}
        g = {n: float("inf") for n in graph}
        g[start] = 0
        f = {n: float("inf") for n in graph}
        f[start] = h[start]
        explored = []

        while heap:
            _, _, cur = heapq.heappop(heap)
            open_hash.discard(cur)
            explored.append(cur)
            self.steps.append({
                "current":   cur,
                "explored":  explored.copy(),
                "open_set":  list(open_hash),
                "came_from": came_from.copy(),
                "g": g.copy(), "f": f.copy(), "h": h,
                "path": self._path(came_from, cur),
            })
            if cur == goal:
                break
            for nbr, w in graph[cur].items():
                tg = g[cur] + w
                if tg < g[nbr]:
                    came_from[nbr] = cur
                    g[nbr] = tg
                    f[nbr] = tg + h[nbr]
                    if nbr not in open_hash:
                        ctr += 1
                        heapq.heappush(heap, (f[nbr], ctr, nbr))
                        open_hash.add(nbr)

    def _path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path


# ─────────────────────────────────────────────
#  Graph Rendering
# ─────────────────────────────────────────────
def render_frame(graph, positions, step, title, step_num, total):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    spread_x = max(xs) - min(xs) + 1e-6
    spread_y = max(ys) - min(ys) + 1e-6
    pad_x = spread_x * 0.20 + 0.9
    pad_y = spread_y * 0.24 + 0.9
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    path = step["path"]
    path_edge_set = set()
    if len(path) > 1:
        for i in range(len(path) - 1):
            path_edge_set.add((path[i], path[i + 1]))
            path_edge_set.add((path[i + 1], path[i]))

    # ── Edges ────────────────────────────────────
    seen = set()
    for node in graph:
        for nbr, w in graph[node].items():
            key = tuple(sorted([node, nbr]))
            if key in seen:
                continue
            seen.add(key)
            x1, y1 = positions[node]
            x2, y2 = positions[nbr]
            is_path_edge = (node, nbr) in path_edge_set

            if is_path_edge:
                # soft glow halo
                ax.plot([x1, x2], [y1, y2],
                        color=P["path_edge"], linewidth=10,
                        alpha=0.10, solid_capstyle="round", zorder=2)
                # main line
                ax.plot([x1, x2], [y1, y2],
                        color=P["path_edge"], linewidth=3.2,
                        alpha=0.95, solid_capstyle="round", zorder=3)
            else:
                ax.plot([x1, x2], [y1, y2],
                        color=P["edge"], linewidth=1.6,
                        alpha=0.9, solid_capstyle="round", zorder=1)

            # weight label — offset perpendicular to edge
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2) + 1e-9
            ox, oy = -dy / length * 0.20, dx / length * 0.20
            ax.text(mx + ox, my + oy, f"{w:.0f}",
                    ha="center", va="center", fontsize=7.5,
                    color=P["path_edge"] if is_path_edge else P["edge_weight"],
                    fontfamily="monospace",
                    fontweight="bold" if is_path_edge else "normal",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc=P["bg"], ec="none", alpha=0.88))

    # ── Nodes ────────────────────────────────────
    explored_set = set(step["explored"])
    open_set     = set(step["open_set"])
    path_set     = set(path)
    cur          = step["current"]
    node_r       = min(spread_x, spread_y) * 0.06 + 0.18   # adaptive radius

    for node, (nx_, ny) in positions.items():
        if node == cur:
            fc, ec, r, lw = P["node_cur"],     P["node_cur_border"],     node_r * 1.18, 2.8
        elif node in path_set:
            fc, ec, r, lw = P["node_path"],    P["node_path_border"],    node_r * 1.10, 2.2
        elif node in explored_set:
            fc, ec, r, lw = P["node_exp"],     P["node_exp_border"],     node_r,        2.0
        elif node in open_set:
            fc, ec, r, lw = P["node_open"],    P["node_open_border"],    node_r,        2.0
        else:
            fc, ec, r, lw = P["node_default"], P["node_default_border"], node_r,        1.6

        # subtle drop shadow
        shadow = plt.Circle((nx_ + 0.04, ny - 0.04), r,
                             facecolor=(0, 0, 0, 0.07), edgecolor="none", zorder=4)
        ax.add_patch(shadow)

        # node disc
        circle = plt.Circle((nx_, ny), r,
                             facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=5)
        ax.add_patch(circle)

        # node letter
        txt_color = "#FFFFFF" if fc != P["node_default"] else P["text"]
        ax.text(nx_, ny, node,
                ha="center", va="center",
                fontsize=max(8, int(node_r * 32)),
                fontweight="bold", color=txt_color,
                fontfamily="monospace", zorder=6)

        # score pill below the node
        g_v = step["g"][node]
        f_v = step["f"][node]
        h_v = step["h"][node]
        g_s = f"{g_v:.1f}" if g_v != float("inf") else "∞"
        f_s = f"{f_v:.1f}" if f_v != float("inf") else "∞"
        score = f"g={g_s}  h={h_v:.1f}  f={f_s}"
        ax.text(nx_, ny - r - 0.14, score,
                ha="center", va="top", fontsize=6.5,
                color=P["path_edge"] if node in path_set else P["subtext"],
                fontfamily="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22",
                          fc=P["bg2"], ec=P["border"],
                          linewidth=0.7, alpha=0.92))

    # ── Title & step badge ────────────────────────
    ax.text(0.013, 0.982, title,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold",
            color=P["text"], fontfamily="monospace")
    ax.text(0.013, 0.944, f"step {step_num} / {total}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=P["subtext"], fontfamily="monospace")

    # ── Footer path ───────────────────────────────
    if len(path) > 1:
        cost    = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        footer  = f"{' → '.join(path)}   ·   cost = {cost:.1f}"
        fc_text = P["path_edge"]
    else:
        footer  = "searching…"
        fc_text = P["subtext"]
    ax.text(0.5, 0.013, footer,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=fc_text, fontfamily="monospace")

    # ── Legend ────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=P["node_cur"],     edgecolor=P["node_cur_border"],     linewidth=1.2, label="Expanding"),
        mpatches.Patch(facecolor=P["node_path"],    edgecolor=P["node_path_border"],    linewidth=1.2, label="On path"),
        mpatches.Patch(facecolor=P["node_exp"],     edgecolor=P["node_exp_border"],     linewidth=1.2, label="Explored"),
        mpatches.Patch(facecolor=P["node_open"],    edgecolor=P["node_open_border"],    linewidth=1.2, label="Frontier"),
        mpatches.Patch(facecolor=P["node_default"], edgecolor=P["node_default_border"], linewidth=1.2, label="Unvisited"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper right",
                    framealpha=0.95, fontsize=8,
                    facecolor=P["bg2"], edgecolor=P["border"],
                    labelcolor=P["text"], handlelength=1.5,
                    borderpad=0.8, labelspacing=0.5)
    leg.get_frame().set_linewidth(0.8)

    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout(pad=0.9)
    return fig


def build_gif(graph, positions, steps, name, fps):
    os.makedirs("temp", exist_ok=True)
    frames, total = [], len(steps)
    for i, step in enumerate(steps):
        fig = render_frame(graph, positions, step, name, i + 1, total)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120,
                    facecolor=P["bg"], bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    out      = f"temp/{name}_astar.gif"
    duration = max(120, int(1000 / fps))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   optimize=True, duration=duration, loop=0)
    return out


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="A* Pathfinder", layout="wide", page_icon="🔍")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Sora:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        background-color: {P["bg"]} !important;
        color: {P["text"]} !important;
        font-family: 'Sora', sans-serif;
    }}
    .stApp {{ background-color: {P["bg"]}; }}

    [data-testid="stSidebar"] {{
        background-color: {P["bg2"]} !important;
        border-right: 1px solid {P["border"]} !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{ color: {P["text"]} !important; }}

    [data-testid="stSelectbox"] > div > div {{
        background-color: {P["bg"]} !important;
        border: 1px solid {P["border"]} !important;
        border-radius: 8px;
        color: {P["text"]} !important;
        font-family: 'DM Mono', monospace;
        font-size: 13px;
    }}

    .stButton > button {{
        background: {P["path_edge"]} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 0.55rem 1.4rem !important;
        width: 100% !important;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px #2563EB28;
        transition: opacity 0.15s ease;
    }}
    .stButton > button:hover {{ opacity: 0.86 !important; }}

    [data-testid="stMetric"] {{
        background-color: {P["bg2"]};
        border: 1px solid {P["border"]};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {P["subtext"]} !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-family: 'DM Mono', monospace;
    }}
    [data-testid="stMetricValue"] {{
        color: {P["text"]} !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 500;
    }}

    [data-testid="stTable"] table {{
        background-color: {P["bg2"]};
        border: 1px solid {P["border"]};
        border-radius: 10px;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        overflow: hidden;
    }}
    [data-testid="stTable"] th {{
        background-color: {P["bg"]} !important;
        color: {P["subtext"]} !important;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.07em;
        border-bottom: 1px solid {P["border"]} !important;
        padding: 8px 12px !important;
    }}
    [data-testid="stTable"] td {{
        color: {P["text"]} !important;
        border-bottom: 1px solid {P["border"]} !important;
        padding: 7px 12px !important;
    }}
    [data-testid="stTable"] tr:last-child td {{ border-bottom: none !important; }}

    [data-testid="stSlider"] * {{ color: {P["text"]} !important; }}

    .section-label {{
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        color: {P["subtext"]};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }}

    .path-strip {{
        display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0 16px;
    }}
    .path-node {{
        background: #EFF6FF; color: {P["path_edge"]};
        border: 1px solid #BFDBFE; border-radius: 6px;
        padding: 3px 10px;
        font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500;
    }}
    .path-arrow {{ color: {P["subtext"]}; font-size: 12px; line-height: 26px; }}

    hr {{ border: none; border-top: 1px solid {P["border"]} !important; margin: 1.1rem 0; }}
    .stAlert {{
        background-color: {P["bg2"]} !important;
        border: 1px solid {P["border"]} !important;
        border-radius: 8px !important;
    }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    [data-testid="stImage"] img {{
        border-radius: 12px;
        border: 1px solid {P["border"]};
        box-shadow: 0 4px 20px #00000010;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.6rem;">
        <div style="font-family:'DM Mono',monospace; font-size:11px;
                    color:{P['subtext']}; letter-spacing:0.12em;
                    text-transform:uppercase; margin-bottom:4px;">
            Graph Search Visualiser
        </div>
        <div style="font-family:'Sora',sans-serif; font-size:28px;
                    font-weight:700; color:{P['text']}; letter-spacing:-0.02em;">
            A* Pathfinding
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────
    for k, v in [("viz", None), ("last_map", None), ("gif_path", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-label">Map</div>', unsafe_allow_html=True)
        maps_dir = "maps"
        if not os.path.exists(maps_dir):
            st.error("`maps/` directory not found.")
            return
        available = sorted([f[:-3] for f in os.listdir(maps_dir) if f.endswith(".pt")])
        if not available:
            st.warning("No `.pt` map files found.")
            return

        selected_map = st.selectbox("", available, label_visibility="collapsed")
        if selected_map != st.session_state.last_map:
            st.session_state.viz      = AStarVisualizer()
            st.session_state.last_map = selected_map
            st.session_state.gif_path = None

        map_data = torch.load(f"{maps_dir}/{selected_map}.pt", weights_only=False)
        nodes    = sorted(map_data["graph"].keys())

        st.markdown("---")
        st.markdown('<div class="section-label">Nodes</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div style="font-size:11px;color:{P["subtext"]};margin-bottom:2px;">Start</div>', unsafe_allow_html=True)
            start_node = st.selectbox("", nodes, index=0, key="start", label_visibility="collapsed")
        with c2:
            st.markdown(f'<div style="font-size:11px;color:{P["subtext"]};margin-bottom:2px;">Goal</div>', unsafe_allow_html=True)
            end_node = st.selectbox("", nodes, index=len(nodes) - 1, key="goal", label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<div class="section-label">Heuristic</div>', unsafe_allow_html=True)
        heuristic_type = st.selectbox("", ["euclidean", "manhattan", "dijkstra"],
            format_func=lambda x: {
                "euclidean": "Euclidean  (L2)",
                "manhattan": "Manhattan  (L1)",
                "dijkstra":  "Dijkstra   (h = 0)",
            }[x], label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<div class="section-label">Playback speed (fps)</div>', unsafe_allow_html=True)
        fps = st.slider("", 1, 10, 2, label_visibility="collapsed")

        st.markdown("---")
        run_btn = st.button("▶  Run A* Search")

        if run_btn:
            if start_node == end_node:
                st.error("Start and goal must be different nodes.")
            else:
                viz = AStarVisualizer()
                viz.run(map_data["graph"], map_data["positions"],
                        start_node, end_node, heuristic_type)
                st.session_state.viz = viz
                with st.spinner("Rendering frames…"):
                    gif = build_gif(map_data["graph"], map_data["positions"],
                                    viz.steps, selected_map, fps)
                st.session_state.gif_path = gif

    # ── Main panel ────────────────────────────────
    viz: AStarVisualizer = st.session_state.viz

    if st.session_state.gif_path and viz and viz.steps:
        st.image(st.session_state.gif_path, use_container_width=True)
        st.markdown("---")
        final = viz.steps[-1]

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Nodes Explored", len(final["explored"]))
        with c2: st.metric("Search Steps",   len(viz.steps))
        with c3:
            if final["path"] and len(final["path"]) > 1:
                cost = sum(map_data["graph"][final["path"][i]][final["path"][i + 1]]
                           for i in range(len(final["path"]) - 1))
                st.metric("Path Cost", f"{cost:.1f}")
            else:
                st.metric("Path Cost", "—")
        with c4:
            st.metric("Path Length", len(final["path"]) if final["path"] else 0)

        if final["path"] and len(final["path"]) > 1:
            st.markdown('<div class="section-label" style="margin-top:1rem;">Optimal Path</div>', unsafe_allow_html=True)
            badges = "".join(
                f'<span class="path-node">{n}</span>'
                + (f'<span class="path-arrow"> › </span>' if i < len(final["path"]) - 1 else "")
                for i, n in enumerate(final["path"])
            )
            st.markdown(f'<div class="path-strip">{badges}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Node Scores</div>', unsafe_allow_html=True)
        rows = []
        for node in sorted(map_data["graph"].keys()):
            g_v = final["g"][node]
            f_v = final["f"][node]
            rows.append({
                "Node":            node,
                "g (from start)":  f"{g_v:.2f}" if g_v != float("inf") else "∞",
                "h (to goal)":     f"{final['h'][node]:.2f}",
                "f = g + h":       f"{f_v:.2f}" if f_v != float("inf") else "∞",
                "Status": (
                    "● Path"     if node in final["path"]     else
                    "● Explored" if node in final["explored"] else
                    "● Frontier" if node in final["open_set"] else
                    "○ Unvisited"
                ),
            })
        st.table(rows)

    else:
        st.markdown(f"""
        <div style="margin-top:3.5rem; text-align:center; padding:3rem 2rem;
                    background:{P['bg2']}; border:1px solid {P['border']};
                    border-radius:14px;">
            <div style="font-size:38px; margin-bottom:0.8rem;">🔍</div>
            <div style="font-family:'Sora',sans-serif; font-size:16px;
                        font-weight:600; color:{P['text']}; margin-bottom:6px;">
                Ready to search
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:12px;
                        color:{P['subtext']}; max-width:340px; margin:0 auto; line-height:1.7;">
                Select a map and nodes in the sidebar,<br>
                then click <strong>Run A* Search</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 How to read the graph"):
            st.markdown("""
| Node colour | Meaning |
|-------------|---------|
| 🔴 Red | Currently expanding |
| 🟢 Green | On the optimal path |
| 🔵 Blue | Already explored |
| 🟡 Amber | In the open frontier |
| ⚪ White | Not yet visited |

**Score labels** shown beneath each node:

| Label | Meaning |
|-------|---------|
| `g` | Exact cost from start to this node |
| `h` | Heuristic estimate to goal |
| `f = g + h` | Total estimated path cost |

Edge numbers show the travel cost between adjacent nodes.  
Blue edges highlight the current best path.
            """)


if __name__ == "__main__":
    main()
