import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import heapq
from matplotlib.collections import LineCollection
import torch
import os
import io
from PIL import Image
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
#  Core A* Logic
# ─────────────────────────────────────────────

class AStarVisualizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps: List[Dict] = []

    def heuristic(self, positions: Dict, goal: str, kind: str, scale: float = 1.0) -> Dict:
        gx, gy = positions[goal]
        h = {}
        for node, (nx_, ny) in positions.items():
            dx, dy = abs(nx_ - gx), abs(ny - gy)
            if kind == "euclidean":
                h[node] = np.sqrt(dx**2 + dy**2) * scale
            elif kind == "manhattan":
                h[node] = (dx + dy) * scale
            else:  # dijkstra / zero heuristic
                h[node] = 0.0
        return h

    def _consistent(self, graph: Dict, h: Dict) -> bool:
        for node in graph:
            for nbr, cost in graph[node].items():
                if h[node] > h[nbr] + cost + 1e-9:
                    return False
        return True

    def admissible_heuristic(self, graph: Dict, positions: Dict, goal: str, kind: str) -> Dict:
        scale = 1.0
        for _ in range(12):
            h = self.heuristic(positions, goal, kind, scale)
            if self._consistent(graph, h):
                return h
            scale *= 0.5
        return h  # best effort

    def run(self, graph: Dict, positions: Dict, start: str, goal: str, kind: str):
        self.reset()
        h = self.admissible_heuristic(graph, positions, goal, kind)

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (h[start], counter, start))
        open_hash = {start}
        came_from: Dict = {}
        g = {n: float("inf") for n in graph}
        g[start] = 0
        f = {n: float("inf") for n in graph}
        f[start] = h[start]
        explored: List[str] = []

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            open_hash.discard(cur)
            explored.append(cur)

            self.steps.append({
                "current": cur,
                "explored": explored.copy(),
                "open_set": list(open_hash),
                "came_from": came_from.copy(),
                "g": g.copy(),
                "f": f.copy(),
                "h": h,
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
                        counter += 1
                        heapq.heappush(open_heap, (f[nbr], counter, nbr))
                        open_hash.add(nbr)

    def _path(self, came_from: Dict, cur: str) -> List[str]:
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path


# ─────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────

PALETTE = {
    "bg":        "#0D1117",
    "bg2":       "#161B22",
    "border":    "#30363D",
    "text":      "#E6EDF3",
    "subtext":   "#8B949E",
    "edge":      "#30363D",
    "path_edge": "#58A6FF",
    "node_def":  "#21262D",
    "node_open": "#D29922",
    "node_exp":  "#388BFD",
    "node_path": "#3FB950",
    "node_cur":  "#F85149",
    "node_bdr":  "#484F58",
}

def render_frame(graph: Dict, positions: Dict, step: Dict, title: str, step_num: int, total: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    path = step["path"]
    path_edges = set()
    if path:
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i+1]))
            path_edges.add((path[i+1], path[i]))

    seen_edges = set()
    normal_segs, path_segs = [], []
    for node in graph:
        for nbr in graph[node]:
            key = tuple(sorted([node, nbr]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            x1, y1 = positions[node]
            x2, y2 = positions[nbr]
            seg = ((x1, y1), (x2, y2))
            if (node, nbr) in path_edges:
                path_segs.append(seg)
            else:
                normal_segs.append(seg)

    ax.add_collection(LineCollection(normal_segs, colors=PALETTE["edge"], linewidths=1.5, alpha=0.5, zorder=1))
    if path_segs:
        ax.add_collection(LineCollection(path_segs, colors=PALETTE["path_edge"], linewidths=3.0, alpha=0.9, zorder=2))

    # Edge weight labels
    for node in graph:
        for nbr, w in graph[node].items():
            if node < nbr:
                mx = (positions[node][0] + positions[nbr][0]) / 2
                my = (positions[node][1] + positions[nbr][1]) / 2
                ax.text(mx, my, f"{w:.0f}", ha="center", va="center",
                        fontsize=7, color=PALETTE["subtext"], zorder=3,
                        bbox=dict(boxstyle="round,pad=0.15", fc=PALETTE["bg"], ec="none", alpha=0.8))

    # Draw nodes
    explored_set = set(step["explored"])
    open_set = set(step["open_set"])
    path_set = set(path)

    for node, (nx_, ny) in positions.items():
        if node == step["current"]:
            color = PALETTE["node_cur"]
            size = 420
            border = "#FFFFFF"
            bw = 2.5
        elif node in path_set:
            color = PALETTE["node_path"]
            size = 380
            border = "#FFFFFF"
            bw = 2
        elif node in explored_set:
            color = PALETTE["node_exp"]
            size = 340
            border = PALETTE["node_bdr"]
            bw = 1.5
        elif node in open_set:
            color = PALETTE["node_open"]
            size = 340
            border = PALETTE["node_bdr"]
            bw = 1.5
        else:
            color = PALETTE["node_def"]
            size = 300
            border = PALETTE["node_bdr"]
            bw = 1.5

        circle = plt.Circle((nx_, ny), 0.22, color=color, zorder=4, linewidth=bw,
                             linestyle="-", fill=True, ec=border)
        ax.add_patch(circle)
        ax.text(nx_, ny, node, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#FFFFFF", zorder=5)

        # Score label
        g_val = step["g"][node]
        h_val = step["h"][node]
        f_val = step["f"][node]
        g_str = f"{g_val:.1f}" if g_val != float("inf") else "∞"
        f_str = f"{f_val:.1f}" if f_val != float("inf") else "∞"
        score_txt = f"g={g_str}  h={h_val:.1f}\nf={f_str}"
        ax.text(nx_, ny - 0.42, score_txt, ha="center", va="top",
                fontsize=6.5, color=PALETTE["subtext"], zorder=5,
                fontfamily="monospace")

    # Title
    ax.set_title(
        f"{title}  ·  Step {step_num} / {total}",
        color=PALETTE["text"], fontsize=14, fontweight="bold", pad=14,
        fontfamily="monospace", loc="left"
    )

    # Path info footer
    if path:
        cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
        footer = f"Path: {' → '.join(path)}   |   Cost: {cost:.1f}"
    else:
        footer = "Searching…"
    ax.text(0.5, -0.03, footer, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color=PALETTE["subtext"], fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE["node_cur"],  label="Current"),
        mpatches.Patch(color=PALETTE["node_path"], label="Path"),
        mpatches.Patch(color=PALETTE["node_exp"],  label="Explored"),
        mpatches.Patch(color=PALETTE["node_open"], label="Open"),
        mpatches.Patch(color=PALETTE["node_def"],  label="Unvisited"),
    ]
    legend = ax.legend(handles=legend_items, loc="upper right",
                       framealpha=0.15, labelcolor=PALETTE["text"],
                       fontsize=8, facecolor=PALETTE["bg2"],
                       edgecolor=PALETTE["border"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    ax.set_axis_off()
    plt.tight_layout(pad=1.2)
    return fig


def build_gif(graph, positions, steps, city_name, fps) -> str:
    os.makedirs("temp", exist_ok=True)
    frames = []
    total = len(steps)
    for i, step in enumerate(steps):
        fig = render_frame(graph, positions, step, city_name, i + 1, total)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=PALETTE["bg"])
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    out = f"temp/{city_name}_astar.gif"
    duration = max(100, int(1000 / fps))
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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        background-color: {PALETTE["bg"]} !important;
        color: {PALETTE["text"]} !important;
        font-family: 'Space Grotesk', sans-serif;
    }}
    .stApp {{ background-color: {PALETTE["bg"]}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {PALETTE["bg2"]} !important;
        border-right: 1px solid {PALETTE["border"]};
    }}
    [data-testid="stSidebar"] * {{ color: {PALETTE["text"]} !important; }}

    /* Selectbox / inputs */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div {{
        background-color: {PALETTE["bg"]} !important;
        border: 1px solid {PALETTE["border"]} !important;
        border-radius: 6px;
        color: {PALETTE["text"]} !important;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Buttons */
    .stButton > button {{
        background: {PALETTE["node_path"]} !important;
        color: {PALETTE["bg"]} !important;
        border: none;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 14px;
        padding: 0.5rem 1.5rem;
        width: 100%;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{ opacity: 0.85; }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {PALETTE["bg2"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {PALETTE["subtext"]} !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    [data-testid="stMetricValue"] {{ color: {PALETTE["text"]} !important; font-family: 'JetBrains Mono', monospace; font-size: 22px; }}

    /* Table */
    [data-testid="stTable"] table {{
        background-color: {PALETTE["bg2"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }}
    [data-testid="stTable"] th {{
        background-color: {PALETTE["bg"]} !important;
        color: {PALETTE["subtext"]} !important;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.06em;
        border-bottom: 1px solid {PALETTE["border"]};
    }}
    [data-testid="stTable"] td {{ color: {PALETTE["text"]} !important; border-bottom: 1px solid {PALETTE["border"]}; }}

    /* Slider */
    [data-testid="stSlider"] * {{ color: {PALETTE["text"]} !important; }}

    /* Info box */
    .stAlert {{ background-color: {PALETTE["bg2"]} !important; border: 1px solid {PALETTE["border"]} !important; color: {PALETTE["subtext"]} !important; }}

    /* Divider */
    hr {{ border-color: {PALETTE["border"]} !important; }}

    /* Remove default top padding */
    .block-container {{ padding-top: 2rem; }}

    /* Page title */
    .page-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: {PALETTE["text"]};
        letter-spacing: -0.02em;
    }}
    .page-subtitle {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: {PALETTE["subtext"]};
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }}
    .badge {{
        display: inline-block;
        background: {PALETTE["node_path"]}22;
        color: {PALETTE["node_path"]};
        border: 1px solid {PALETTE["node_path"]}55;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        margin-right: 6px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────
    st.markdown('<p class="page-title">A* Pathfinding Visualiser</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Interactive step-by-step graph search</p>', unsafe_allow_html=True)

    # ── Session state ────────────────────────
    for key, default in [("viz", None), ("current_step", 0),
                         ("last_map", None), ("gif_path", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        maps_dir = "maps"
        if not os.path.exists(maps_dir):
            st.error("No `maps/` directory found.")
            return
        available = sorted([f[:-3] for f in os.listdir(maps_dir) if f.endswith(".pt")])
        if not available:
            st.warning("No `.pt` map files found in `maps/`.")
            return

        selected_map = st.selectbox("🗺️  Map", available)

        if selected_map != st.session_state.last_map:
            st.session_state.viz = AStarVisualizer()
            st.session_state.last_map = selected_map
            st.session_state.gif_path = None

        map_data = torch.load(f"{maps_dir}/{selected_map}.pt", weights_only=False)
        nodes = sorted(map_data["graph"].keys())

        st.markdown("#### Nodes")
        col1, col2 = st.columns(2)
        with col1:
            start_node = st.selectbox("Start", nodes, index=0)
        with col2:
            end_node = st.selectbox("Goal", nodes, index=len(nodes) - 1)

        st.markdown("#### Heuristic")
        heuristic_type = st.selectbox(
            "Function",
            ["euclidean", "manhattan", "dijkstra"],
            format_func=lambda x: {"euclidean": "Euclidean (L2)", "manhattan": "Manhattan (L1)", "dijkstra": "Dijkstra (h=0)"}[x]
        )

        st.markdown("#### Playback")
        fps = st.slider("Frames per second", 1, 10, 2)

        st.markdown("---")
        run_btn = st.button("▶  Run A* Search")

        if run_btn:
            if start_node == end_node:
                st.error("Start and goal must differ.")
            else:
                viz = AStarVisualizer()
                viz.run(map_data["graph"], map_data["positions"], start_node, end_node, heuristic_type)
                st.session_state.viz = viz

                with st.spinner("Rendering animation…"):
                    gif = build_gif(map_data["graph"], map_data["positions"],
                                    viz.steps, selected_map, fps)
                st.session_state.gif_path = gif

    # ── Main content ─────────────────────────
    viz: AStarVisualizer = st.session_state.viz

    if st.session_state.gif_path and viz and viz.steps:
        st.image(st.session_state.gif_path, use_container_width=True)

        st.markdown("---")
        final = viz.steps[-1]
        found = final["path"] and final["path"][-1] == final["path"][-1]

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes Explored", len(final["explored"]))
        with col2:
            st.metric("Search Steps", len(viz.steps))
        with col3:
            if final["path"]:
                cost = sum(map_data["graph"][final["path"][i]][final["path"][i+1]]
                           for i in range(len(final["path"]) - 1))
                st.metric("Path Cost", f"{cost:.1f}")
            else:
                st.metric("Path Cost", "N/A")
        with col4:
            st.metric("Path Length", len(final["path"]) if final["path"] else 0)

        # Path display
        if final["path"]:
            st.markdown("**Optimal path found:**")
            badges = "".join(f'<span class="badge">{n}</span>' for n in final["path"])
            st.markdown(badges, unsafe_allow_html=True)

        # Node scores table
        st.markdown("#### Node Score Summary")
        rows = []
        for node in sorted(map_data["graph"].keys()):
            g_v = final["g"][node]
            f_v = final["f"][node]
            status = (
                "🟢 Path" if node in final["path"] else
                "🔵 Explored" if node in final["explored"] else
                "🟡 Open" if node in final["open_set"] else
                "⚪ Unvisited"
            )
            rows.append({
                "Node": node,
                "g  (cost from start)": f"{g_v:.2f}" if g_v != float("inf") else "∞",
                "h  (heuristic)": f"{final['h'][node]:.2f}",
                "f  (total)": f"{f_v:.2f}" if f_v != float("inf") else "∞",
                "Status": status,
            })
        st.table(rows)

    else:
        # Empty state
        st.markdown(f"""
        <div style="text-align:center; padding: 4rem 2rem; color: {PALETTE['subtext']};">
            <div style="font-size:48px; margin-bottom:1rem;">🔍</div>
            <p style="font-family:'JetBrains Mono',monospace; font-size:14px;">
                Select a map, choose start & goal nodes, then click <strong style="color:{PALETTE['node_path']}">Run A* Search</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 How to read the visualisation"):
            st.markdown(f"""
| Symbol | Meaning |
|--------|---------|
| 🔴 Red node | Node currently being expanded |
| 🟢 Green node | Node on the optimal path |
| 🔵 Blue node | Already explored |
| 🟡 Yellow node | In the open (frontier) set |
| ⚪ Grey node | Not yet visited |
| **g** | Exact cost from start to this node |
| **h** | Heuristic estimate to goal |
| **f = g + h** | Total estimated cost |
            """)


if __name__ == "__main__":
    main()
