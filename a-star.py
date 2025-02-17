import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import heapq
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import torch
import os
import random
from typing import Dict, List, Tuple, Set
import matplotlib.animation as animation
import io
import base64
from PIL import Image

class AStarVisualizer:
    def __init__(self):
        self.reset_simulation()
        
    def reset_simulation(self):
        self.steps = []
        self.current_step = 0
        self.is_playing = False
        self.speed = 1.0
        
    def calculate_heuristic(self, positions: Dict, goal: str, heuristic_type: str = "euclidean", scale_factor: float = 1.0) -> Dict:
        heuristic = {}
        goal_pos = positions[goal]

        for node, pos in positions.items():
            dx = abs(pos[0] - goal_pos[0])
            dy = abs(pos[1] - goal_pos[1])
            dist = 0 
            if heuristic_type == "euclidean":
                dist = np.sqrt(dx**2 + dy**2)
            elif heuristic_type == "manhattan":
                dist = dx + dy
            elif heuristic_type == "dijkstra":
                dist = 0

            heuristic[node] = dist * scale_factor
        
        return heuristic

    def verify_heuristic_consistency(self, graph: Dict, heuristic: Dict) -> Tuple[bool, List]:
        inconsistencies = []
        
        for node in graph:
            for neighbor, cost in graph[node].items():
                if heuristic[node] > heuristic[neighbor] + cost:
                    inconsistencies.append({
                        'node': node,
                        'neighbor': neighbor,
                        'h_node': heuristic[node],
                        'h_neighbor': heuristic[neighbor],
                        'cost': cost,
                        'difference': heuristic[node] - (heuristic[neighbor] + cost)
                    })
        
        return len(inconsistencies) == 0, inconsistencies

    def ensure_consistent_heuristic(self, graph: Dict, positions: Dict, goal: str, type: str = "manhattan") -> Dict:
        scale_factor = 1.0
        max_attempts = 10
        attempt = 1

        while attempt <= max_attempts:
            heuristic = self.calculate_heuristic(positions, goal, type, scale_factor)
            is_consistent, _ = self.verify_heuristic_consistency(graph, heuristic)

            if is_consistent:
                return heuristic
            
            scale_factor *= 0.5
            attempt += 1

        print("Inconsistent heuristic")
        return heuristic

    def run_astar(self, graph: Dict, positions: Dict, start: str, goal: str, type: str) -> None:
        self.reset_simulation()
        heuristic = self.ensure_consistent_heuristic(graph, positions, goal, type)
        
        open_set = []
        counter = 0
        heapq.heappush(open_set, (heuristic[start], counter, start))
        came_from = {}
        g_score = {node: float('inf') for node in graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph}
        f_score[start] = heuristic[start]
        open_set_hash = {start}
        explored = []
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            explored.append(current)
            
            self.steps.append({
                'current': current,
                'explored': explored.copy(),
                'open_set': list(open_set_hash),
                'came_from': came_from.copy(),
                'g_score': g_score.copy(),
                'f_score': f_score.copy(),
                'path': self.reconstruct_path(came_from, current),
                'heuristic': heuristic
            })
            
            if current == goal:
                break
                
            for neighbor, weight in graph[current].items():
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)

    def reconstruct_path(self, came_from: Dict, current: str) -> List[str]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def create_visualization(self, graph: Dict, positions: Dict, step_data: Dict, city_name: str, step_num: int, total_steps: int) -> plt.Figure:
        plt.clf()
        fig = plt.figure(figsize=(12, 8), facecolor='#F8F9FA')
        ax = plt.gca()
        ax.set_facecolor('#F8F9FA')

        positions_x = [x for x, _ in positions.values()]
        positions_y = [y for _, y in positions.values()]
        margin = 0.8
        plt.xlim(min(positions_x) - margin, max(positions_x) + margin)
        plt.ylim(min(positions_y) - margin, max(positions_y) + margin)

        G = nx.Graph()
        for node in graph:
            G.add_node(node)
            for neighbor, weight in graph[node].items():
                G.add_edge(node, neighbor, weight=weight)

        edges_normal = []
        edges_path = []
        path = step_data['path']

        for (u, v) in G.edges():
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            if path and u in path and v in path:
                i1, i2 = path.index(u), path.index(v)
                if abs(i1 - i2) == 1:
                    edges_path.append(((x1, y1), (x2, y2)))
                else:
                    edges_normal.append(((x1, y1), (x2, y2)))
            else:
                edges_normal.append(((x1, y1), (x2, y2)))

        lc_normal = LineCollection(edges_normal, colors='#DEE2E6', linewidths=2, alpha=0.7)
        ax.add_collection(lc_normal)

        if edges_path:
            lc_path = LineCollection(edges_path, colors='#FA5252', linewidths=3, alpha=0.8)
            ax.add_collection(lc_path)

        nx.draw_networkx_nodes(G, positions,
                             node_color='#F8F9FA',
                             node_size=250,
                             edgecolors='#CED4DA',
                             linewidths=2)

        if step_data['open_set']:
            nx.draw_networkx_nodes(G, positions,
                                 nodelist=step_data['open_set'],
                                 node_color='#FFC107',
                                 node_size=300,
                                 edgecolors='#CED4DA',
                                 linewidths=2)

        if step_data['explored']:
            nx.draw_networkx_nodes(G, positions,
                                 nodelist=step_data['explored'],
                                 node_color='#74C0FC',
                                 node_size=300,
                                 edgecolors='#CED4DA',
                                 linewidths=2)

        if path:
            nx.draw_networkx_nodes(G, positions,
                                 nodelist=path,
                                 node_color='#40C057',
                                 node_size=350,
                                 edgecolors='#FFFFFF',
                                 linewidths=2.5)

        labels = {}
        for node in G.nodes():
            g = step_data['g_score'][node]
            f = step_data['f_score'][node]
            h = step_data['heuristic'][node]
            label = f"{node}\ng={g:.1f}\nh={h:.1f}\nf={f:.1f}"
            labels[node] = label

        label_pos = {k: (v[0], v[1]+0.2) for k, v in positions.items()}
        nx.draw_networkx_labels(G, label_pos,
                              labels=labels,
                              font_size=8,
                              font_color='#495057',
                              font_weight='bold',
                              font_family='sans-serif')

        plt.title(f"A* Search - {city_name}\nStep {step_num}/{total_steps}",
                 fontsize=16,
                 pad=20,
                 fontweight='bold',
                 fontfamily='sans-serif',
                 color='#212529')

        if path:
            cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            info_text = f"Path: {' → '.join(path)}\nCost: {cost:.1f}"
            plt.figtext(0.5, 0.02, info_text,
                       ha="center",
                       fontsize=10,
                       fontfamily='sans-serif',
                       color='#495057')

        ax.set_axis_off()
        plt.tight_layout()
        return fig
    
    def create_animation(self, graph: Dict, positions: Dict, city_name: str, fps: int = 1) -> str:
        if not self.steps:
            return None
            
        fig = plt.figure(figsize=(12, 8), facecolor='#F8F9FA')
        frames = []
        
        for i, step_data in enumerate(self.steps):
            plt.clf()
            fig = self.create_visualization(
                graph, 
                positions, 
                step_data, 
                city_name,
                i + 1,
                len(self.steps)
            )
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)
            plt.close(fig)
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        output_path = f'temp/{city_name}_astar.gif'
        duration = int(1000 / fps)
        
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration,
            loop=0
        )
        
        return output_path

def main():
    st.set_page_config(page_title="A* Search", layout="wide")
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #F8F9FA;
        }
        .centered-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("A* Search")
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = AStarVisualizer()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'last_map' not in st.session_state:
        st.session_state.last_map = None
    if 'animation_path' not in st.session_state:
        st.session_state.animation_path = None
    
    with st.sidebar:
        st.header("Settings")
        
        available_maps = [f[:-3] for f in os.listdir('maps') if f.endswith('.pt')]
        if not available_maps:
            st.warning("No maps found")
            return
        
        selected_map = st.selectbox("Map", available_maps)
        
        if selected_map != st.session_state.last_map:
            st.session_state.visualizer = AStarVisualizer()
            st.session_state.current_step = 0
            st.session_state.last_map = selected_map
            st.session_state.animation_path = None
        
        map_data = torch.load(f'maps/{selected_map}.pt')
        
        nodes = sorted(map_data['graph'].keys())
        types = ['euclidean', 'manhattan','djisktra']
        col1, col2 = st.columns(2)
        with col1:
            start_node = st.selectbox("Start", nodes, index=0)
        with col2:
            end_node = st.selectbox("Goal", nodes, index=len(nodes)-1)
        
        type = st.selectbox("Type of Heuristic", types, index = 0)
        fps = st.slider("Speed", 1, 10, 2, 1)
        
        if st.button("Run Search"):
            st.session_state.visualizer.run_astar(
                map_data['graph'],
                map_data['positions'],
                start_node,
                end_node, 
                type
            )
            
            with st.spinner("Generating..."):
                animation_path = st.session_state.visualizer.create_animation(
                    map_data['graph'],
                    map_data['positions'],
                    selected_map,
                    fps=fps
                )
                st.session_state.animation_path = animation_path
    
    if st.session_state.animation_path:
        st.image(st.session_state.animation_path, use_column_width=True)
        
        if st.session_state.visualizer.steps:
            final_step = st.session_state.visualizer.steps[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes Explored", len(final_step['explored']))
            with col2:
                if final_step['path']:
                    path_cost = sum(map_data['graph'][final_step['path'][i]][final_step['path'][i+1]] 
                                   for i in range(len(final_step['path'])-1))
                    st.metric("Path Cost", f"{path_cost:.1f}")
            with col3:
                st.metric("Steps", len(st.session_state.visualizer.steps))
            
            if final_step['path']:
                st.write("Path found: " + " → ".join(final_step['path']))
                st.subheader("Node Scores")
                scores_data = []
                for node in map_data['graph'].keys():
                    scores_data.append({
                        "Node": node,
                        "g": f"{final_step['g_score'][node]:.1f}" if final_step['g_score'][node] != float('inf') else "∞",
                        "h": f"{final_step['heuristic'][node]:.1f}",
                        "f": f"{final_step['f_score'][node]:.1f}" if final_step['f_score'][node] != float('inf') else "∞",
                        "Status": ("Path" if node in final_step['path'] else 
                                  "Explored" if node in final_step['explored'] else 
                                  "Open" if node in final_step['open_set'] else 
                                  "Unexplored")
                    })
                st.table(scores_data)
    else:
        st.info("Select start and goal nodes, then click 'Run Search'")
        st.markdown("""
        ### Legend
        - 🟢 Path
        - 🔵 Explored
        - 🟡 Open
        - ⚪ Unexplored
        
        ### Score Labels
        - g = Path cost from start
        - h = Distance to goal
        - f = Total cost
        """)

if __name__ == "__main__":
    main()
