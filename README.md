# A* Search Visualization

A interactive visualization tool for the A* pathfinding algorithm using Streamlit and NetworkX.

## Setup

1. Install dependencies:
```bash
pip install streamlit networkx matplotlib numpy torch pillow
```
2. Create a `maps` directory in your project folder:
```bash
mkdir maps
```
3. Place your map files (`.pt` format) in the `maps` directory. Each map file should contain:
   - A graph dictionary defining node connections and weights
   - A positions dictionary defining node coordinates

## Usage

1. Run the application:
```bash
streamlit run a-star.py
```
2. In the web interface:
   - Select a map from the dropdown
   - Choose start and goal nodes
   - Adjust animation speed if needed
   - Click "Run Search" to start the visualization

## Requirements

- Python 3.7+
- Streamlit
- NetworkX
- Matplotlib
- NumPy
- PyTorch
- Pillow

## Project Structure

```
project/
│
├── a-atar.py              # Main application file
├── maps/             # Directory for map files
│   └── *.pt         # Map data files
└── temp/            # Generated visualizations
```#
