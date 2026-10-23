from ast import main
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causalnex.structure.notears import from_pandas_lasso
from causalnex.structure.structuremodel import StructureModel

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'CAFE_data'))
hdf_files = [
    "amazon_employee.hdf", "german_credit.hdf", "ionosphere.hdf",
    "messidor_features.hdf", "svmguide3.hdf", "wine_red.hdf", "wine_white.hdf", "airfoil.hdf", "housing_boston.hdf", 
    "openml_586.hdf", "openml_589.hdf", "openml_607.hdf",
    "openml_616.hdf", "openml_618.hdf", "openml_620.hdf"
]


def analyze_causal_graph(df, target):
    # Build causal graph
    sm = from_pandas_lasso(df, beta=0.05)
    direct_causes = []
    indirect_causes = []
    interventions = []
    causal_scores = {}

    # Direct causes: nodes with edges directly to target
    for node in sm.predecessors(target):
        direct_causes.append(node)
        causal_scores[node] = sm.get_edge_data(node, target)['weight']

    # Indirect causes: nodes with a path to target but not direct
    MAX_PATH_LENGTH = 3  # Limit search depth for indirect causes

    for node in sm.nodes:
        if node != target and node not in direct_causes:
            try:
                if nx.has_path(sm, node, target):
                    # Only consider short paths
                    found = False
                    max_score = None
                    for path in nx.all_simple_paths(sm, node, target, cutoff=MAX_PATH_LENGTH):
                        found = True
                        score = sum([sm.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
                        if (max_score is None) or (abs(score) > abs(max_score)):
                            max_score = score
                        break  # Only consider the first path for speed
                    if found:
                        indirect_causes.append(node)
                        if max_score is not None:
                            causal_scores[node] = max_score
            except Exception:
                continue
    
    interventions = direct_causes + sorted(
        [n for n in indirect_causes if abs(causal_scores.get(n, 0)) > 0.5],
        key=lambda n: -abs(causal_scores.get(n, 0))
    )
    plot_causal_graph(sm, target, direct_causes, indirect_causes)
    plot_causal_impact_bar(causal_scores, direct_causes, indirect_causes, target)

    return direct_causes, indirect_causes, interventions, causal_scores

def plot_causal_graph(sm, target, direct_causes, indirect_causes):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(sm, seed=42)

    # Node colors
    node_colors = []
    for node in sm.nodes:
        if node == target:
            node_colors.append('red')
        elif node in direct_causes:
            node_colors.append('orange')
        elif node in indirect_causes:
            node_colors.append('yellow')
        else:
            node_colors.append('lightblue')

    # Draw nodes
    nx.draw_networkx_nodes(sm, pos, node_color=node_colors, node_size=500, alpha=0.8)
    # Draw edges
    nx.draw_networkx_edges(sm, pos, arrows=True, arrowstyle='-|>', arrowsize=20, width=2, alpha=0.5)
    # Draw labels
    nx.draw_networkx_labels(sm, pos, font_size=10, font_color='black')

    # Optionally, draw edge weights
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in sm.edges(data=True)}
    nx.draw_networkx_edge_labels(sm, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f'Causal Graph for Target: {target}\nRed: Target, Orange: Direct, Yellow: Indirect, Blue: Others')
    plt.axis('off')
    plt.tight_layout()
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Figures'))
    plt.savefig(os.path.join(fig_dir, f'causal_graph_{target}.png'), dpi=300)

def plot_causal_impact_bar(causal_scores, direct_causes, indirect_causes, target):
    # Prepare data
    items = list(causal_scores.items())
    items.sort(key=lambda x: abs(x[1]), reverse=True)  # Sort by absolute impact

    variables = [k for k, v in items]
    impacts = [v for k, v in items]
    colors = []
    for var in variables:
        if var in direct_causes:
            colors.append('orange')
        elif var in indirect_causes:
            colors.append('green')
        else:
            colors.append('blue')

    plt.figure(figsize=(10, max(6, len(variables)//3)))
    bars = plt.barh(variables, impacts, color=colors)
    plt.xlabel('Causal Score (Impact on {})'.format(target))
    plt.title(f'Causal Impact of Variables on {target}\nOrange: Direct, Green: Indirect, Blue: Others')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # Save to parent directory
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Figures'))
    plt.savefig(os.path.join(fig_dir, f'causal_impact_bar_{target}.png'), dpi=300)

def save_causal_map(fname, direct_causes, indirect_causes, causal_scores, out_dir):
    causal_map = {}
    for f in causal_scores:
        if f in direct_causes:
            group = 'direct'
        elif f in indirect_causes:
            group = 'indirect'
        else:
            group = 'other'
        causal_map[f] = {'group': group, 'score': causal_scores[f]}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'causal_map_{fname}.json'), 'w') as f:
        json.dump(causal_map, f, indent=2)


for fname in hdf_files:
    print(f"\nProcessing {fname}")
    df = pd.read_hdf(os.path.join(data_dir, fname))
    target = df.columns[-1]
    direct, indirect, interventions, scores = analyze_causal_graph(df, target)
    print(f"Target: {target}")
    print(f"Direct causes: {direct}")
    print(f"Indirect causes: {indirect}")
    print(f"Interventions: {interventions}")
    print("Causal scores (w.r.t target):")
    for k, v in scores.items():
        print(f"  {k}: {v:.10f}")

if __name__ == "__main__":
    for fname in hdf_files:
        print(f"\nProcessing {fname}")
        df = pd.read_hdf(os.path.join(data_dir, fname))
        target = df.columns[-1]
        direct, indirect, interventions, scores = analyze_causal_graph(df, target)
        print(f"Target: {target}")
        print(f"Direct causes: {direct}")
        print(f"Indirect causes: {indirect}")
        print(f"Interventions: {interventions}")
        print("Causal scores (w.r.t target):")
        for k, v in scores.items():
            print(f"  {k}: {v:.10f}")
        save_causal_map(fname, direct, indirect, scores, out_dir=os.path.join(data_dir, 'causal_maps'))
  


