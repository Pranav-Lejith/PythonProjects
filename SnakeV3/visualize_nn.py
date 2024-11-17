import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

def visualize_nn(model, input_size, filename='neural_network.png'):
    G = nx.DiGraph()
    layers = [input_size] + [l.out_features for l in model.children() if isinstance(l, nn.Linear)]
    
    pos = {}
    for i, layer_size in enumerate(layers):
        for j in range(layer_size):
            G.add_node(f"L{i}_{j}")
            pos[f"L{i}_{j}"] = (i, j - layer_size / 2)
    
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                G.add_edge(f"L{i}_{j}", f"L{i+1}_{k}")
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=100, node_color='lightblue', 
            edge_color='gray', arrows=True, arrowsize=10)
    
    plt.title("Neural Network Architecture")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Usage example:
from dqn_agent import DQN
model = DQN(11, 64, 3)
visualize_nn(model, 11)