import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# Exercitiul 1

# a)


model = MarkovNetwork()

nodes = ['A1', 'A2', 'A3', 'A4', 'A5']
model.add_nodes_from(nodes)

edges = [
    ('A1', 'A2'),
    ('A1', 'A3'),
    ('A2', 'A4'),
    ('A2', 'A5'),
    ('A3', 'A4'),
    ('A4', 'A5')
]
model.add_edges_from(edges)

plt.figure(figsize=(10, 8))
pos = {
    'A1': (0.5, 2),
    'A2': (0, 1),
    'A3': (1, 1),
    'A4': (0.5, 0),
    'A5': (1.5, 0)
}

G = nx.Graph()
G.add_edges_from(edges)

nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=16, font_weight='bold',
        edge_color='gray', width=2, arrows=False)
plt.axis('off')
plt.tight_layout()
plt.savefig('ex1.png', dpi=300, bbox_inches='tight')

cliques = list(nx.find_cliques(G))
print(f"\nClicile modelului (total: {len(cliques)}):")
for i, clique in enumerate(cliques, 1):
    print(f"  C{i}: {{{', '.join(sorted(clique))}}}")



# b)

t1 = 0.5
t2 = 0.5
cardinality = 2

def compute_factor_values(var1, var2, t1, t2):
    values = np.zeros((2, 2))
    states = [-1, 1]
    for i in range(2):
        for j in range(2):
            values[i, j] = np.exp(t1 * states[i] + t2 * states[j])

    return values


factors = []

print("\nFactorii pentru fiecare clica:")
for edge in edges:
    var1, var2 = edge
    values = compute_factor_values(var1, var2, t1, t2)

    factor = DiscreteFactor(
        variables=[var1, var2],
        cardinality=[2, 2],
        values=values.flatten()
    )
    factors.append(factor)

    print(f"\nFi({var1}, {var2}):")
    print(f"  {var1}=-1, {var2}=-1: {values[0, 0]:.4f}")
    print(f"  {var1}=-1, {var2}=+1: {values[0, 1]:.4f}")
    print(f"  {var1}=+1, {var2}=-1: {values[1, 0]:.4f}")
    print(f"  {var1}=+1, {var2}=+1: {values[1, 1]:.4f}")

model.add_factors(*factors)

print(f"  Noduri: {model.nodes()}")
print(f"  Muchii: {model.edges()}")
print(f"  Numar de factori: {len(model.get_factors())}")
print(f"  Model este valid: {model.check_model()}")


bp = BeliefPropagation(model)

print("\nDistributii marginale:")
for node in nodes:
    marginal = bp.query(variables=[node])
    print(f"\nP({node}):")
    print(f"  {node} = -1 (index 0): {marginal.values[0]:.6f}")
    print(f"  {node} = +1 (index 1): {marginal.values[1]:.6f}")
    print(f"  Starea mai probabila: {'+1' if marginal.values[1] > marginal.values[0] else '-1'}")

best_config = None
best_prob = -np.inf
all_configs = []

from itertools import product

for config in product([0, 1], repeat=5):
    evidence = {nodes[i]: config[i] for i in range(5)}

    prob = 1.0
    for factor in factors:
        var1, var2 = factor.variables
        idx1 = evidence[var1]
        idx2 = evidence[var2]
        prob *= factor.get_value(**{var1: idx1, var2: idx2})

    all_configs.append((config, prob))

    if prob > best_prob:
        best_prob = prob
        best_config = config

all_configs.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 configuratii cu probabilitate maxima:")
print(f"{'Rank':<6} {'A1':<4} {'A2':<4} {'A3':<4} {'A4':<4} {'A5':<4}")

for rank, (config, prob) in enumerate(all_configs[:5], 1):
    state_str = ['+1' if c == 1 else '-1' for c in config]
    print(
        f"{rank:<6} {state_str[0]:<4} {state_str[1]:<4} {state_str[2]:<4} {state_str[3]:<4} {state_str[4]:<4} {prob:<20.6f}")

print("\nREZULTAT FINAL:")
best_state = ['+1' if c == 1 else '-1' for c in best_config]
print(f"\nConfiguratia optima (MAP):")
print(f"  A1 = {best_state[0]}")
print(f"  A2 = {best_state[1]}")
print(f"  A3 = {best_state[2]}")
print(f"  A4 = {best_state[3]}")
print(f"  A5 = {best_state[4]}")
print(f"\nProbabilitate nenormalizata: {best_prob:.6f}")