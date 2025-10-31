import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

n_states = 3
n_observations = 4

transition_matrix = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

emission_matrix = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

start_probability = np.array([1/3, 1/3, 1/3])

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

print("\nMatricea de tranzitie:")
print("     Diff   Med    Easy")
print(f"Diff {transition_matrix[0]}")
print(f"Med  {transition_matrix[1]}")
print(f"Easy {transition_matrix[2]}")

print("\nMatricea de emisie:")
print("     FB    B     S     NS")
print(f"Diff {emission_matrix[0]}")
print(f"Med  {emission_matrix[1]}")
print(f"Easy {emission_matrix[2]}")

G = nx.DiGraph()
states = ['Difficult', 'Medium', 'Easy']

for i, state in enumerate(states):
    G.add_node(state)

for i in range(len(states)):
    for j in range(len(states)):
        if transition_matrix[i][j] > 0:
            G.add_edge(states[i], states[j], weight=transition_matrix[i][j])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50)

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

for (u, v, d) in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, [(u, v)], width=2,
                          connectionstyle='arc3,rad=0.1', arrowsize=20)

edge_labels = {}
for i in range(len(states)):
    for j in range(len(states)):
        if transition_matrix[i][j] > 0:
            edge_labels[(states[i], states[j])] = f'{transition_matrix[i][j]}'

nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)

plt.title('Diagrama de stari a HMM-ului', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('hmm_state_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

observations = np.array([[0], [0], [2], [1], [1], [2], [1], [1], [3], [1], [1]])

log_prob = model.score(observations)
prob = np.exp(log_prob)

# b)

print(f"Log-probabilitate: {log_prob}")
print(f"Probabilitate: {prob:.10e}")

log_prob_viterbi, hidden_states = model.decode(observations, algorithm="viterbi")
prob_viterbi = np.exp(log_prob_viterbi)

state_names = ['Difficult', 'Medium', 'Easy']
state_sequence = [state_names[s] for s in hidden_states]

# c)

print(f"Secventa: {' -> '.join(state_sequence)}")
print(f"Log-probabilitate: {log_prob_viterbi}")
print(f"Probabilitate: {prob_viterbi:.10e}")

print("\nDetalii secventa:")
grade_names = ['FB', 'B', 'S', 'NS']
for i, (obs, state) in enumerate(zip(observations, hidden_states)):
    print(f"Test {i+1}: Grade {grade_names[obs[0]]}, Dificultate {state_names[state]}")

print("\nDiagrama de stari salvata in: hmm_state_diagram.png")