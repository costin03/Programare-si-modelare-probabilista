import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from hmmlearn import hmm

partial_model = DiscreteBayesianNetwork([
    ('O', 'H'),('O', 'W'),('H', 'R'),('W', 'R'),('H', 'E'),('R', 'C')
])

CPD_O = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]], state_names={'O': ['cold', 'mild']})

CPD_H = TabularCPD(variable='H', variable_card=2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=['O'], evidence_card=[2],
                   state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']})

CPD_W = TabularCPD(variable='W', variable_card=2,values=[[0.1, 0.6], [0.9, 0.4]], evidence=['O'], evidence_card=[2],
                   state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']})

CPD_R = TabularCPD(variable='R', variable_card=2,
                   values=[[0.6, 0.9, 0.3, 0.5],
                           [0.4, 0.1, 0.7, 0.5]],
                   evidence=['H', 'W'], evidence_card=[2, 2],
                   state_names={'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']})

CPD_E = TabularCPD(variable='E', variable_card=2, values=[[0.8, 0.2],[0.2, 0.8]], evidence=['H'], evidence_card=[2],
                   state_names={'E': ['high', 'low'], 'H': ['yes', 'no']})

CPD_C = TabularCPD(variable='C', variable_card=2, values=[[0.85, 0.40], [0.15, 0.60]], evidence=['R'], evidence_card=[2],
                   state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']})

partial_model.add_cpds(CPD_O, CPD_H, CPD_W, CPD_R, CPD_E, CPD_C)
print(partial_model)
print(partial_model.check_model())

pos = nx.circular_layout(partial_model)
nx.draw(partial_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

# subpunctul b

infer = VariableElimination(partial_model)

print("\nP(H = yes | C = comfortable):")
p1 = infer.query(variables=['H'], evidence={'C': 'comfortable'})
print(p1)

print("\n2. P(E = high | C = comfortable):")
p2 = infer.query(variables=['E'], evidence={'C': 'comfortable'})
print(p2)


# Exercitiul 2


prob_initiale = np.array([0.4, 0.3, 0.3])

A = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])

B = np.array([
    [0.1, 0.7, 0.2],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])

# Am notat medium cu 1, high cu 2 si low cu 0
obs = [1, 2, 0]

# subpucntul b

def forward_algorithm(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = A.shape[0]
    alpha = np.zeros((T, N))

    alpha[0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, obs_seq[t]]

    return np.sum(alpha[T - 1])


prob_forward = forward_algorithm(obs, prob_initiale, A, B)
print(f"\nProbabilitatea: {prob_forward}")

# subpunctul c

def viterbi_algorithm(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = A.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(N):
            vals = delta[t - 1] * A[:, j]
            delta[t, j] = np.max(vals) * B[j, obs_seq[t]]
            psi[t, j] = np.argmax(vals)
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(delta[T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


best_path_indices = viterbi_algorithm(obs, prob_initiale, A, B)
states_map = {0: 'Walking', 1: 'Running', 2: 'Resting'}
best_path = [states_map[i] for i in best_path_indices]
print(f"\nCea mai probabila secventa de stari: {best_path}")

# subpunctul d

np.random.seed(42)
num_simulations = 10000
count_match = 0

target_obs = [1, 2, 0]

for _ in range(num_simulations):
    states = []

    current_state = np.random.choice(3, p=prob_initiale)
    states.append(current_state)

    for _ in range(2):
        next_state = np.random.choice(3, p=A[current_state])
        states.append(next_state)
        current_state = next_state

    generated_obs = []
    for s in states:
        obs_val = np.random.choice(3, p=B[s])
        generated_obs.append(obs_val)

    if generated_obs == target_obs:
        count_match += 1

empirical_prob = count_match / num_simulations
print(f"\nProbabilitatea empirica: {empirical_prob}")