from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random

model = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])


cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],
                           [0.1, 0.7]],
                   evidence=['S'],
                   evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['S'],
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],
                           [0.2, 0.6, 0.5, 0.9]],
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

print("Verificare model valid:", model.check_model())

print("\nIndependete in retea:")
independencies = model.get_independencies()
for ind in independencies.get_assertions():
    print(f"  {ind}")

inference = VariableElimination(model)

print("\nCLASIFICAREA EMAIL-URILOR")

test_cases = [
    {'O': 0, 'L': 0, 'M': 0},
    {'O': 0, 'L': 0, 'M': 1},
    {'O': 0, 'L': 1, 'M': 0},
    {'O': 0, 'L': 1, 'M': 1},
    {'O': 1, 'L': 0, 'M': 0},
    {'O': 1, 'L': 0, 'M': 1},
    {'O': 1, 'L': 1, 'M': 0},
    {'O': 1, 'L': 1, 'M': 1},
]

for evidence in test_cases:
    result = inference.query(variables=['S'], evidence=evidence)
    prob_spam = result.values[1]
    prob_not_spam = result.values[0]

    classification = "SPAM" if prob_spam > prob_not_spam else "NON-SPAM"

    print(f"\nEmail cu O={evidence['O']}, L={evidence['L']}, M={evidence['M']}:")
    print(f"  P(S=1|evidence) = {prob_spam:.4f}")
    print(f"  P(S=0|evidence) = {prob_not_spam:.4f}")
    print(f"  Clasificare: {classification}")




# Exercitiul 2



model = DiscreteBayesianNetwork([('DieRoll', 'AddedBall'), ('AddedBall', 'DrawnBall')])

cpd_die = TabularCPD(variable='DieRoll', variable_card=6,
                     values=[[1/6]]*6)

cpd_added = TabularCPD(variable='AddedBall', variable_card=3,
                       values=[
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [1, 1, 1, 0, 1, 0]],
                       evidence=['DieRoll'],
                       evidence_card=[6])

cpd_drawn = TabularCPD(variable='DrawnBall', variable_card=3,
                       values=[
                           [(3+1)/10, 3/10, 3/10],
                           [4/10, (4+1)/10, 4/10],
                           [2/10, 2/10, (2+1)/10]],
                       evidence=['AddedBall'],
                       evidence_card=[3])

model.add_cpds(cpd_die, cpd_added, cpd_drawn)

print("Model valid:", model.check_model())

infer = VariableElimination(model)

result = infer.query(variables=['DrawnBall'])
for val, prob in zip(['Red','Blue','Black'], result.values):
    print(f"{val}: {prob:.4f}")




# Exercitiul 3


def simulate_game():
    starter = random.choice(['P0','P1'])
    n = random.randint(1,6)
    if starter == 'P0':
        m = sum(random.choices([0,1], weights=[3,4], k=2*n))
        winner = 'P0' if n>=m else 'P1'
    else:
        m = sum(random.choices([0,1], weights=[1,1], k=2*n))
        winner = 'P1' if n>=m else 'P0'
    return starter, n, m, winner

def estimate_winner(num_simulations=10000):
    results = {'P0':0,'P1':0}
    for _ in range(num_simulations):
        _,_,_,winner = simulate_game()
        results[winner] += 1
    results['P0'] /= num_simulations
    results['P1'] /= num_simulations
    return results

results = estimate_winner()
print("Probabilitate castig P0:", results['P0'])
print("Probabilitate castig P1:", results['P1'])

model = DiscreteBayesianNetwork([('Starter','N')])

cpd_starter = TabularCPD('Starter',2,values=[[0.5],[0.5]])

cpd_n = TabularCPD(
    variable='N',
    variable_card=6,
    values=[
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6]
    ],
    evidence=['Starter'],
    evidence_card=[2]
)

model.add_cpds(cpd_starter,cpd_n)

infer = VariableElimination(model)

count_starter = {'P0':0,'P1':0}
simulations = 100000
for _ in range(simulations):
    starter, n, m, _ = simulate_game()
    if m==1:
        count_starter[starter] += 1

total = count_starter['P0'] + count_starter['P1']
print("Probabilitate Starter daca M=1:")
print("P0:", count_starter['P0']/total)
print("P1:", count_starter['P1']/total)
