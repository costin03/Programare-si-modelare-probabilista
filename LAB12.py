import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/costin/PycharmProjects/PMP/Laborator/date_promovare_examen.csv')

# a)

counts = df['Promovare'].value_counts()
print("Balansarea datelor")
print(counts)
if counts[0] == counts[1]:
    print("Datele sunt PERFECT balansate.")
else:
    print("Datele nu sunt perfect balansate.")

X_studiu = df['Ore_Studiu'].values
X_somn = df['Ore_Somn'].values
y_promovare = df['Promovare'].values

with pm.Model() as model_promovare:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_studiu = pm.Normal('beta_studiu', mu=0, sigma=10)
    beta_somn = pm.Normal('beta_somn', mu=0, sigma=10)

    mu = alpha + beta_studiu * X_studiu + beta_somn * X_somn
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_promovare)
    idata = pm.sample(2000, return_inferencedata=True, chains=2)

summary = az.summary(idata, var_names=['alpha', 'beta_studiu', 'beta_somn'])
print("\nRezumatul parametrilor a posteriori:")
print(summary)

# b)

mean_alpha = idata.posterior['alpha'].mean().item()
mean_beta_studiu = idata.posterior['beta_studiu'].mean().item()
mean_beta_somn = idata.posterior['beta_somn'].mean().item()

print(f"Ecuatia granitei (p=0.5): {mean_alpha:.2f} + {mean_beta_studiu:.2f}*Studiu + {mean_beta_somn:.2f}*Somn = 0")

plt.figure(figsize=(10, 6))
plt.scatter(X_studiu[y_promovare == 1], X_somn[y_promovare == 1], color='green', label='Promovat (1)')
plt.scatter(X_studiu[y_promovare == 0], X_somn[y_promovare == 0], color='red', label='Nepromovat (0)')

x_vals = np.linspace(X_studiu.min(), X_studiu.max(), 100)
y_vals = -(mean_alpha + mean_beta_studiu * x_vals) / mean_beta_somn

plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Granita de decizie (medie)')
plt.xlabel('Ore Studiu')
plt.ylabel('Ore Somn')
plt.title('Granita de decizie si separarea claselor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# c)

print(f"Coeficient Ore Studiu: {mean_beta_studiu:.2f}")
print(f"Coeficient Ore Somn:   {mean_beta_somn:.2f}")

if abs(mean_beta_somn) > abs(mean_beta_studiu):
    print("Concluzie: Orele de somn influenteaza mai mult promovabilitatea.")
else:
    print("Concluzie: Orele de studiu influenteaza mai mult promovabilitatea.")