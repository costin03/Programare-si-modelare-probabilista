import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


df = pd.read_csv('date_colesterol.csv')
X = df['Ore_Exercitii'].values
y = df['Colesterol'].values

X_mean = X.mean()
X_std = X.std()
X_s = (X - X_mean) / X_std

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, c='gray')
plt.xlabel('Ore Exerciții')
plt.ylabel('Colesterol')
plt.title('Distributia datelor observate')
plt.show()

k_values = [3, 4, 5]
idatas = {}
models = {}

for K in k_values:
    print(f"\n--- Construire wi Antrenare Model pentru K={K} ---")

    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(K))
        alpha = pm.Normal('alpha', mu=y.mean(), sigma=50, shape=K)
        beta = pm.Normal('beta', mu=0, sigma=20, shape=K)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=K)

        sigma = pm.HalfNormal('sigma', sigma=20, shape=K)
        X_mat = X_s[:, None]

        mu = alpha + beta * X_mat + gamma * (X_mat ** 2)
        y_obs = pm.NormalMixture('y_obs', w=w, mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True, progressbar=True)

        pm.compute_log_likelihood(idata)

        idatas[K] = idata
        models[K] = model


for K in k_values:
    print(f"\nREZULTATE PENTRU K={K}")

    post = idatas[K].posterior
    mean_w = post['w'].mean(dim=["chain", "draw"]).values
    mean_alpha = post['alpha'].mean(dim=["chain", "draw"]).values
    mean_beta = post['beta'].mean(dim=["chain", "draw"]).values
    mean_gamma = post['gamma'].mean(dim=["chain", "draw"]).values

    print("Ponderi estimate (w):", np.round(mean_w, 3))

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, c='gray', alpha=0.3, s=10, label='Date')

    x_plot = np.linspace(X.min(), X.max(), 100)
    x_plot_s = (x_plot - X_mean) / X_std

    sorted_indices = np.argsort(mean_alpha)
    for idx in sorted_indices:
        y_plot = mean_alpha[idx] + mean_beta[idx] * x_plot_s + mean_gamma[idx] * (x_plot_s ** 2)
        plt.plot(x_plot, y_plot, linewidth=3, label=f'Grup {idx + 1} (w={mean_w[idx]:.2f})')

    plt.title(f'Model Mixtura de Regresii (K={K})')
    plt.xlabel('Ore Exercitii')
    plt.ylabel('Colesterol')
    plt.legend()
    plt.show()


print("\n--- COMPARATIE WAIC ---")
comp_dict = {f'K={k}': idatas[k] for k in k_values}

comp_waic = az.compare(comp_dict, ic="waic", scale="deviance")
print(comp_waic)

az.plot_compare(comp_waic, insample_dev=False)
plt.title("Compararea Modelelor (WAIC)")
plt.show()

best_k = comp_waic.index[0]
print(f"\nCONCLUZIE: Conform WAIC, cel mai bun model este cel cu {best_k}.")