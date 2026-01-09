import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('date.csv', header=None, delim_whitespace=True)
except:
    df = pd.read_csv('date.csv', header=None)
df.columns = ['x', 'y']
df = df.sort_values(by='x')

x_mean = df['x'].mean()
x_std = df['x'].std()
x_s = (df['x'].values - x_mean) / x_std
y_data = df['y'].values


def run_polynomial_model(x, y, order, beta_sd, samples=1000):
    X_poly = np.vstack([x ** i for i in range(1, order + 1)]).T

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)

        betas = pm.Normal('betas', mu=0, sigma=beta_sd, shape=order)

        sigma = pm.HalfNormal('sigma', sigma=10)

        mu = alpha + pm.math.dot(X_poly, betas)

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(samples, return_inferencedata=True, progressbar=False)

    return idata, model

# ex 1
order = 5

idata_10, _ = run_polynomial_model(x_s, y_data, order=5, beta_sd=10)

idata_100, _ = run_polynomial_model(x_s, y_data, order=5, beta_sd=100)

sd_array = np.array([10, 0.1, 0.1, 0.1, 0.1])
idata_custom, _ = run_polynomial_model(x_s, y_data, order=5, beta_sd=sd_array)

plt.figure(figsize=(12, 6))
plt.scatter(df['x'], y_data, color='k', label='Date', zorder=5)

X_poly = np.vstack([x_s ** i for i in range(1, order + 1)]).T

for idata, label, color, style in zip([idata_10, idata_100, idata_custom],
                                      ['SD=10', 'SD=100 (Wiggly)', 'SD=[10, 0.1...] (Smooth)'],
                                      ['blue', 'red', 'green'],
                                      ['-', '--', '-.']):
    post = idata.posterior
    mean_alpha = post['alpha'].mean().item()
    mean_betas = post['betas'].mean(dim=["chain", "draw"]).values
    y_pred = mean_alpha + np.dot(X_poly, mean_betas)

    plt.plot(df['x'], y_pred, color=color, linestyle=style, linewidth=2, label=label)

plt.title("Exercitiul 1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ex 2
np.random.seed(42)
x_500 = np.linspace(df['x'].min(), df['x'].max(), 500)
y_500 = 1.5 * x_500**3 - 3 * x_500**2 - 5 * x_500 + np.random.normal(0, 10, 500)
x_500_s = (x_500 - x_500.mean()) / x_500.std()
idata_500, _ = run_polynomial_model(x_500_s, y_500, order=5, beta_sd=100)

plt.figure(figsize=(10, 5))
plt.scatter(x_500, y_500, s=10, color='gray', alpha=0.5, label='Date (N=500)')

X_poly_500 = np.vstack([x_500_s**i for i in range(1, 6)]).T
post_500 = idata_500.posterior
y_pred_500 = post_500['alpha'].mean().item() + np.dot(X_poly_500, post_500['betas'].mean(dim=["chain", "draw"]).values)

plt.plot(x_500, y_pred_500, color='red', linewidth=2, label='Model Ord 5 (SD=100)')
plt.title("Exercitiul 2")
plt.legend()
plt.show()


idata_ord1, model_ord1 = run_polynomial_model(x_s, y_data, order=1, beta_sd=10)
idata_ord2, model_ord2 = run_polynomial_model(x_s, y_data, order=2, beta_sd=10)
idata_ord3, model_ord3 = run_polynomial_model(x_s, y_data, order=3, beta_sd=10)

for idata, model in zip([idata_ord1, idata_ord2, idata_ord3], [model_ord1, model_ord2, model_ord3]):
    if 'log_likelihood' not in idata:
        with model:
            pm.compute_log_likelihood(idata)

compare_dict = {
    'Linear (Ord 1)': idata_ord1,
    'Quadratic (Ord 2)': idata_ord2,
    'Cubic (Ord 3)': idata_ord3
}

comp_waic = az.compare(compare_dict, ic="waic", scale="deviance")
comp_loo = az.compare(compare_dict, ic="loo", scale="deviance")

print(comp_waic)

az.plot_compare(comp_waic, insample_dev=False)
plt.title("Exercitiul 3")
plt.show()