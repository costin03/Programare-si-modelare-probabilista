import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

def main():
    publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                          6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
    sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                      15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfCauchy('sigma', 5)

        mu = alpha + beta * publicity

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=sales)

        idata = pm.sample(2000, tune=2000, return_inferencedata=True, progressbar=False)

    print("a), b)")
    summary = az.summary(idata, var_names=['alpha', 'beta', 'sigma'], hdi_prob=0.95)
    print(summary)

    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    alpha_mean = posterior_g['alpha'].mean().item()
    beta_mean = posterior_g['beta'].mean().item()

    print(f"\nEcuatia dreptei de regresie estimata: y = {alpha_mean:.2f} + {beta_mean:.2f} * x")

    x_new = 12.0
    posterior_predictive_new = posterior_g['alpha'] + posterior_g['beta'] * x_new
    mean_pred = posterior_predictive_new.mean().item()
    hdi_pred = az.hdi(posterior_predictive_new.values, hdi_prob=0.95)

    print(f"\nc) Predictie pentru cheltuieli de publicitate = {x_new}")
    print(f"Vanzari estimate (medie): {mean_pred:.2f}")
    print(f"Interval de predictie (95% HDI): {hdi_pred}")

    plt.figure(figsize=(10, 6))
    plt.scatter(publicity, sales, c='C0', label='Date observate')
    plt.plot(publicity, alpha_mean + beta_mean * publicity, c='k',
             label=f'Regresie: y = {alpha_mean:.2f} + {beta_mean:.2f} * x')

    alpha_vals = posterior_g['alpha'].values
    beta_vals = posterior_g['beta'].values
    regression_lines = alpha_vals[:, None] + beta_vals[:, None] * publicity

    az.plot_hdi(publicity, regression_lines, hdi_prob=0.95, color='gray')

    plt.xlabel('Cheltuieli publicitate')
    plt.ylabel('Vanzari')
    plt.legend()
    plt.title('Regresie Liniara Bayesiana')
    plt.show()

if __name__ == '__main__':
    main()