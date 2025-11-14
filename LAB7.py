import pymc as pm
import numpy as np
import arviz as az

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

print("\na)")

x = 60
print(f"x = {x} dB")
print(f"\nPrior pentru mu: N({x}, 10^2)")
print(f"Prior pentru sigma: HalfNormal(10)")

with pm.Model() as model:
    mu = pm.Normal('mu', mu=x, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42, progressbar=False)

print("\nb)")

hdi_mu = az.hdi(trace.posterior['mu'], hdi_prob=0.95)
hdi_sigma = az.hdi(trace.posterior['sigma'], hdi_prob=0.95)

hdi_mu_vals = hdi_mu.to_array().values.flatten()
hdi_sigma_vals = hdi_sigma.to_array().values.flatten()

print(f"\nmu: media posteriori = {trace.posterior['mu'].mean().item():.2f} dB")
print(f"    HDI 95% = [{hdi_mu_vals[0]:.2f}, {hdi_mu_vals[1]:.2f}] dB")

print(f"\nsigma: media posteriori = {trace.posterior['sigma'].mean().item():.2f} dB")
print(f"       HDI 95% = [{hdi_sigma_vals[0]:.2f}, {hdi_sigma_vals[1]:.2f}] dB")

print("\nc)")

sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)

bayes_mean_mu = trace.posterior['mu'].mean().item()
bayes_mean_sigma = trace.posterior['sigma'].mean().item()

print(f"\nFrecventist:")
print(f"  Media: {sample_mean:.2f} dB")
print(f"  SD: {sample_std:.2f} dB")

print(f"\nBayesian:")
print(f"  mu: {bayes_mean_mu:.2f} dB")
print(f"  sigma: {bayes_mean_sigma:.2f} dB")

print(f"\nDiferente:")
print(f"  Delta mu: {abs(bayes_mean_mu - sample_mean):.2f} dB")
print(f"  Delta sigma: {abs(bayes_mean_sigma - sample_std):.2f} dB")

print("\nTabel")
print(f"\n{'Parametru':<15} {'Frecventist':<15} {'Bayesian':<15} {'HDI 95%':<25}")
print(f"{'mu':<15} {sample_mean:<15.2f} {bayes_mean_mu:<15.2f} [{hdi_mu_vals[0]:.2f}, {hdi_mu_vals[1]:.2f}]")
print(f"{'sigma':<15} {sample_std:<15.2f} {bayes_mean_sigma:<15.2f} [{hdi_sigma_vals[0]:.2f}, {hdi_sigma_vals[1]:.2f}]")

print("\n\nd)")

print(f"\nPrior pentru mu: N(50, 1^2)")
print(f"Prior pentru sigma: HalfNormal(10)")

with pm.Model() as model_strong:
    mu_strong = pm.Normal('mu', mu=50, sigma=1)
    sigma_strong = pm.HalfNormal('sigma', sigma=10)
    y_obs_strong = pm.Normal('y_obs', mu=mu_strong, sigma=sigma_strong, observed=data)
    trace_strong = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42, progressbar=False)

hdi_mu_strong = az.hdi(trace_strong.posterior['mu'], hdi_prob=0.95)
hdi_sigma_strong = az.hdi(trace_strong.posterior['sigma'], hdi_prob=0.95)

hdi_mu_strong_vals = hdi_mu_strong.to_array().values.flatten()
hdi_sigma_strong_vals = hdi_sigma_strong.to_array().values.flatten()

bayes_mean_mu_strong = trace_strong.posterior['mu'].mean().item()
bayes_mean_sigma_strong = trace_strong.posterior['sigma'].mean().item()

print(f"\nmu: media posteriori = {bayes_mean_mu_strong:.2f} dB")
print(f"    HDI 95% = [{hdi_mu_strong_vals[0]:.2f}, {hdi_mu_strong_vals[1]:.2f}] dB")

print(f"\nsigma: media posteriori = {bayes_mean_sigma_strong:.2f} dB")
print(f"       HDI 95% = [{hdi_sigma_strong_vals[0]:.2f}, {hdi_sigma_strong_vals[1]:.2f}] dB")

print("\nComparatie:")
print(f"\n{'Parametru':<15} {'Prior slab':<15} {'Prior puternic':<15} {'Diferenta':<15}")
print(f"{'mu':<15} {bayes_mean_mu:<15.2f} {bayes_mean_mu_strong:<15.2f} {abs(bayes_mean_mu - bayes_mean_mu_strong):<15.2f}")
print(f"{'sigma':<15} {bayes_mean_sigma:<15.2f} {bayes_mean_sigma_strong:<15.2f} {abs(bayes_mean_sigma - bayes_mean_sigma_strong):<15.2f}")
