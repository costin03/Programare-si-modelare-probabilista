import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Ex 1

df = pd.read_csv('bike_daily.csv')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df['temp_c'], df['rentals'])
plt.xlabel('Temperature')
plt.ylabel('Rentals')
plt.subplot(1, 3, 2)
plt.scatter(df['humidity'], df['rentals'])
plt.xlabel('Humidity')
plt.subplot(1, 3, 3)
plt.scatter(df['wind_kph'], df['rentals'])
plt.xlabel('Wind Speed')
plt.tight_layout()
plt.show()

# Ex 2

cols_std = ['temp_c', 'humidity', 'wind_kph']
for c in cols_std:
    df[c + '_std'] = (df[c] - df[c].mean()) / df[c].std()

df['rentals_std'] = (df['rentals'] - df['rentals'].mean()) / df['rentals'].std()

season_dummies = pd.get_dummies(df['season'], drop_first=True, dtype=int)
season_cols = season_dummies.columns.tolist()
df = pd.concat([df, season_dummies], axis=1)

X_cols_lin = ['temp_c_std', 'humidity_std', 'wind_kph_std', 'is_holiday'] + season_cols
X_lin = df[X_cols_lin].values
y_std = df['rentals_std'].values

print('Rulare Model Liniar...')
with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, shape=X_lin.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=10)

    mu = alpha + pm.math.dot(X_lin, betas)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_std)

    idata_lin = pm.sample(draws=1000, chains=2, target_accept=0.9, return_inferencedata=True, progressbar=False)
    pm.compute_log_likelihood(idata_lin)

df['temp_sq_std'] = df['temp_c_std'] ** 2
X_cols_poly = ['temp_c_std', 'temp_sq_std', 'humidity_std', 'wind_kph_std', 'is_holiday'] + season_cols
X_poly = df[X_cols_poly].values

print('Rulare Model Polinomial...')
with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, shape=X_poly.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=10)

    mu = alpha + pm.math.dot(X_poly, betas)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_std)

    idata_poly = pm.sample(draws=1000, chains=2, target_accept=0.9, return_inferencedata=True, progressbar=False)
    pm.compute_log_likelihood(idata_poly)

# Ex 3

print('Rezumat Model Liniar:')
print(az.summary(idata_lin, var_names=['betas']))

print('Rezumat Model Polinomial:')
print(az.summary(idata_poly, var_names=['betas']))

beta_means_lin = np.abs(idata_lin.posterior['betas'].mean(dim=['chain', 'draw']).values)
idx_lin = np.argmax(beta_means_lin)
print(f'Cea mai influenta variabila Model Liniar: {X_cols_lin[idx_lin]}')

beta_means_poly = np.abs(idata_poly.posterior['betas'].mean(dim=['chain', 'draw']).values)
idx_poly = np.argmax(beta_means_poly)
print(f'Cea mai influenta variabila Model Polinomial: {X_cols_poly[idx_poly]}')

print('Afisare Trace Plots...')
az.plot_trace(idata_lin, var_names=['alpha', 'betas', 'sigma'])
plt.suptitle('Trace Plot - Liniar')
plt.show()

az.plot_trace(idata_poly, var_names=['alpha', 'betas', 'sigma'])
plt.suptitle('Trace Plot - Polinomial')
plt.show()

# Ex 4

comp = az.compare({'Linear': idata_lin, 'Polynomial': idata_poly}, ic='waic', scale='deviance')
print('Rezultate comparatie WAIC:')
print(comp)

az.plot_compare(comp, insample_dev=False)
plt.title('Comparatie WAIC')
plt.show()

if comp.index[0] == 'Polynomial':
    print('Modelul Polinomial este preferat conform WAIC.')
    best_model = model_poly
    best_idata = idata_poly
else:
    print('Modelul Liniar este preferat conform WAIC.')
    best_model = model_linear
    best_idata = idata_lin

print('Generare Posterior Predictive Check...')
with best_model:
    ppc = pm.sample_posterior_predictive(best_idata)

y_pred_mean = ppc.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
hdi = az.hdi(ppc.posterior_predictive['y_obs'])
hdi_vals = hdi['y_obs'].values

plt.figure(figsize=(10, 6))
temp_vals = df['temp_c'].values
rentals_std_vals = df['rentals_std'].values
idx = np.argsort(temp_vals)

plt.scatter(temp_vals, rentals_std_vals, color='k', alpha=0.5, label='Date observate')
plt.plot(temp_vals[idx], y_pred_mean[idx], color='C1', label='Medie predictiva')
plt.fill_between(temp_vals[idx], hdi_vals[idx, 0], hdi_vals[idx, 1], color='C1', alpha=0.3, label='HDI 94%')
plt.xlabel('Temperature (Original)')
plt.ylabel('Rentals (Standardized)')
plt.title('Posterior Predictive Check')
plt.legend()
plt.show()

# Ex 5

Q = df['rentals'].quantile(0.75)
df['is_high_demand'] = (df['rentals'] >= Q).astype(int)
y_binary = df['is_high_demand'].values

print('Rulare Regresie Logistica...')
with pm.Model() as model_logistic:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, shape=X_poly.shape[1])

    mu = alpha + pm.math.dot(X_poly, betas)
    p = pm.Deterministic('p', pm.math.sigmoid(mu))

    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_binary)

    idata_log = pm.sample(draws=1000, chains=2, target_accept=0.9, return_inferencedata=True, progressbar=False)

print('Rezumat Regresie Logistica:')
print(az.summary(idata_log, var_names=['alpha', 'betas']))

az.plot_trace(idata_log, var_names=['alpha', 'betas'])
plt.suptitle('Trace Plot - Regresie Logistica')
plt.show()

# Ex 7

post_log = idata_log.posterior
mean_betas_log = post_log['betas'].mean(dim=['chain', 'draw']).values
hdi_log = az.hdi(idata_log, var_names=['betas'], hdi_prob=0.95)['betas'].values

print('Coeficienti Regresie Logistica si HDI 95%:')
for i, col in enumerate(X_cols_poly):
    print(f'{col}: Mean={mean_betas_log[i]:.4f}, HDI=[{hdi_log[i, 0]:.4f}, {hdi_log[i, 1]:.4f}]')

max_idx = np.argmax(np.abs(mean_betas_log))
print(f'Cea mai influenta variabila pentru High Demand este: {X_cols_poly[max_idx]}')