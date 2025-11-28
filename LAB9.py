import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def main():
    y_values = [0, 5, 10]
    theta_values = [0.2, 0.5]

    fig_a, axes_a = plt.subplots(2, 3, figsize=(16, 9))
    fig_a.suptitle("Distributia a posteriori pentru n")

    fig_c, axes_c = plt.subplots(2, 3, figsize=(16, 9))
    fig_c.suptitle("Distributia predictiva a posteriori pentru Y*")

    axes_a = axes_a.flatten()
    axes_c = axes_c.flatten()

    plot_idx = 0

    for theta in theta_values:
        for y in y_values:
            with pm.Model() as model:
                n = pm.Poisson("n", mu=10)

                obs = pm.Binomial("obs", n=n, p=theta, observed=y)

                idata = pm.sample(1000, return_inferencedata=True, progressbar=False)

                az.plot_posterior(idata, var_names=["n"], ax=axes_a[plot_idx])
                axes_a[plot_idx].set_title(f"Y = {y}, theta = {theta}")

                pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True, progressbar=False)

                posterior_predictive_samples = idata.posterior_predictive["obs"]
                az.plot_dist(posterior_predictive_samples, ax=axes_c[plot_idx])
                axes_c[plot_idx].set_title(f"Y* | Y = {y}, theta = {theta}")
                axes_c[plot_idx].set_xlabel("Y* (Cumparatori viitori)")

            plot_idx += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Voi scrie aici subpunctele b) si d)

# Subpunctul b

# 1. Efectul lui Y (numarul observat de cumparatori):
#    Odata cu cresterea numarului de cumparatori observati (Y), distributia a posteriori pentru n se deplaseaza catre dreapta
#    (spre valori mai mari). Acest lucru este intuitiv: daca vedem mai multi cumparatori, este necesar ca numarul total de vizitatori (n)
#    sa fie cel putin la fel de mare ca Y. De asemenea, limita inferioara a distributiei va fi intotdeauna cel putin Y.
#
# 2. Efectul lui theta (probabilitatea de cumparare):
#    Theta actioneaza invers proportional cu media lui n. Relatia aproximativa este n ~ Y / theta.
#    - Daca theta este mic (ex. 0.2): Pentru a obtine acelasi numar de cumparatori Y, avem nevoie de un numar mult mai mare
#    de vizitatori n. Distributia a posteriori va fi centrata pe valori mai mari si va fi mai lata (incertitudine mai mare),
#    deoarece un theta mic face procesul mai putin determinist.

#    - Daca theta este mare (ex. 0.5): Numarul total de vizitatori n estimat va fi mai mic (mai apropiat de Y),
#    iar distributia va fi mai ingusta (mai precisa), deoarece o rata mare de conversie ne ofera mai multa informatie despre n.

# Subpunctul d

# 1. Natura estimarii:
#    Distributia a posteriori pentru n estimeaza un parametru latent (numarul total de vizitatori care au fost deja in magazin),
#    in timp ce distributia predictiva a posteriori pentru Y* estimeaza o data observabila viitoare (cati oameni vor cumpara produsul data viitoare).
#
# 2. Incertitudinea (Varianta):
#    Distributia predictiva a posteriori este intotdeauna mai "lata" (are o varianta mai mare) decat distributia a posteriori a parametrului n.
#    Acest lucru se intampla deoarece predictiva incorporeaza doua straturi de incertitudine:
#    a) Incertitudinea epistemica legata de valoarea reala a lui n (reflectata in distributia a posteriori).
#    b) Incertitudinea aleatoare a procesului de esantionare viitor (zgomotul inerent al distributiei Binomiale pentru noii clienti).
#
# 3. Scara valorilor:
#    Valorile din distributia a posteriori pentru n sunt pe o scara a numarului total de vizitatori. Valorile din distributia predictiva Y* sunt pe o scara a numarului de cumparatori.
#    Deoarece theta < 1, media lui Y* va fi mai mica decat media lui n (aproximativ n * theta).