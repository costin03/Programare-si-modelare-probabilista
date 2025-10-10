import random
import numpy as np
import matplotlib.pyplot as plt


# exercitiul 1

def subpunctul_a():

    urn = [0, 0, 0, 1, 1, 1, 1, 2, 2]
    # am ales ca bilele rosii sa fie '0' in urna, cele albastre '1', iar
    # cele negre sa fie '2'.

    die_roll = random.randint(1, 6)

    if die_roll in [2, 3, 5]:
        urn.append(2)
    elif die_roll == 6:
        urn.append(0)
    elif die_roll in [1, 4]:
        urn.append(1)

    bila_extrasa = random.choice(urn)
    # vedem ce bila am extras
    print(bila_extrasa)
    # vedem daca bila a fost rosie
    if bila_extrasa == 0:
        return True
    else:
        return False

print(subpunctul_a())


def subpunctul_b(nr_simulari):
    nr_rosii = 0

    for _ in range(nr_simulari):
        if subpunctul_a():
            nr_rosii += 1

    probabilitate = nr_rosii / nr_simulari
    return probabilitate, nr_rosii

print(subpunctul_b(3))


# exercitiul 2

def subpunctul_1():
    n = 1000
    x1 = np.random.poisson(1, n)
    x2 = np.random.poisson(2, n)
    x3 = np.random.poisson(5, n)
    x4 = np.random.poisson(10, n)

    return x1, x2, x3, x4

print (subpunctul_1())


def subpunctul_2():
    lambdas = [1, 2, 5, 10]
    X_randomized = []

    for _ in range(1000):
        lambda_ales = random.choice(lambdas)
        valoare = np.random.poisson(lam=lambda_ales)
        X_randomized.append(valoare)

    return np.array(X_randomized)

def subpunctul_2():
    n = 1000
    val_lambda = [1, 2, 5, 10]
    X = []
    for _ in range(n):
        lam = np.random.choice(val_lambda)
        val = np.random.poisson(lam)
        X.append(val)
    return np.array(X)

print(subpunctul_2())

def subpunctul_a():
    x1, x2, x3, x4 = subpunctul_1()
    x_sub2 = subpunctul_2()

    plt.figure(figsize=(10, 6))

    plt.hist(x1, label='Poisson(1)')
    plt.hist(x2, label='Poisson(2)')
    plt.hist(x3, label='Poisson(5)')
    plt.hist(x4, label='Poisson(10)')
    plt.hist(x_sub2, label='lambda = {1, 2, 5, 10}')

    plt.xlabel("Nr de apeluri")
    plt.ylabel("Densitate empirica")
    plt.title("Cele 4 distributii cu parametrii fixi si random")
    plt.legend()
    plt.show()

print(subpunctul_a())

# def subpunctul_b():

    # Distributiile Poisson cu lambda fix sunt concentrate in jurul valorii lambda, iar
    # cu cat lambda este mai mare, histograma este mai intinsa.
    # Distributia randomizata este mai raspandita deoarece combina mai multe
    # distributii Poisson cu parametri diferiti.
