# 1.1

x = x_min = x_max = int(input("Entrez un entier : "))

while x != 0:
    if x > x_max:
        x_max = x
    elif x < x_min:
        x_min = x
    x = int(input("Entrez un entier : "))
    
print(f"Min: {x_min}\nMax: {x_max}")

# 1.21

x = int(input("Entrez un entier : "))

print([i for i in range(1, x) if x % i == 0])

# 1.3

a, b = int(input("Entier : ")), int(input("Diviseur : "))

q = 0
while a >= b:
    a -= b
    q += 1

print(f"Quotient : {q}\nReste : {a}")

# 1.4

m, n = int(input("Entier 1 : ")), int(input("Entier 2 : "))

def produit_etrange(m, n):
    if n % 2 != 0:
        return produit_etrange(m, n-1) + m
    else:
        if n > 0:
            return produit_etrange(m+m, n/2)
    return 0

print(f"Produit étrange : {produit_etrange(m, n)}")

# 1.5.b

def sum_to(n):
    x = 0
    for i in range(n+1):
        x += i
    return x

def sum_to_recursive(n):
    if n != 0:
        return sum_to_recursive(n-1) + n
    return n

# 1.5.c

def fact(n):
    x = 1
    for i in range(1, n+1):
        x *= i
    return x

def fact_recursive(n):
    if n != 0:
        return fact_recursive(n-1) * n
    return 1

# 1.5.d

def lister(n):
    if n != 1:
        lister(n-1)
        print(n)
        return
    print(1)

# 1.5.e

def lister_reverse(n):
    if n != 1:
        print(n)
        lister(n-1)
        return
    print(1)

# 1.6.a

def sigma(n):
    return sum([1/i**2 for i in range(1, n+1)])

def sigma_recursive(n):
    if n != 1:
        return sigma_recursive(n-1) + 1/n**2
    return 1

# 1.6.b

from math import pi

n = 0
while abs(sigma(n) - pi**2/6) > 10**(-3):
    n += 1

print(f"n = {n}") # n = 1000

# 2.1

def puissance_etrange(x, n):
    if n % 2: # Impair
        return x * puissance_etrange(x, n-1)
    if n != 0: # Pair et non nul
        return puissance_etrange(x*x, n/2)
    return 1 # Nul

# 2.1.a

x, n = int(input("x = ")), int(input("n = "))

print(f"x**n = {puissance_etrange(x, n)}")

# 2.1.b

import numpy as np

def puissance_etrange_matricielle(m, n):
    if n % 2: # Impair
        return np.dot(m, puissance_etrange_matricielle(m, n-1))
    if n != 0: # Pair et non nul
        return puissance_etrange_matricielle(np.dot(m, m), n/2)
    return np.identity(m.shape[0])

m = np.array(eval(input("Entrez la matrice carrée M sous forme de liste :\n")))
n = int(input("Entrez la puissance entière n à laquelle mettre cette matrice : "))

print(f"M**n :\n{puissance_etrange_matricielle(m, n)}")

# 3.1

def fibo(n):
    if n in (0, 1):
        return n
    return fibo(n-1) + fibo(n-2)
 
def fibo_iter(n):
    x0 = 0
    x1 = 1
    fibo = [x0, x1]
    for _ in range(n-1):
        x2 = x0 + x1
        fibo.append(x2)
        x0 = x1
        x1 = x2
    return fibo

# 3.2

import numpy as np
import numpy.linalg as la

def fibo_matrix(n):    
    f = np.matrix((
        (1,),
        (0,)
    ))
    m = np.matrix((
        (1, 1),
        (1, 0)
    ))

    m = la.matrix_power(m, n-1)
    return np.dot(m, f)[0][0]

# Complément au 3.2 : Visualisation de la complexité de l'algorithme

# Ce programme lance de multiples fois la fonction fibo_matrix précédente, en
# mesurant son temps d'exécution. Un graphique du temps d'éxécution en
# fonction du nombren d'itérations de la suite à calculer est ensuite affiché.

from matplotlib import pyplot as plt
from timeit import Timer
from functools import partial

def plot_complexity(function, rng=range(10**3), tests=1, scale=1):
    x = []
    y = []
    for i in rng:
        time = Timer(partial(function, i)).timeit(number=tests)
        x.append(i)
        y.append(time*scale)
    plt.plot(x, y, label=function.__name__)
    
def plot_log(rng):
    y = [np.log(i) for i in rng]
    plt.plot(rng, y, label="log(n)")

# Décommenter ce bloc pour calculer le graphique :

# plot_complexity(fibo_matrix, range(1, 10000, 20), tests=100, scale=650)
# plot_log(np.arange(1, 10000, 20))
# plt.show()

# Le résultat est visible ici :
# https://github.com/LeMinaw/TP1/raw/master/fibo_matrix_complexity.png
# 
# La trace bleue est issue des mesures du temps d'exécution, la trace orange
# correspond à une courble de logarithme naturel.
#
# On voit que la complexité du calcul est bien O(log n). En effet, la
# puissance matricielle de NumPy effectue autant que possible des puissances
# sucessives, réduisant drastiquement la complexité du calcul.
#
# Par exemple, pour mettre à la puissance 8, au lieu d'effectuer huit fois
# le produit matriciel (M**8 = M*M*M*M*M*M*M*M), NumPy calculera la matrice
# trois fois au carré (M**8 = ((M**2)**2)**2), n'effectuant que trois fois le
# produit matriciel.
#
# Ce phénomène apparaît dans le graphique sous forme de "dents de scie", puisque
# les puissances de nombres se décomposant mieux en puissances de deux sucessives
# sont plus rapides à calculer.

# 4.1

# m(0, s) est le gain maximum obtenu en plaçant une quantité s dans le
# "0-ième" entrepôt. Cela signifie qu'on ne place de stock nulle part,
# le gain est ainsi nul : m(0, s) = 0.

# 4.2

# Le gain maximal sur les k premiers entrepôts m(k, s) est égal au
# maximum de la somme du gain sur le k-ième entrepôt et du gain sur les
# k-1 premiers entrepôts.
# Comme le premier entrepôt est indicé 0, le k-ième entrepôt est d'indice
# k-1. Avec s' une partie du stock s, le gain sur le k-ième entrepôt est 
# donc g(k-1, s').
# Le stock restant pour les k-1 premiers entrepôts est alors s privé de
# ce qui a été déposé de le k-ième entrepôt, soit s-s'.
# Par définition, la répartition optimale sur les k-1 premiers entrepôts est
# alors m(k-1, s-s').
# On a alors m(k, s) = max(g(k-1, s') + m(k-1, s-s')), pour 0 <= s' <= s.