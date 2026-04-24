import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4
import math

# c)
def f(t, y, dy):
    return 2*np.exp(-t)/np.sin(t)*y*dy - 2*np.exp(t)*np.sin(t)

coef = splines_edo_implicite(np.sqrt(2)/2 * np.exp(np.pi/4), np.sqrt(2) * np.exp(np.pi/4), f, np.pi/4, 3, 16)


# spline
def S(t):
    h = (3 - np.pi/4)/16
    i = min(int((t - np.pi/4)//h), 15)
    a, b, c, d = coef[i]
    return a*t**3 + b*t**2 + c*t + d

# on crée les intervalles de valeurs:
t_int = np.linspace(np.pi/4, 3, 16*20) #20 points en plus par intervalle pour la clarté
S_pasfonction = np.array([S(t) for t in t_int]) # on "sincronise" S et t 
y_exacte = np.exp(t_int)*np.sin(t_int)

plt.figure()
plt.plot(t_int, y_exacte, label = "solution exacte")
plt.plot(t_int, S_pasfonction, label = "Spline cubique")
plt.legend()
plt.title('Solutions exacte vs spline cubique')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()

# d)
# Solution exact
def y(t):
    return np.exp(t) * np.sin(t)

N = [2**6, 2**7, 2**8, 2**9, 2**10] 

erreur = []
h = []

for i in N:
    coef = splines_edo_implicite(np.sqrt(2)/2 * np.exp(np.pi/4), np.sqrt(2) * np.exp(np.pi/4), f, np.pi/4, 3, i)
    hi = (3 - np.pi/4)/i
    h.append(hi)
    t_int = np.linspace(np.pi/4, 3, i+1)
    erreur_max = 0

    for j in range(i):
        ti = t_int[j]
        ai, bi, ci, di = coef[j]
        S_ti = ai*ti**3 + bi*ti**2 + ci*ti + di
        erre = abs(y(ti) - S_ti)

        if erre > erreur_max:
            erreur_max = erre
    tN = t_int[-1]
    aN, bN, cN, dN = coef[-1]
    S_tN = aN*tN**3 + bN*tN**2 + cN*tN + dN
    erreur_max = max(erreur_max, abs(y(tN) - S_tN))

    erreur.append(erreur_max)

plt.figure()
plt.loglog(h, erreur)
plt.title('Erreur global en fonction de h (en log)')
plt.xlabel("h")
plt.ylabel("E(h)")
plt.show()