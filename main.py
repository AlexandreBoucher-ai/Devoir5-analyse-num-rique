import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4
import math

# c)
def f(t, y, dy):
    return 2*np.exp(-t)*np.sin(t)*y*dy - 2*np.exp(t)*np.sin(t)

coef = splines_edo_implicite(np.sqrt(2)/2 * np.exp(np.pi/4), np.sqrt(2) * np.exp(np.pi/4), f, np.pi/4, 3, 16)

# spline
def S(t):
    h = (3 - np.pi/4)/16
    i = min(int((t - np.pi/4)//h), 15)
    a, b, c, d = coef[i]
    return a*t**3 + b*t**2 + c*t + d

# on crée les intervalles de valeurs:
t = np.linspace(np.pi/4, 3, 16*20) #20 points en plus par intervalle pour la clarté
S_pasfonction = np.array([S(t) for t in t]) # on "sincronise" S et t 
y_exacte = np.exp(t)*np.sin(t)

plt.figure()
plt.plot(t, y_exacte, label = "solution exacte")
plt.plot(t, S_pasfonction, label = "Spline cubique")
plt.legend()
plt.title('Solutions exacte vs spline cubique')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()