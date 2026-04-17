import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4
import math

# c)
def f(t, y, dy):
    return 2*np.exp(-t)*np.sin(t)*y*dy - 2*np.exp(t)*np.sin(t)

coef = splines_edo_implicite(np.sqrt(2)/2 * np.exp(np.pi/4), np.sqrt(2) * np.exp(np.pi/4), f, np.pi/4, 3, 16)

