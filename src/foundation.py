import numpy as np
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])

def function(t):
    c = 4 # damping constant
    k = 2 # spring constant
    m = 20 # load mass
    # F = 0 # external force
    d = c / (2 * m)
    w0 = np.sqrt(k / m)

    # x(t) = exp^(-d * t) * (2A * cos(phi + w * t))
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1 / (2 * np.cos(phi))
    x = (np.exp(-d * t)) * (2 * A * np.cos(phi + w * t))
    return x

# DATA PROCESSING
# generate full dataset
t_full = np.arange(0, 61, 0.1)
x_full = function(t_full)
# Randomly extract 25 points from the function until t = 20
t_train = np.arange(0, 21, 0.1)
random_indices = np.random.choice(len(t_train), 25, replace=False)
t_train = t_full[random_indices]
x_train = x_full[random_indices]

# NEURAL NETWORK

# PHYSICS INFORMED NEURAL NETWORK

# VISUALS