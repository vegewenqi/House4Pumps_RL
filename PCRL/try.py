import sys
import numpy as np
import casadi as csd
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


def phi_fun(P, ns, na, horizon, history):
    temp = ns * (ns + na)
    Phi1 = P[: temp * horizon * history].reshape((ns + na) * history, ns * horizon).T
    Phi2 = np.zeros((ns * horizon, na * horizon))
    c = 0
    for j in range(na * horizon):
        for i in range(ns * horizon):
            if j < na*(1 + (i // ns)):
                Phi2[i, j] = P[temp * horizon * history + c]
                c += 1
    Phi = np.concatenate((Phi1, Phi2), 1)
    return Phi

obs_dim, action_dim = 3, 2
history, horizon = 4, 3
nums = int(obs_dim * (obs_dim + action_dim) * horizon * history + obs_dim * action_dim * horizon * (horizon + 1) / 2)
P = np.zeros(nums)
for i in range(nums):
    P[i] = 0.5 * i
phi = phi_fun(P, obs_dim, action_dim, horizon, history)

plt.matshow(phi)
plt.show()