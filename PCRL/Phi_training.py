import sys
sys.path.append("/home/savvyfox/Projects/Safe-RL/Project")
import numpy as np
import casadi as csd
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer


def phi_fun(P, ns, na, horizon, history):
    temp = ns * (ns + na)
    Phi1 = P[: temp * horizon * history].reshape((ns + na) * history, ns * horizon).T
    Phi2 = torch.zeros(ns * horizon, na * horizon)
    c = 0
    for j in range(na * horizon):
        for i in range(ns * horizon):
            if j < na*(1 + (i // ns)):
                Phi2[i, j] = P[temp * horizon * history + c]
                c += 1
    Phi = torch.cat((Phi1, Phi2), 1)
    return Phi


def model_dyn(Acts, Hists, Phi):
    Z = torch.cat((Hists, Acts), 1)
    y = torch.matmul(Z, Phi.T)
    return y


## Main run
obs_dim, action_dim = 4, 2
history, horizon = 25, 15
seed = 1
data_path = "./Results/point_mass/ddpg_soln"

# Phi init
p_dyn = torch.rand(
    int(
        obs_dim * (obs_dim + action_dim) * horizon * history
        + obs_dim * action_dim * horizon * (horizon + 1) / 2
    )
).requires_grad_()
phi_dyn = phi_fun(p_dyn, obs_dim, action_dim, horizon, history)
phi_optimizer = optim.Adam([p_dyn], lr=0.1)

# data loading
data_buffer = ReplayBuffer(1, seed)
with open(data_path + "/replay_buffer.pkl", "rb") as fp:
    data_buffer.update_buffer(pickle.load(fp))
# self.data_buffer.update_buffer(list(self.data_buffer.buffer)[0:100])
print(len(data_buffer.buffer))

# training
batch_size = 64
train_it = 500
# print([batch_size, train_it])

for i in range(train_it):
    print(i)
    hist_batch = []
    u_batch = []
    y_batch = []
    for _ in range(batch_size):
        _, obss, actions, _, _, next_obss, _, _ = data_buffer.sample_sequence(
            history + horizon
        )
        hist_batch.append(
            np.concatenate(
                (
                    np.array(actions[:history]).reshape(-1),
                    np.array(obss[:history]).reshape(-1),
                )
            )
        )
        u_batch.append(np.array(actions[history:]).reshape(-1))
        y_batch.append(np.array(obss[history:]).reshape(-1))
    hist_batch = torch.FloatTensor(hist_batch)
    u_batch = torch.FloatTensor(u_batch)
    y_batch = torch.FloatTensor(y_batch)

    phi_dyn = phi_fun(p_dyn, obs_dim, action_dim, horizon, history)
    y_hat = model_dyn(u_batch, hist_batch, phi_dyn)
    phi_loss = F.mse_loss(y_hat, y_batch)

    phi_optimizer.zero_grad()
    phi_loss.backward()
    phi_optimizer.step()
    print(phi_loss)