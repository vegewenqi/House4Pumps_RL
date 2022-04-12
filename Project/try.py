import numpy as np
from gym.spaces.box import Box
import casadi as ca
import math
import matplotlib.pyplot as plt
from Project.Environments.smart_home import SmartHome
import json

''''Gaussian plot'''
# u = 0  # 均值μ
# sig = math.sqrt(9)  # 标准差δ
# x = np.linspace(u - 6*sig, u + 6*sig, 50)   # 定义域
# y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
# plt.plot(x, y, "g", linewidth=2)    # 加载曲线
# plt.grid(True)  # 网格线
# plt.show()  # 显示

''''Plot'''
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.axis([0, 10, 0, 1])
#
# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.5)

# plt.show()

# plt.figure(figsize=(9, 9)) #设定图像尺寸
# x = range(100000) #画散点图x与y要size相同
# new_data = np.random.rand(9, 100000)
# for i in range(9):
#     plt.subplot(3, 3, i + 1) #做一个3*3的图 range（9）从0开始，因需要从1开始，所以i+1
#     plt.scatter(x, new_data[i, :], color = 'black')
#     plt.axis = 'off' #关闭坐标 让图更美观
# plt.show()


''''Save,read and plot data'''
# import os
# import pandas as pd
#
# state = np.zeros((3, 5))
# action = np.ones((2, 5))
# reward = np.array([1, 2, 3, 4, 5]).reshape(1, 5)
# data_set = np.concatenate((state, action, reward))
# df = pd.DataFrame(data=data_set)
# print(df)
# df.to_csv(r'C:\Users\Administrator\Dropbox (KAUST)\ENMPC_RL'
#           r'\Safe-RL\Project\Results\result.csv', index=False, header=False)
#
# data_path = r'C:\Users\Administrator\Dropbox (KAUST)\ENMPC_RL\Safe-RL\Project\Results\result.csv'
# data = pd.read_csv(data_path, header=None)
# # state = np.array([data["state"]])
# # print(pd.to_numpy(data["state"]))
# a = data.values
# print(a)

#
# x = ca.SX.sym('x')  # x轴状态
# y = ca.SX.sym('y')  # y轴状态
# theta = ca.SX.sym('theta')  # z轴转角　
# states1 = ca.vertcat(*[x, y, theta])
# states2 = ca.vcat([x, y, theta])
# n_states1 = states1.size()
# n_states2 = states2.size()[0]
# print(states1, states2)
# print(n_states1, n_states2)

import casadi as csd
Q = csd.SX.sym("theta_Q", 2, 2)
a = Q[:]
print(Q)
print(a)