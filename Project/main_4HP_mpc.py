import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import os
from Environments import env_init
from Agents import agent_init
from replay_buffer import BasicBuffer
from helpers import tqdm_context
from rollout_utils import rollout_sample, train_controller
import pickle

# cmd_line = sys.argv
# with open(cmd_line[1]) as f:
#     params = json.load(f)
#     print(params)

json_path = os.path.abspath(os.path.join(os.getcwd(), '../Settings/other/house_4pumps_rl_mpc_lstd.json'))
with open(json_path, 'r') as f:
    params = json.load(f)
    params["env_params"]["json_path"] = json_path
    print(f'env_params = {params}')

### Environment
env = env_init(params["env"], params["env_params"])
params["n_steps"] = len(env.config['dt'])   # extra line for house 4pumps project

### Agent
agent = agent_init(env, params["agent"], params["agent_params"])

### Reply buffer
replay_buffer = BasicBuffer(params["buffer_maxlen"])

### mode=mpc
rollout_return = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="mpc")

### save results
Results = {'buffer': replay_buffer,
           'input': env.InputDict,
           'config': env.config}
f = open('Results/House4Pumps/Results.pkl', "wb")
pickle.dump(Results, f)
f.close()
print('Results saved successfully！')



