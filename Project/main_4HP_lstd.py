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

### Returns
train_returns = []
eval_returns = []

for it in tqdm_context(range(params["n_iterations"]), desc="Iterations", pos=3):
    print(f"Iteration: {it}------")

    # [[Sampling]]
    # run rollout_sample function:
    #       run agent.get_action(), env.step(), and push the data to buffer
    #       rollout n_step
    #       return n_step return
    train_return = []
    for train_runs in tqdm_context(range(params["n_trains"]), desc="Train Rollouts"):
        rollout_return = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="train")
        # once execute rollout_sample，inside run env.reset()
        train_return.append(rollout_return)
    train_returns.append(mean(train_return))
    print(f"Training return: {mean(train_return)}")

    # [[Replay + Learning]]-policy and critics parameters update
    # run agent.train function:
    #   run replay_buffer.sample function:
    #       sample batch_size data
    #       then use those data to calculate w,v, and then theta
    #       update self.critic_wt, self.adv_wt, and self.actor.actor_wt
    train_controller(agent, replay_buffer)

    # # # [[Evaluation]]-evaluate the performance of the learned self.actor.actor_wt
    # # # run rollout_sample function:
    # # #       get action, interact with env, but no need to push the data to buffer, without action noise
    # # #       rollout n_evals
    # # #       return n_evals return
    eval_return = []
    if (it+1) % 10 == 0:
        for eval_runs in tqdm_context(range(params["n_evals"]), desc="Evaluation Rollouts"):
            rollout_return = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="eval")
        eval_return.append(rollout_return)
        eval_returns.append(mean(eval_return))
        print(f"Evaluation return: {mean(eval_return)}")


### Final rollout for visualization
# _ = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="final")
# # Evaluation performance plot
# perf = plt.figure("Evaluation Performance")
# plt.plot(eval_returns)
# plt.ylabel("return")
# plt.show()
# print(eval_returns)
# plt.pause(5)




