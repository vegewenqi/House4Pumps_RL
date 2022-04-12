import numpy as np

from helpers import tqdm_context
from Agents.abstract_agent import TrainableController


def rollout_sample(env, agent, replay_buffer, n_step, mode="train"):
    state = env.reset()
    # once execute rollout_sample，run env.reset(), reset state and t
    rollout_return = 0
    rendering = True if mode == "final" and env.supports_rendering else False
    if rendering:
        env.prepare_for_render()

    # get action, env.step, buffer.push
    for step in tqdm_context(range(n_step), desc="Episode", pos=1):
        print(f'step {step}-------------')
        action, add_info = agent.get_action(state, mode=mode)
        next_state, reward, done, power_info = env.step(action)
        add_info['power_info'] = power_info  # extra power info for the house4pumps project

        if mode == "train":
            replay_buffer.push(state, action, reward, next_state, done, add_info)
        elif mode == "mpc":
            ### pickle can't save DMStruct, convert to DM using .cat!!! save for plot when mode="mpc"
            replay_buffer.push(state.cat, action.cat, reward, next_state.cat, done, add_info)

            ### debug
            # print(f'state = {state.cat}, \n'
            #       f'action = {action.cat}, \n'
            #       f'reward = {reward}, \n'
            #       f'next_state = {next_state.cat}')

        rollout_return += reward

        if rendering:
            env.render()
            print(state, next_state, action, reward, done)

        # state = next_state.copy()
        state = next_state

    if rendering:
        env.end_render()

    return rollout_return


# execute agent.train:
#  run replay_buffer.sample function:
#     sample batch_size data
#     then use those data to calculate w,v, and then theta
#     update self.critic_wt, self.adv_wt, self.actor.actor_wt
def train_controller(agent, replay_buffer):
    if isinstance(agent, TrainableController):
        # train_it = min(
        #     agent.iterations, int(5 * len(replay_buffer.buffer) / agent.batch_size)
        # )
        train_it = agent.iterations
        if len(replay_buffer) > agent.batch_size:
            agent.train(replay_buffer, train_it)


# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in many_episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in many_episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in many_episode_rewards]

    return minimum, maximum, mean