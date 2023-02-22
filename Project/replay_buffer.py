import random
import numpy as np
from collections import deque


class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done, info):
        experience = (state, action, np.array([reward]), next_state, done, info)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        info_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done, info = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            info_batch.append(info)
        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            info_batch,
        )

    def sample_sequence(self, batch_size):
        # batch_size is taken to be the size of each episode
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        info_batch = []

        max_episodes = len(self.buffer) // batch_size
        start = np.random.randint(0, max_episodes)

        for sample in range(start * batch_size, (start + 1) * batch_size):
            state, action, reward, next_state, done, info = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            info_batch,
        )

    def __len__(self):
        return len(self.buffer)
