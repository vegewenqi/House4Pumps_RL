import random
import numpy as np
from collections import deque


class BasicBuffer:
    """
    Simple buffer class to store all experiences

    ...

    Attributes
    ----------
    buffer : list of tuples
        Experiences stored as a list of tuples
    """

    def __init__(self):
        self.buffer = []

    def push(self, state, obs, action, reward, next_state, next_obs, done, info):
        experience = (
            state,
            obs,
            action,
            np.array([reward]),
            next_state,
            next_obs,
            done,
            info,
        )
        self.buffer.append(experience)

    def sample(self):
        transition = random.sample(self.buffer, 1)
        state, obs, action, reward, next_state, next_obs, done, info = transition
        return (state, obs, action, reward, next_state, next_obs, done, info)


class ReplayBuffer:
    """
    Replay buffer class, enabling random batch sampling

    ...

    Attributes
    ----------
    buffer : deque object
        Experiences stored as a list of rollouts
        Each rollout has its own list of experiences

    size : int
        Current size of the buffer

    max_size : int
        Maximum size of the replay buffer

    """

    def __init__(self, max_size, seed):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size_list = deque(maxlen=max_size)
        self.size = 0
        self.rng1 = np.random.default_rng(seed)  # Not thread safe
        self.rng2 = random.Random(seed)

    def push(self, rollout):
        """
        Add rollout to replay buffer

        Parameters
        ----------
        rollout :
            Data from a full rollout

        """
        self.size_list.append(len(rollout))
        self.size = sum(self.size_list)
        self.buffer.append(rollout)

    def update_buffer(self, buffer):
        buffer = (
            buffer if isinstance(buffer, deque) else deque(buffer, maxlen=len(buffer))
        )
        self.buffer = buffer
        self.size_list.append(len(buffer))
        self.size = len(buffer)

    def sample(self, batch_size):
        """
        Sample batch_size number of experiences from all the available ones

        Parameters
        ----------
        batch_size : int
            Number of experiences required to be sampled

        Returns
        -------
        list of tuples, each a randomly sampled experience

        """
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        for _ in range(batch_size):
            rollout = self.rng2.sample(self.buffer, 1)[0]
            (
                state,
                obs,
                action,
                reward,
                next_state,
                next_obs,
                done,
                info,
            ) = self.rng2.sample(rollout, 1)[0]
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)
        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def sample_sequence(self, sample_len):
        """
        Sample a sequence of experiences with length = sample_len
        One rollout is randomly chosen. If length of this rollout < sample_len,
        then the returned sequence is also smaller

        Parameters
        ----------
        sample_len : int
            Length of the sequence to be sampled

        Returns
        -------
        tuple with
        x_batch : x is the appropriate variable containing all samples

        """
        # batch_size is taken to be the size of each episode
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        rollout = self.buffer[self.rng1.integers(0, len(self.buffer))]
        if len(rollout) >= sample_len:
            start = self.rng1.integers(0, len(rollout) - sample_len + 1)
            rollout_sample = rollout[start:start + sample_len]
        else:
            rollout_sample = self.buffer[self.rng1.integers(0, len(self.buffer))]

        for transition in rollout_sample:
            (
                state,
                obs,
                action,
                reward,
                next_state,
                next_obs,
                done,
                info,
            ) = transition
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def last_rollout(self):
        """
        Sample the latest rollout in the replay buffer.

        Returns
        -------
        tuple with
        x_batch : x is the appropriate variable containing all samples

        """
        # batch_size is taken to be the size of each episode
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        rollout_sample = self.buffer[-1]

        for transition in rollout_sample:
            state, obs, action, reward, next_state, next_obs, done, info = transition
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def flatten_buffer(self):
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        for i in range(len(self.buffer)):
            for j in range(len(self.buffer[i])):
                (
                    state,
                    obs,
                    action,
                    reward,
                    next_state,
                    next_obs,
                    done,
                    info,
                ) = self.buffer[i][j]
                state_batch.append(state)
                obs_batch.append(obs)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                next_obs_batch.append(next_obs)
                done_batch.append(done)
                info_batch.append(info)
        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def __len__(self):
        return len(self.buffer)
