# only fully abstract types should go here. Every thing with implementation goes one down in the hierarchy
import gym
import numpy as np
from abc import ABC, abstractmethod


class Env(gym.Env, ABC):
    goal_state = None
    goal_mask = None
    supports_rendering = False

    def __init__(self, *, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    @abstractmethod
    def cost_fn(self, state, action, next_state):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reward_fn(self, state, action, next_state):
        pass

    @abstractmethod
    def reset(self):
        pass


class Agent(ABC):
    needs_training = False
    needs_data = False
    has_state = False
    addition_solninfo = False
    required_settings = []

    # noinspection PyUnusedLocal
    def __init__(self, *, env: Env):
        self.env = env

    @abstractmethod
    def get_action(self, state, mode="train"):
        """Performs an action.
        :param obs: observation from environment
        :param state: some internal state from the environment that might be used
        :param mode: "train" or "eval" or "expert"
        """
        pass
