import numpy as np
import gym
from gym import spaces
import random

class NormalizeGymWrapper(gym.ActionWrapper):
    """
    This wrapper normalize action to be in [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)

        high = self.env.action_space.high 
        low = self.env.action_space.low 
        
        np.testing.assert_array_equal(high, -low)   ## check that the original action bound is symmetric. 

        self.action_scale = np.abs(high) 
        
        self.action_space = spaces.Box(low=low / self.action_scale, high=high / self.action_scale)

    def step(self, action):

        action = action * self.action_scale # re-scale back to the original bound

        ob, reward, done, info = self.env.step(action)
        return ob, reward, done, info

class ClipGymWrapper(gym.ActionWrapper):
    """
    This wrapper clip action to be in [low, high].
    Cliped actions are required to prevent errors in box2d envs (LunarLander and BipedalWalker)
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        return self.env.step(np.clip(action, a_min=self.env.action_space.low, a_max=self.env.action_space.high))
