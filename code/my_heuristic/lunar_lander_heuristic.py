## Script to generate demos  from gym's lunar lander heuristic policy in gym/envs/box2d/lunar_lander.py 
import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

import h5py 
import random 
import pathlib 
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"
from itertools import count 

## from https://github.com/floodsung/DDPG-tensorflow/blob/master/ou_noise.py. 
class OU(object):
    def __init__(self, action_dim=1, a_low=1, a_high=1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.01, decay_period=1000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim 
        self.low          = a_low
        self.high         = a_high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim,) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_noise(self, t=0): 
        noise_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)   #OU. Noise decreasing
        return noise_state.squeeze() 

## The original heuristic from gym's lunarlander code. 
def heuristic(env, s, noise_1, noise_2):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    angle_gain_p = 0.5 
    angle_gain_d = 1.0 
    hover_gain_p = 0.5 
    hover_gain_d = 0.5

    """ To simulate noise, we perturb P/D gains. """            
    noise_1 = -np.abs(noise_1)  
    noise_2 = -np.abs(noise_2)  

    angle_gain_p = np.clip(angle_gain_p + noise_1, a_min = 0, a_max=None)
    angle_gain_d = np.clip(angle_gain_d + noise_2, a_min = 0, a_max=None)

    # hover_gain_p = np.clip(hover_gain_p + noise_1, a_min = 0, a_max=None)
    # hover_gain_d = np.clip(hover_gain_d + noise_2, a_min = 0, a_max=None)

    """ """

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*angle_gain_p - (s[5])*angle_gain_d
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*hover_gain_p - (s[3])*hover_gain_d
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
    a = np.clip(a, -1, +1)

    return a

def test_noise():
    #env = LunarLander()
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)    
    env.seed(1)
    np.random.seed(1) 
    max_step = 1000

    for i in count():
        s = env.reset()
        noise_level = 1
    
        noise_model_1 = OU(1, 1, 1, mu=0.0, theta=0.15, max_sigma=noise_level, min_sigma=0.01, decay_period=500)
        noise_model_2 = OU(1, 1, 1, mu=0.0, theta=0.15, max_sigma=noise_level, min_sigma=0.01, decay_period=500)

        noise_1 = 0 
        noise_2 = 0 

        total_reward = 0
        for t in range(0, max_step):
            noise_1 = noise_model_1.get_noise(t) 
            noise_2 = noise_1 
            # noise_2 = noise_model_2.get_noise(t) 

            # if t % 20 == 0:
            #     print(noise_2) 

            a = heuristic(env, s, noise_1, noise_2 )

            s, r, done, info = env.step(a)
            env.render()
            total_reward += r
            # total_reward += -1  # penalize long trajectory 
            # if r == -100: total_reward += -100

            if t+1 == max_step:
                done = 1
            if done:
                # print(["{:+0.2f}".format(x) for x in s])
                print("step {} total_reward {:+0.2f}".format(t, total_reward))
                break

def save_demo():
    #env = LunarLander()
    env_name = "LunarLanderContinuous-v2"
    env =  gym.make(env_name)    
    max_step = 1000

    noise_level_list = [0.01, 0.05, 0.1, 0.25, 0.4,      0.6, 0.7, 0.8, 0.9, 1.0]
    demo_file_size = 2000   # 2000 sa pair per noise level. 

    # noise_level_list = [0.00]
    # demo_file_size = 10000

    """ Set path for trajectory files """
    traj_path = "../../imitation_data/TRAJ_h5/%s/" % (env_name)
    pathlib.Path(traj_path).mkdir(parents=True, exist_ok=True) 
    print("%s trajectory will be saved at %s" % (colored("heuristic", p_color), colored(traj_path, p_color)))
        
    for noise_level in noise_level_list:
        env.seed(1)
        np.random.seed(1) 
        random.seed(1)

        expert_state_list = []     
        expert_action_list = []     
        expert_mask_list = []
        expert_reward_list = []
        total_step, avg_reward_episode = 0, 0
        
        for i_episode in count():
            s = env.reset()
        
            noise_model_1 = OU(1, 1, 1, mu=0.0, theta=0.15, max_sigma=noise_level, min_sigma=0.001, decay_period=500)
            noise_model_2 = OU(1, 1, 1, mu=0.0, theta=0.15, max_sigma=noise_level, min_sigma=0.001, decay_period=500)

            noise_1 = 0
            noise_2 = 0

            total_reward = 0
            for t in range(0, max_step):
                noise_1 = noise_model_1.get_noise(t) 
                noise_2 = noise_model_2.get_noise(t) 

                a = heuristic(env, s, noise_1, noise_2)
                s_next, r, done, info = env.step(a)
                
                total_reward += r

                s = s_next 
                total_step += 1

                # env.render()
                if t + 1 == max_step:
                    done = 1

                expert_state_list.append(s)
                expert_action_list.append(a)
                expert_mask_list.append(int(not done))
                expert_reward_list.append(r)

                if done:
                    # print(["{:+0.2f}".format(x) for x in s])
                    print("step {} total_reward {:+0.2f}".format(t, total_reward))
                    break 

            avg_reward_episode += total_reward

            if total_step >= demo_file_size:
                break

        expert_states = np.array(expert_state_list)
        expert_actions = np.array(expert_action_list)
        expert_masks = np.array(expert_mask_list)
        expert_rewards = np.array(expert_reward_list)

        print("Total steps %d, total episode %d, AVG reward: %f" % ( total_step, i_episode + 1, avg_reward_episode/(i_episode+1)))

        h_name = "heuristic"

        traj_filename = traj_path + ("/%s_TRAJ-N%d_%s%0.2f" % (env_name, demo_file_size, h_name, noise_level))
            
        print(expert_actions.min()) 
        print(expert_actions.max()) 

        hf = h5py.File(traj_filename + ".h5", 'w')
        hf.create_dataset('expert_source_path', data=h_name)    # 
        hf.create_dataset('expert_states', data=expert_states)
        hf.create_dataset('expert_actions', data=expert_actions)
        hf.create_dataset('expert_masks', data=expert_masks)
        hf.create_dataset('expert_rewards', data=expert_rewards)

        print("TRAJ result is saved at %s" % traj_filename)

if __name__=="__main__":
    # test_noise()
    save_demo() 