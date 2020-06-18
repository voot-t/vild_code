"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""
import os
import h5py
import argparse
import random
import numpy as np

import pickle
import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import IKWrapper

import pathlib 
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, default=21, help='Id of of the environment to run')
    args = parser.parse_args()
    
    env_dict = {
                ## Robosuite
                21 : "SawyerNutAssemblyRound",
                22 : "SawyerNutAssemblySquare",
                23 : "SawyerNutAssembly",
                24 : "SawyerPickPlaceBread",
                25 : "SawyerPickPlaceCan",
                26 : "SawyerPickPlaceCereal",
                27 : "SawyerPickPlaceMilk",
                28 : "SawyerPickPlace",
    }
    # env_name = env_dict[args.env_id]
    
    
    demo_dict = {
                ## Robosuite
                21 : "pegs-RoundNut",   #
                22 : "pegs-SquareNut",  #
                23 : "pegs-full",
                24 : "bins-Bread",      #
                25 : "bins-Can",        # 
                26 : "bins-Cereal",
                27 : "bins-Milk",        # 
                28 : "bins-full",
    }
    demo_name = demo_dict[args.env_id]
    
    demo_path = "%s/Documents/Git/imitation_data/RoboTurkPilot/%s" % (pathlib.Path.home(), demo_name)
    
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]

    pathlib.Path("%s/Documents/Git/imitation_data/TRAJ_robo/%s" % (pathlib.Path.home(), env_name) ).mkdir(parents=True, exist_ok=True) 
    
    do_render = False 
    use_shape = 1

    env = robosuite.make(
        env_name,
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=do_render,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=False,
        reward_shaping=use_shape,
        control_freq=100,
    )

    # env = IKWrapper(env)
    env = GymWrapper(env)

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    """ we will save 1 file for 1 demo trajectory. """
    for ep_i in range(1, len(demos) + 1):
    
        # ep = demos[ep_i]  ## The demos list variable does not sort 1, 2, 3, ... Instead it sorts 1, 10, 100, ...
        ep = "demo_%d" % ep_i 

        model_file = f["data/{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        ## reset model for each episode
        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
    
        if do_render:
            env.viewer.set_camera(0)
    
        states = f["data/{}/states".format(ep)].value
        joint_velocities = f["data/{}/joint_velocities".format(ep)].value
        gripper_actuations = f["data/{}/gripper_actuations".format(ep)].value
                    
        expert_traj = []
        expert_mask = []
        expert_reward = []

        expert_traj_list = []
        expert_mask_list = []
        expert_reward_list = []

        num_steps = 0
        reward_episode = 0
        
        for t in range (0, len(states)):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()

            ## These lines get obs and action.
            obs = env._flatten_obs(env._get_observation())     # This is the current state, not next state.
            action = np.concatenate((joint_velocities[t], gripper_actuations[t]))
            reward = env.reward() 
            done = (t == (len(states) - 1))
            
            
            reward_episode += reward
            num_steps += 1
            
            mask = 0 if done else 1   # done=1 only at the last step.
            
        
            expert_traj.append(np.hstack([obs, action]))
            expert_mask.append(mask)
            expert_reward.append(reward)

            
            if do_render:
                env.render()
    
            # if done:  
            #     expert_traj_list += [expert_traj]
            #     expert_mask_list += [expert_mask]
            #     expert_reward_list += [expert_reward]
            #     break
    
        expert_traj_list += [expert_traj]
        expert_mask_list += [expert_mask]
        expert_reward_list += [expert_reward]

        
        """ save data """        
        traj_filename = "%s/Documents/Git/imitation_data/TRAJ_robo/%s/%s_demo%d.p" % (pathlib.Path.home(), env_name, env_name, ep_i)

        print("Demo %4d: total steps %5d, return: %f" % ( ep_i, num_steps, reward_episode ) )

        # pickle.dump( (expert_traj_list, expert_mask_list), open(traj_filename, "wb") )
        pickle.dump( (expert_traj_list, expert_mask_list, expert_reward_list), open(traj_filename, "wb") )
        # print("TRAJ result is saved at %s" % traj_filename)

        
        
        
        