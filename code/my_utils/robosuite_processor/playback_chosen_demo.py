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
import time

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import IKWrapper

import pathlib 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, default=21, help='Id of of the environment to run')
    parser.add_argument('--demo_list', nargs='+', type=int, default=[ None ], help='list of demo to play')
    parser.add_argument('--render', type=int, default=0, help='render')
    parser.add_argument('--robo_task', action="store", default="reach", choices=["reach", "grasp", "full"], help='task')   

    args = parser.parse_args()
    
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

    do_render = args.render 

    env = robosuite.make(
        env_name,
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=do_render,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=False,
        reward_shaping=1,
        control_freq=100,
    )
    # env = IKWrapper(env)
    env = GymWrapper(env)

    # if args.demo_list[0] is None:
    #     args.demo_list = range(0, len(demos))

    # start = 700 
    # if args.env_id == 24:
    #     start = 630 
    # sort_list = np.arange(start, start + 20)

    # ## load from sorted list 
    # if args.demo_list[0] is None:
    #     args.demo_list = []
    #     filename = "%s/Documents/Git/imitation_data/TRAJ_robo/%s/%s_sort.txt" % (pathlib.Path.home(), env_name, env_name)
    #     # filename = "%s/Documents/Git/imitation_data/TRAJ_robo/%s/%s_chosen.txt" % (pathlib.Path.home(), env_name, env_name)
        
    #     demo_idx = -1
    #     with open(filename, 'r') as ff:
    #         i = 0
    #         for line in ff:

    #             i = i + 1
    #             line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
    #             if demo_idx == -1:
    #                 demo_idx = line.index("demo") + 1


    #             if np.any(sort_list == i):
    #                 args.demo_list += [int(line[demo_idx])]
    #     print(args.demo_list)

    # # list of all demonstrations episodes
    # demos = list(f["data"].keys())

    demo_list = []
    step_list = []
    if args.robo_task != "full":
        filename = "%s/Documents/Git/imitation_data/TRAJ_robo/%s_%s/%s_%s_chosen.txt" % (pathlib.Path.home(), env_name, args.robo_task, env_name, args.robo_task)
    else:
        filename = "%s/Documents/Git/imitation_data/TRAJ_robo/%s/%s_chosen.txt" % (pathlib.Path.home(), env_name, env_name)
    demo_idx = -1
    step_idx = -1
    with open(filename, 'r') as ff:
        i = 0
        for line in ff:

            i = i + 1
            line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
            if demo_idx == -1:
                demo_idx = line.index("demo") + 1
                step_idx = line.index("step") + 1
            demo_list += [int(line[demo_idx])]
            step_list += [int(line[step_idx])]
    print(demo_list)

    demos = list(f["data"].keys())

    total_step = 0
    
    ep_count = 0
    for ep_count in range(0, len(demo_list)):
        
        if ep_count > 9:
            break   # use only 10 demonstrations in experiments. 

        ep_i = demo_list[ep_count]
        print("Playing back chosen episode... (press ESC to quit)")

        # ep =  demos[ep_i]
        ep = "demo_%d" % ep_i 

        print("Epi %d" % (ep_i))

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["data/{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)

        if do_render:
            env.viewer.set_camera(0)

        print(ep) 

        time.sleep(15)

        states = f["data/{}/states".format(ep)].value
                    
        count = 0
        ret = 0
        contact = False 
        # force the sequence of internal mujoco states one by one
        for step in range(0, step_list[ep_count]):
            state = states[step]
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            ## Because a longer demonstration gets higher rewards even though it is worse, we add -1 as cost in each time-step.
            # done = 0
            # reward = 0

            # if args.robo_task == "full":
            #     reward = env.reward() 

            # elif args.robo_task == "reach":
            #     r_reach, _, _, _ = env.staged_rewards()

            #     reward = r_reach * 10

                # if env._check_contact(): 
                #     ## If contact, then reward > 0.028 
                #     done = 1 

            # elif args.robo_task == "grasp":
            #     _, r_grasp, _, _ = env.staged_rewards()   ## remove this part when render because it makes rendering slow.

            #     reward = r_grasp    
            #     if r_grasp > 0:
            #         done = 1

            # ret = ret + reward 

            # if done:
            #     break 

            if do_render:
                pic = env.render()
                time.sleep(0.005)

                print(pic) 

                # time.sleep(1)

            count = count + 1

        ret = ret # / len(states) #average reward along traj.
        total_step += count 
        print("Return: %f, step: %d" % (ret, count))
    print("Total steps: %d" % total_step)

    f.close()
