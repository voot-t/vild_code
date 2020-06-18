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
import pickle 

import pathlib 
import torch 

# from get_config import *
# from args_parser import arg_parser
    
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
print_color = "yellow"

def sort():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, default=21, help='Id of of the environment to run')
    parser.add_argument('--robo_task', action="store", default="reach", choices=["reach", "grasp", "full"], help='task')   
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
    args.env_name = env_dict[args.env_id]
    
    traj_name = "traj_roboturk"
    demo_path = "%s/Documents/Git/imitation_data/TRAJ_robo/%s" % (pathlib.Path.home(), args.env_name)
    if args.robo_task != "full":
        demo_path += "_%s" % args.robo_task 
    
    real_traj_tensor_list = []  # legnth equal to num_worker
    real_mask_tensor_list = []  # legnth equal to num_worker
    real_time_tensor_list = []  # legnth equal to num_worker
    real_worker_tensor_list = []  ## legnth equal to num_worker  indices to worker ID, i.e., i_noise

    return_list = []
    traj_len_list = []

    total_size = 0  ## number of (s,a) pairs 

    total_traj = len(os.listdir(demo_path)) - 1
    for demo_i in range(1, total_traj + 1):
        if args.robo_task != "full":
            traj_filename = demo_path + ("/%s_%s_demo%d.p" % (args.env_name, args.robo_task, demo_i))
        else:
            traj_filename = demo_path + ("/%s_demo%d.p" % (args.env_name, demo_i))

        real_traj_list, real_mask_list, real_reward_list = pickle.load(open(traj_filename, "rb"))
        ## The loaded list is actually a 2 dimensional lsit (list of list) which contains trajecctories of the same demonstrator.
        ## For roboturk dataset, (we assume) 1 demonstrator collect 1 trajectory, so  the first dim is size 1.
        ## So we have [0] indexing below 

        traj_len = len(real_mask_list[0]) 
        total_size += traj_len 

        real_traj_tensor_list += [ torch.FloatTensor(real_traj_list[0]) ]    # real_traj_list is a list of state-action pairs
        real_mask_tensor_list += [ torch.FloatTensor(real_mask_list[0]) ]
        real_time_tensor_list += [ torch.FloatTensor( np.arange(1, traj_len+1) ) ]
        real_worker_tensor_list += [ torch.LongTensor(traj_len).fill_(demo_i) ]

        return_list += [np.sum(np.asarray( real_reward_list[0] ))]  ## these were computed with shaped reward

        traj_len_list += [traj_len] 


    traj_len_list = np.asarray(traj_len_list)

    rwd_ratio = return_list / traj_len_list 
    # sort_index = np.argsort(rwd_ratio)[::-1]

    sort_index = np.argsort(traj_len_list) 
    # filename = demo_path + ('../../%s/%s_sort.txt' % (args.env_name, args.env_name))
    
    if args.robo_task != "full":
        filename = demo_path + ('/%s_%s_sort.txt' % (args.env_name, args.robo_task))
    else:
        filename = demo_path + ('/%s_sort.txt' % (args.env_name))
    
    open(filename, 'w').close()
    with open(filename, 'a') as f:
        for i in range(0, total_traj):
            result_text = "demo %4d, step %4d, return %f, return_ratio %f" % (sort_index[i] + 1, traj_len_list[sort_index[i]], return_list[sort_index[i]], rwd_ratio[sort_index[i]]) 
            print(result_text, file=f) 

if __name__ == "__main__":
    sort()

