# Source code of Variational Imitation Learning with Diverse-quality Demonstrations
Source code of ICML2020 paper "Variational Imitation Learning with Diverse-quality Demonstrations".
This repo includes pytorch code and datasets used for experiments in the paper. 

## Requirements
Experiments were run with Python 3.6.9 and these packages:
* pytorch == 1.3.1
* numpy == 1.14.0
* scipy == 1.0.1
* gym == 0.10.5
* mujoco-py == 1.50.1
* robosuite == 0.1.0

## How to run experiments
Important files are 
* code/vild_main.py - Script to run experiments with RL-based IL methods. More details are provided below.
* code/bc_main.py - Script to run experiments with SL-based IL methods. 
* code/args_parser.py - Script for parsing arguments. Default hyper-parameters can be found here.
* code/core/vild.py - Script implementing VILD algorithm. The VILD class extends the IRL baseclass below.
* code/core/irl.py - Script implementing classes of IRL/GAIL baselines. 
* code/core/ac.py - Script implementing classes of RL algorithms (TRPO, PPO, SAC).

Experiments with RL-based IL methods is run via vild_main.py. To set algorithms, set argument --il_method *algorithm_name*.
*algorithm_name* can be as follows: vild, irl (This is maxent-irl), gail, airl, vail, infogail.
Without setting --il_method, the default behavior of the code is to run TRPO to perform RL.

To use log-sigmoid reward function for VILD, set argument --vild_loss_type BCE

