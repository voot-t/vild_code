# Source code of Variational Imitation Learning with Diverse-quality Demonstrations
Source code of ICML2020 paper "Variational Imitation Learning with Diverse-quality Demonstrations".
This repo includes pytorch code and datasets used for experiments in the paper. 

## Requirements
The code is tested on Python 3.6.9 with these python packages:
* pytorch == 1.3.1
* numpy == 1.14.0
* scipy == 1.0.1
* gym == 0.10.5
* mujoco-py == 1.50.1
* robosuite == 0.1.0

## How to run experiments
The main files are 
* code/vild_main.py - Script to run experiments. More details are provided below.
* code/args_parser.py - Script for parsing arguments. Default hyper-parameters can be found here.
