from my_utils import *

def args_parser():
    parser = argparse.ArgumentParser(description='RL and IL cores.')

    parser.add_argument('--env_id', type=int, default=0, help='Id of environment')
    parser.add_argument('--render', type=int, default=0, help='render the environment during testing')
    parser.add_argument('--t_max', type=int, default=1000, help='maximum time step in one episode of each environment')
    parser.add_argument('--pixel_mode', type=int, default=0, help='Use pixel observation? WIP.')
    parser.add_argument('--debug_mode', type=int, default=0, help='Debug/protoptyping mode')

    ## Seeds    
    parser.add_argument('--seed', type=int, default=1, help='random seed for all (default: 1)')
    parser.add_argument('--max_step', type=int, default=None, help='maximal number of steps. (1 Million default)')
    parser.add_argument('--rl_method', default="trpo", help='method to optimize policy')
    parser.add_argument('--il_method', default=None, help='method to optimize reward')

    ## Network architecuture
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[None], help='list of hidden layers')
    parser.add_argument('--activation', action="store", default="relu", choices=["relu", "tanh", "sigmoid", "leakyrelu"], help='activation function')
 
    ## Step size, batch size, etc
    parser.add_argument('--learning_rate_pv', type=float, default=None, help='learning rate of policy and value')
    parser.add_argument('--learning_rate_d', type=float, default=None, help='learning rate of discriminator ')
    parser.add_argument('--mini_batch_size', type=int, default=256, help='Mini batch size for all updates (256)')
    parser.add_argument('--big_batch_size', type=int, default=1000, help='Big batch size per on-policy update')    
    parser.add_argument('--random_action', type=int, default=10000, help='Nmber of initial random action for off-policy methods (10000)')
    parser.add_argument('--tau_soft', type=float, default=0.005, help='tau for soft update (0.01 or 0.005)')  
    parser.add_argument('--gamma', type=float, default=None, help='discount factor. default: 0.99')
    parser.add_argument('--entropy_coef', default=None, help='Coefficient of entropy bonus')

    ## SAC options
    parser.add_argument('--log_std', type=float, default=0, help='initial log std (default: 0)')
    parser.add_argument('--symmetric', type=int, default=1, help='Use symmetric sampler (antithetic) or not)')
    parser.add_argument('--target_entropy_scale', type=float, default=1, help='Scale of entropy target')

    ## TRPO/PPO common options
    parser.add_argument('--trpo_max_kl', type=float, default=0.01, help='max KL for TRPO')
    parser.add_argument('--trpo_damping', type=float, default=0.1, help='trpo_damping scale of FIM for TRPO')
    parser.add_argument('--gae_tau', type=float, default=0.97, help='lambda for GAE (default: 0.97)')
    parser.add_argument('--gae_l2_reg', type=float, default=0, help='l2-regularization for GAE (default: no regularization)')
    
    ## PPO options
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='Clip epsilon of PPO')
    parser.add_argument('--ppo_early', type=int, default=1, help='Early stop in PPO?')
    parser.add_argument('--ppo_gradient_clip', type=float, default=0, help='gradient clipping in PPO')
    parser.add_argument('--ppo_separate_net', type=int, default=1, help='Use separate nets for policy and value?')

    ## DDPG/TD3 options 
    parser.add_argument('--explore_std', type=float, default=0.1, help='exploration standard deviation')

    ## IRL options     
    parser.add_argument('--d_step', type=int, default=5, help='Number of discriminator update for on-policy methods')
    parser.add_argument('--clip_discriminator', default=None, help='clip reward in (-x, x) by sigmoid?')
    parser.add_argument('--gp_lambda', default=None, help='gradient penalty regularization')
    parser.add_argument('--gp_center', type=float, default=1, help='center of gradient penalty regularization')
    parser.add_argument('--gp_alpha', action="store", default="mix",choices=["mix", "real", "fake"], help='interpolation type of gradient penalty regularization')
    parser.add_argument('--gp_lp', type=int, default=0, help='Use LP version (with max) of gradient penalty regularization?')
    parser.add_argument('--bce_negative', type=int, default=0, help='Use negative reward for gail')

    ## InfoGAIL    
    parser.add_argument('--encode_dim', type=int, default=None, help='Dimension of z ')
    parser.add_argument('--info_coef', type=float, default=0.1, help='coefficient in infogail ')
    parser.add_argument('--info_loss_type', default="bce", help='loss type ("linear", "bce", "ace")')

    ## VILD  
    parser.add_argument('--worker_model', type=int, default=1, help='noise model of worker (p_omega). 1=Gaussian in the paper, 2=WIP extensions..')
    parser.add_argument('--worker_reward', type=int, default=0, help='(Only use when worker_modal >= 2) Add rewards from worker model or not.')
    parser.add_argument('--bc_step', type=int, default=1000, help='number of bc pre-trained steps for q_psi')
    parser.add_argument('--noise_t', type=float, default=1e-8, help='The standard deviation of Gaussian noise to be added before making transition')
    parser.add_argument('--per_alpha', type=float, default=2, help='Imporatnce sampling mode. 0=no IS, 1=IS without truncate, 2=IS with truncate')
    parser.add_argument('--mc_num', type=int, default=1, help='number of mc sample for q_psi (# of epsilon, without antithetic +-)')
    parser.add_argument('--q_step', type=int, default=None, help='q_psi update per one d_step')
    parser.add_argument('--vild_loss_type', default="linear", help='loss type ("linear", "BCE", "ACE")')
    parser.add_argument('--worker_reg_coef', type=float, default=0.0, help='(Only use when worker_modal >= 2) Coefficient of worker mu regularization')
    parser.add_argument('--psi_param_std', default=None, help='Use parameterized standard deviation of Gaussian or not. ')
    parser.add_argument('--psi_coef', default=None, help='Psi coefficient scaling')
    parser.add_argument('--test_string', default=None, help='testing/prototyping string to be added in name')

    ## Demonstration      
    parser.add_argument('--c_data', type=int, default=7, help='Use a preset of dataset')
    parser.add_argument('--demo_iter', type=int, default=None, help='index to expert data')
    parser.add_argument('--demo_file_size', type=int, default=10000, help='file name with max number of the expert state-action pairs (100000)')
    parser.add_argument('--demo_split_k', type=int, default=0, help='Split demonstrations into trajectories each with unique k or not? (0)')
    parser.add_argument('--traj_deterministic', type=int, default=0, help='Use trajectory with determinisitic policy')
    parser.add_argument('--noise_type', action="store", default="normal", choices=["normal", "SDNTN"], help="Noise type (normal: Gaussian noisy policy, SDNTN: TSD noisy policy)")
    parser.add_argument('--noise_level_list', type=float, default=[0.0], help='Noise level of demo to load')
    
    ## testing for robosuitereacher and rendering.
    parser.add_argument('--test_step_list', nargs='+', default=[None], type=int, help='list of test iteration')
    parser.add_argument('--test_save', type=int, default=1, help='save test results into numpy files.')

    ## plotting.
    parser.add_argument('--plot_save', type=int, default=0, help='save plot or not.')
    parser.add_argument('--plot_large', type=int, default=1, help='plot large figures.')
    parser.add_argument('--plot_show', type=int, default=1, help='show plot or not.')

    args = parser.parse_args()

    """ Using ID instead of env name is more convenient """
    env_dict = {
                # Standard continuous control 
                -4: "CarRacing-v0",     # Probably throw errors because states are images.
                -3: "LunarLanderContinuous-v2", 
                -2: "BipedalWalker-v2",
                -1: "MountainCarContinuous-v0",
                0 : "Pendulum-v0",

                # Mujoco
                1 : "InvertedPendulum",
                2 : "HalfCheetah",
                3 : "Reacher",
                4 : "Swimmer",
                5 : "Ant",
                6 : "Hopper",
                7 : "Walker2d",
                8 : "InvertedDoublePendulum",
                9 : "Humanoid",
                10: "HumanoidStandup",

                # Pybullet
                11 : "InvertedPendulumBulletEnv-v0",
                12 : "HalfCheetahBulletEnv-v0",
                13 : "ReacherBulletEnv-v0",
                14 : "",
                15 : "AntBulletEnv-v0",
                16 : "HopperBulletEnv-v0",
                17 : "Walker2DBulletEnv-v0",
                18 : "InvertedDoublePendulumBulletEnv-v0",
                19 : "HumanoidBulletEnv-v0",
                20 : "",

                # Robosuite.
                21 : "SawyerNutAssemblyRound",
                22 : "SawyerNutAssemblySquare",
                23 : "SawyerNutAssembly",
                24 : "SawyerPickPlaceBread",    
                25 : "SawyerPickPlaceCan",
                26 : "SawyerPickPlaceCereal",
                27 : "SawyerPickPlaceMilk",
                28 : "SawyerPickPlace",

                # Gym's Robotics tasks. Need newer gym version...

                # Standard discretre control
                40 : "CartPole-v0",
                41 : "MountainCar-v0",
                42 : "Acrobot-v1", 
                43 : "LunarLander-v2",  
                44 : "CarRacing-v0",     # untested

                # Atari ?
                50 : "BreakoutNoFrameskip-v4",
                51 : "SpaceInvaders-v0"

    }
    args.env_name = env_dict[args.env_id] 
    if (args.env_id >= 1 and args.env_id <= 10):  # change to v3 after update gym version.
        args.env_name += "-v2"
    args.env_discrete = True if args.env_id >= 40 else False
    args.env_atari = True if args.env_id >= 50 else False
    args.env_bullet = True if args.env_id >= 11 and args.env_id <= 20 else False
    args.env_robosuite = True if args.env_id >= 21 and args.env_id <= 28 else False  
    args.cnn = args.env_atari 

    """ learning methods' defaults """
    rl_default_parser(args) 
    irl_default_parser(args) 

    return args

def rl_default_parser(args):     
    if args.hidden_size[0] is None:
        if args.rl_method == "sac":
            args.hidden_size = (100, 100)
            args.activation = "relu"
        elif args.rl_method == "td3":
            args.hidden_size = (100, 100)
            args.activation = "relu"
        elif args.rl_method == "trpo":
            args.hidden_size = (100, 100)
            args.activation = "tanh"
            if args.env_robosuite:
                args.activation = "relu"
        elif args.rl_method == "ppo":
            args.hidden_size = (100, 100)
            args.activation = "relu"

        if args.il_method is not None and "bc" in args.il_method:
            args.hidden_size = (100, 100)
            args.activation = "tanh"
            if args.env_id == 9:
                args.activation = "relu"
        
    if args.learning_rate_pv is None:
        args.learning_rate_pv = 3e-4

    if args.learning_rate_d is None:
        args.learning_rate_d = 3e-4

    if args.gamma is None:
        args.gamma = 0.99  #  
        if args.rl_method == "trpo":    # 0.99 is okay. Use 0.995 to reproduce old results. 
            args.gamma = 0.995
            
    if args.entropy_coef == "None": args.entropy_coef = None 
    if args.entropy_coef is None:
        if args.rl_method == "trpo":
            args.entropy_coef = 0.0001
        elif args.rl_method == "ppo":
            args.entropy_coef = 0.0001
        else:
            args.entropy_coef = 0.0001
    args.entropy_coef = float(args.entropy_coef) 
            
    if "Reacher" in args.env_name:  # not included custom reacher task in robosuite. 
        args.t_max = 50 

    if not args.cnn:
        cprint("Using (%d, %d) %s networks." % (args.hidden_size[0], args.hidden_size[1], args.activation), p_color)   
    
def irl_default_parser(args):   
    if args.il_method == "None": args.il_method = None 

    if args.demo_iter is None:
        if args.rl_method == "sac" or args.rl_method == "td3":
            args.demo_iter = 1000000    # demonstrations from sac
        else:
            args.demo_iter = 5000000    # demonstrations from trpo 

    """ default datasets """    
    if args.c_data == 0:       ## clean data 
        args.noise_level_list = [0.0]
        args.demo_file_size = 10000
        args.worker_num = len(args.noise_level_list)
        if "LunarLanderContinuous" in args.env_name:    
            args.noise_type = "heuristic"   ## PD controller without noise
            args.demo_file_size = 10000 
    elif args.c_data == 1:       ## 1 chosen noisy data
        args.noise_level_list = [ args.noise_level_list]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
    elif args.c_data == 7:         ## Current 
        args.noise_level_list = [0.01, 0.05, 0.1, 0.25, 0.4,      0.6, 0.7, 0.8, 0.9, 1.0]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
        if "LunarLanderContinuous" in args.env_name:    
            args.noise_type = "heuristic" 
            args.demo_file_size = 2000 
    else:
        raise NotImplementedError      
    
    if args.gp_lambda == "None": args.gp_lambda = None 
    if args.gp_lambda is None:
        args.gp_lambda = 10    
        if args.rl_method == "ppo":    
            args.gp_lambda = 0.0 # otherwise it converge too slow. 
    args.gp_lambda = float(args.gp_lambda)      

    if args.clip_discriminator == "None": args.clip_discriminator = None 
    if args.clip_discriminator is None:
        if args.il_method == "irl" or args.il_method == "vild" or args.il_method == "airl":    
            args.clip_discriminator = 5     # bound reward in (0, 5) by using sigmoid * 5
            if args.il_method == "vild" and args.vild_loss_type.lower() == "bce": # log-sigmoid version of vild 
                args.clip_discriminator = 0
        elif args.il_method == "infogail" and args.info_loss_type == "linear":   # Wasserstein version of infogail 
            args.clip_discriminator = 5    
        else:
            args.clip_discriminator = 0    # no clip.
    args.clip_discriminator = float(args.clip_discriminator) 
    
    """ infogail defaults """ 
    if args.encode_dim == "None": args.encode_dim = None 
    if args.encode_dim is None: args.encode_dim = args.worker_num

    """ vild defaults """
    if args.q_step == "None": 
        args.q_step = None 
    if args.q_step is None: 
        args.q_step = 10       
        if args.rl_method == "sac": # off-policy loop. 
            args.q_step = 1 
    if args.psi_coef == "None": args.psi_coef = None 
    if args.psi_coef is None: 
        args.psi_coef = 1   # 1 works fine. Have this option just in case. 
    args.psi_coef = float(args.psi_coef) 

def rl_hypers_parser(args):
    hypers = ""
    if args.env_atari:
        hypers += "CNN-"
    else: 
        for i in range(len(args.hidden_size)):
            hypers += "%d-" % (args.hidden_size[i])
    hypers += "%s" % (args.activation)

    if args.rl_method != "sac" and args.rl_method != "td3" and "dqn" not in args.rl_method:
        hypers += "_ec%0.5f" % args.entropy_coef
        
    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers 

def irl_hypers_parser(args):
    hypers = ""
    hypers += "gp%0.3f" % args.gp_lambda
    hypers += "_cr%d" % args.clip_discriminator #legacy name is Clip Reward

    if args.bce_negative:   # use negative version of log-sigmoid rewards. 
        if args.il_method == "gail" \
            or args.il_method == "vail" \
            or (args.il_method == "vild" and args.vild_loss_type.lower() == "bce") \
            or (args.il_method == "infogail" and args.info_loss_type.lower() == "bce"):
            hypers += "_neg" 

    if args.il_method == "vild":
        hypers += "_alp%0.2f_qs%d_wm%d" % (args.per_alpha, args.q_step, args.worker_model)           
        hypers += "_wr%d" % (args.worker_reward)         
        if args.worker_model > 1 and args.worker_reg_coef > 0:
            hypers += "_rc%0.3f" % (args.worker_reg_coef)   
        if args.psi_coef != 1:
            hypers += "_pc%0.2f" % (args.psi_coef)
        if args.test_string is not None:
            hypers += "_%s" % args.test_string 

    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers 

def bc_hypers_parser(args):    
    hypers = ""
    for i in range(len(args.hidden_size)):
        hypers += "%d-" % (args.hidden_size[i])
    hypers += "%s" % (args.activation)

    weight_decay = 1e-5 # should be set via args, but fixed at this value for now.
    hypers += "_wdecay%0.5f" % weight_decay      

    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers 