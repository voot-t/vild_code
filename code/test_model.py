from my_utils import *
from args_parser import * 
from core.ac import *

""" Load policy model file, replay policy, and save return of each episode in np format. """
def main(args):

    if args.il_method is None:
        method_type = "RL"
        info_method = False 
        encode_dim = 0 
    else:
        method_type = "IL"
        if "info" in args.il_method:
            info_method = True
            encode_dim = args.encode_dim
        else:
            info_method = False 
            encode_dim = 0 

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True      # True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name 
    args.num_worker = 10 
    """ Create environment and get environment's info. """
    if args.env_atari:
        from my_utils.atari_wrappers import Task 
        env = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
    elif args.env_bullet:
        import pybullet 
        import pybullet_envs 
        pybullet.connect(pybullet.DIRECT)
        env = gym.make(env_name)
        env.seed(args.seed)  
        if args.render:
            env.render(mode="human")
    elif args.env_robosuite:
        from my_utils.my_robosuite_utils import make_robosuite_env
        args.t_max = 500 
        env = make_robosuite_env(args)
        # the sampler use functions from python's random, so the seed are already set.
        env_name = args.env_name + "_reach"
    else: 
        env = gym.make(env_name)     
        env.seed(args.seed)  

    state_dim = env.observation_space.shape[0]
    is_disc_action = args.env_discrete
    action_dim = (0 if is_disc_action else env.action_space.shape[0])
    if args.env_robosuite:
        action_dim = action_dim - 1     # we disable gripper for reaching 
    if is_disc_action:
        a_bound = 1
        action_num = env.action_space.n 
        print("State dim: %d, action num: %d" % (state_dim, action_num))
    else:
        """ always normalize env. """ 
        if np.asscalar(env.action_space.high[0]) != 1:
            from my_utils.my_gym_utils import NormalizeGymWrapper
            env = NormalizeGymWrapper(env)
            print("Use state-normalized environments.")
        a_bound = np.asscalar(env.action_space.high[0])
        a_low = np.asscalar(env.action_space.low[0])
        assert a_bound == -a_low 
        assert a_bound == 1 
        print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

        if "LunarLanderContinuous" in env_name or "BipedalWalker" in env_name:
            from my_utils.my_gym_utils import ClipGymWrapper
            env = ClipGymWrapper(env) 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """define actor and critic"""
    if is_disc_action:
        if args.rl_method == "dqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=False, encode_dim=encode_dim)
        if args.rl_method == "ddqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=True, encode_dim=encode_dim)
        if args.rl_method == "qr_dqn":
            policy_updater = QR_DQN(state_dim=state_dim, action_num=action_num, args=args, encode_dim=encode_dim)
        if args.rl_method == "clipped_ddqn":
            policy_updater = Clipped_DDQN(state_dim=state_dim, action_num=action_num, args=args, encode_dim=encode_dim)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=action_num, is_discrete=True, encode_dim=encode_dim)
    else:
        if args.rl_method == "ac":
            policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "sac":
            policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "td3":
            policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "trpo":
            policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)

        if args.il_method is not None and "bc" in args.il_method:
            from core.bc import BC 
            policy_updater = BC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)   #to load data. 

    update_type = policy_updater.update_type  # "on_policy" or "off_policy"
    if args.max_step is None:
        if update_type == "on_policy":
            args.max_step = 5000000
            model_interval = 50000
        elif update_type == "off_policy":
            args.max_step = 1000000     
            model_intereval = 1000
        if args.env_atari:
            args.max_step = args.max_step * 10 
        if args.env_robosuite:
            model_interval = model_interval * 2

    """ Set method and hyper parameter in file name"""
    if method_type == "IL":
        from core.irl import IRL 
        discriminator_updater = IRL(1, 1, args, False)

    """ Set method and hyper parameter in file name"""
    if method_type == "RL":
        method_name = args.rl_method.upper()
        hypers = rl_hypers_parser(args)    
    else:
        method_name = args.il_method.upper() + "_" + args.rl_method.upper()
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)         
        
        if args.il_method == "vild" and args.vild_loss_type.lower() != "linear":
            method_name += "_" + args.vild_loss_type.upper()   
        
        if args.il_method == "infogail" and args.info_loss_type.lower() != "bce":
            method_name += "_" + args.info_loss_type.upper()

        if "bc" in args.il_method:
            method_name = args.il_method.upper()
            hypers = bc_hypers_parser(args)     

    if method_type == "RL":
        exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)
    elif method_type == "IL":
        exp_name = "%s-%s-%s_s%d" % (discriminator_updater.traj_name, method_name, hypers, args.seed)

    info_list = [-1]
    if args.il_method == "infogail":
        info_list = np.arange(0, args.num_worker)

    if args.test_step_list[0] is None:
        test_step_list = list(range(0, args.max_step + 1, model_interval))
    else:
        test_step_list = args.test_step_list 

    for info_i in info_list:

        if info_i != -1:
            latent_code_test = torch.LongTensor(1, 1).fill_(info_i)
            latent_code_onehot_test = torch.FloatTensor(1, args.num_worker)
            latent_code_onehot_test.zero_()
            latent_code_onehot_test.scatter_(1, latent_code_test, 1)
            latent_code_onehot_test = latent_code_onehot_test.squeeze()  #should have size [num_worker]

        test_list = []
        for test_step in test_step_list:
            test_seed = 1   ## 
            if use_gpu:
                torch.cuda.manual_seed(test_seed)
                torch.backends.cudnn.deterministic = True
            np.random.seed(test_seed)
            random.seed(test_seed)

            """ Load policy """
            model_path = "./results_%s/%s_models/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name) 

            model_filename = model_path + ("_policy_T%d.pt" % (test_step))    
            policy_updater.load_model(model_filename)
            policy_updater.policy_to_device(device_cpu) 
            print("Policy model is loaded from %s" % model_filename )

            latent_code = None ## only for infogail 

            test_episode = 100
            return_list = []
            for i_episode in range(0, test_episode):
                state = env.reset()         
                    
                if args.render and args.env_robosuite:
                    env.viewer.set_camera(0)
        
                reward_episode = 0
                for t in range(0, args.t_max):                   
                    state_var = torch.FloatTensor(state)

                    if info_i != -1:
                        state_var = torch.cat((state_var, latent_code_onehot_test), 0)  # input of the policy function. 
                     
                    action = policy_updater.greedy_action(state_var.unsqueeze(0)).to(device_cpu).detach().numpy()
                    next_state, reward, done, _ = env.step(np.clip(action, a_min=a_low, a_max=a_bound) )    #same clip condition as expert trajectory
                    state = next_state
                    reward_episode += reward

                    if args.render:
                        env.render()
                        # time.sleep(0.0001)

                    if t + 1 == args.t_max:
                        done = 1
                    if done : 
                        break

                return_list += [reward_episode]
                if args.render:
                    print('Test epi %d: steps %d, return %.2f' % (i_episode, t, reward_episode))

            return_list = np.asarray(return_list)
            test_list += [return_list]
            print('Test model T%d: Average return: %.2f' % (test_step, return_list.mean()))

        if args.test_save:
            # save the array
            test_list = np.asarray(test_list) 
            test_result_path = "./results_%s/test/%s" % (method_type, env_name)
            pathlib.Path(test_result_path).mkdir(parents=True, exist_ok=True) 
            np.save( ("%s/%s-%s_e%d_test.npy" % (test_result_path, env_name, exp_name, test_episode)), test_list)
        
if __name__ == "__main__":
    args = args_parser()    
    main(args) 
