from my_utils import *
from args_parser import * 
from core.agent import Agent

from core.bc import *

""" The main entry function for RL """
def main(args):

    if args.il_method is None:
        raise NotImplementedError
    else:
        method_type = "IL"
        
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True      # True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name 
    """ Create environment and get environment's info. """
    if args.env_atari:
        from my_utils.atari_wrappers import Task 
        env = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
        env_test = env 
    elif args.env_bullet:
        import pybullet 
        import pybullet_envs 
        pybullet.connect(pybullet.DIRECT)
        env = gym.make(env_name)
        env.seed(args.seed)  
        env_test = env        
        if args.render:
            env_test.render(mode="human")
    elif args.env_robosuite:
        from my_utils.my_robosuite_utils import make_robosuite_env
        args.t_max = 500 
        env = make_robosuite_env(args)
        env_test = env
        # the sampler use functions from python's random, so the seed are already set.
        env_name = args.env_name + "_reach"

    else: 
        env = gym.make(env_name)     
        env.seed(args.seed)  
        env_test = env

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
            env_test = NormalizeGymWrapper(env_test)
            print("Use state-normalized environments.")
        a_bound = np.asscalar(env.action_space.high[0])
        a_low = np.asscalar(env.action_space.low[0])
        assert a_bound == -a_low 
        assert a_bound == 1 
        print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

        if "LunarLanderContinuous" in env_name or "BipedalWalker" in env_name:
            from my_utils.my_gym_utils import ClipGymWrapper
            env = ClipGymWrapper(env) 
            env_test = ClipGymWrapper(env_test) 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.max_step = 1000000     
    if args.il_method == "bc":
        policy_updater = BC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    elif args.il_method == "dbc":
        policy_updater = DBC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    elif args.il_method == "cobc":
        policy_updater = COBC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    discriminator_updater = policy_updater 

    update_type = policy_updater.update_type  # "off_policy"
    
    """ Set method and hyper parameter in file name"""
    method_name = args.il_method.upper()
    hypers = bc_hypers_parser(args)     
    exp_name = "%s-%s-%s_s%d" % (discriminator_updater.traj_name, method_name, hypers, args.seed)

    """ Set path for result and model files """
    result_path = "./results_%s/%s/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
    model_path = "./results_%s/%s_models/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name) 
    pathlib.Path("./results_%s/%s/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 
    # if platform.system() != "Windows":
    pathlib.Path("./results_%s/%s_models/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 
    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))

    """ Storage and counters """
    step, i_iter, tt_g, tt_d, perform_test = 0, 0, 0, 0, 0
    log_interval = args.max_step // 1000     # 1000 lines in the text files
    if args.env_robosuite:
        log_interval = args.max_step // 500 # reduce to 500 lines to save experiment time
    save_model_interval = (log_interval * 10) # * (platform.system() != "Windows")  # do not save on my windows laptop
    print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
         (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

    # """ Reset seed again """  
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    """ Agent for testing in a separated environemnt """
    agent_test = Agent(env_test, render=args.render, t_max=args.t_max, test_cpu=test_cpu)
    if args.env_bullet: 
        log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)

    latent_code = None ## only for infogail 
    # state = env.reset()
    """ The actual learning loop"""
    for total_step in range(0, args.max_step + 1):

        """ Save the learned policy model """
        if save_model_interval > 0 and total_step % save_model_interval == 0: 
            policy_updater.save_model("%s_policy_T%d.pt" % (model_path, total_step))

        """ Test the policy before update """
        if total_step % log_interval == 0:
            perform_test = 1
         
        """ Test learned policy """
        if perform_test:
            log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
            train_acc = policy_updater.evaluate_train_accuray()
            perform_test = 0
            
        """ Update policy """    
        t0_g = time.time()
        policy_updater.update_policy(total_step)  
        tt_g += time.time() - t0_g
                
        """ Print out result to stdout and save it to a text file for plotting """
        if total_step % log_interval == 0:
        
            result_text = t_format("Step %7d " % (total_step), 0) 
            result_text += t_format("(bc%2.1f)s" % (tt_g), 0) 

            result_text += " | [R_te] "
            result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format("max: %.2f" % log_test['max_reward'], 1) \
                + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
         
            result_text += " | [MSE_tr] " + t_format(" %.4f" % (train_acc), 0)
            
            if args.il_method == "dbc":
                ## check estimated worker noise
                estimated_worker_noise = policy_updater.worker_net.get_worker_cov().to(device_cpu).detach().numpy().squeeze()
                if action_dim > 1:
                    estimated_worker_noise = estimated_worker_noise.mean(axis=0)  #average across action dim
                result_text += " | w_noise: %s" % (np.array2string(estimated_worker_noise, formatter={'float_kind':lambda x: "%.3f" % x}).replace('\n', '') )

            tt_g = 0

            print(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f) 

if __name__ == "__main__":
    args = args_parser()
    main(args)

