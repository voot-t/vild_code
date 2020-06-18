from my_utils import *
from args_parser import * 
from core.agent import Agent

from core.dqn import *
from core.ac import *
from core.irl import *
from core.vild import *

""" The main entry function for RL """
def main(args):

    if args.il_method is None:
        method_type = "RL"  # means we just do RL with environment's rewards 
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
    test_cpu = True      # Set to True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name 
    """ Create environment and get environment's info. """
    if args.env_atari:
        from my_utils.atari_wrappers import Task 
        env = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
        env_test = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
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
        env_test = make_robosuite_env(args)
        # the sampler use functions from python's random, so the seed are already set.
        env_name = args.env_name + "_reach"

    else: 
        env = gym.make(env_name)     
        env.seed(args.seed)  
        env_test = gym.make(env_name)
        env_test.seed(args.seed)  

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

    """define actor and critic"""
    if is_disc_action:  # work in progress...
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

    update_type = policy_updater.update_type  # "on_policy" or "off_policy"
    if args.max_step is None:
        if update_type == "on_policy":
            args.max_step = 5000000
            if args.psi_param_std is None: args.psi_param_std = 0 
        elif update_type == "off_policy":
            args.max_step = 1000000     
            if args.psi_param_std is None: args.psi_param_std = 1 
        if args.env_atari:
            args.max_step = args.max_step * 10 
        
    if method_type == "IL":
        if args.il_method == "irl": # maximum entropy IRL
            discriminator_updater = IRL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "gail":
            discriminator_updater = GAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "vail":
            discriminator_updater = VAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "airl":
            discriminator_updater = AIRL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)  # need entropy coefficient and policy         
        elif args.il_method == "vild":  
            discriminator_updater = VILD(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # need entropy coefficient           
        elif args.il_method == "infogail":  
            discriminator_updater = InfoGAIL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # AIRL version need entropy coefficent and policy   

        # pretrain pi for robosuite env. 
        if args.env_robosuite :
            discriminator_updater.behavior_cloning(policy_net=policy_updater.policy_net, learning_rate=args.learning_rate_pv, bc_step=args.bc_step) # pretrain pi 
        elif args.il_method == "vild":  # pretrain only q_psi
            discriminator_updater.behavior_cloning(policy_net=None, learning_rate=args.learning_rate_pv, bc_step=args.bc_step) 

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

    if method_type == "RL":
        exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)
    elif method_type == "IL":
        exp_name = "%s-%s-%s_s%d" % (discriminator_updater.traj_name, method_name, hypers, args.seed)

    """ Set path for result and model files """
    result_path = "./results_%s/%s/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
    model_path = "./results_%s/%s_models/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name) 
    pathlib.Path("./results_%s/%s/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 
    # if platform.system() != "Windows":
    pathlib.Path("./results_%s/%s_models/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 
    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))

    """ Function to update the parameters of value and policy networks"""
    def update_params_g(batch):
        states = torch.FloatTensor(np.stack(batch.state)).to(device)
        next_states = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        masks = torch.FloatTensor(np.stack(batch.mask)).to(device).unsqueeze(-1)

        actions = torch.LongTensor(np.stack(batch.action)).to(device) if is_disc_action else torch.FloatTensor(np.stack(batch.action)).to(device) 

        if method_type == "RL":
            rewards = torch.FloatTensor(np.stack(batch.reward)).to(device).unsqueeze(-1)
            policy_updater.update_policy(states, actions.to(device), next_states, rewards, masks)
        elif method_type == "IL":
            nonlocal d_rewards 
            d_rewards = discriminator_updater.compute_reward(states, actions).detach().data
            
            # Append one-hot vector of context to state.
            if info_method:
                latent_codes = torch.LongTensor(np.stack(batch.latent_code)).to(device).view(-1,1)    # [batch_size, 1] 
                d_rewards += discriminator_updater.compute_posterior_reward(states, actions, latent_codes).detach().data

                latent_codes_onehot = torch.FloatTensor(states.size(0), encode_dim).to(device)
                latent_codes_onehot.zero_()
                latent_codes_onehot.scatter_(1, latent_codes, 1)  #should have size [batch_size, num_worker]

                states = torch.cat((states, latent_codes_onehot), 1) 
                next_states = torch.cat((next_states, latent_codes_onehot), 1)  

            policy_updater.update_policy(states, actions, next_states, d_rewards, masks)
    
    """ Storage and counters """
    memory = Memory(capacity=1000000)   # Memory buffer with 1 million max size.
    step, i_iter, tt_g, tt_d, perform_test = 0, 0, 0, 0, 0
    d_rewards = torch.FloatTensor(1).fill_(0)   ## placeholder
    log_interval = args.max_step // 1000     # 1000 lines in the text files
    if args.env_robosuite:
        log_interval = args.max_step // 500 # reduce to 500 lines to save experiment time
    save_model_interval = (log_interval * 10) # * (platform.system() != "Windows")  # do not save model ?
    print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
         (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

    """ Reset seed again """  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Agent for testing in a separated environemnt """
    agent_test = Agent(env_test, render=args.render, t_max=args.t_max, test_cpu=test_cpu)
    if args.env_bullet: 
        log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)

    latent_code = None ## only for infogail 
    state = env.reset()
    done = 1 
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
            if not info_method:
                if not args.env_bullet:
                    log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                    perform_test = 0
                elif done: # Because env and env_test are the same object for pybullet. 
                    log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                    perform_test = 0
            else:
                log_test = []
                for i_k in range(0, encode_dim):
                    # latent_code_test = discriminator_updater.sample_code().fill_(i_k)   # legacy code that change rng sequences. Use this line to reproduce old results. 
                    latent_code_test = torch.LongTensor(size=(1,1)).fill_(i_k)
                    latent_code_onehot_test = torch.FloatTensor(1, encode_dim)
                    latent_code_onehot_test.zero_()
                    latent_code_onehot_test.scatter_(1, latent_code_test, 1)
                    log_test += [agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10, latent_code_onehot=latent_code_onehot_test.squeeze() )] # use 1 instead of 10 to save time?
                perform_test = 0

        if info_method and latent_code is None:
            latent_code = discriminator_updater.sample_code()    #sample scalar latent code from the prior p(c) which is uniform. 
            latent_code_onehot = torch.FloatTensor(1, encode_dim)
            latent_code_onehot.zero_()
            latent_code_onehot.scatter_(1, latent_code, 1)
            latent_code_onehot = latent_code_onehot.squeeze()  #should have size [encode_dim]
            latent_code = latent_code.detach().numpy()

        state_var = torch.FloatTensor(state)
        if latent_code is not None:
            state_var = torch.cat((state_var, latent_code_onehot), 0)  

        """ take env step """
        if total_step <= args.random_action and update_type == "off_policy":    # collect random actions first for off policy methods
            action = env.action_space.sample()
        else:
            action = policy_updater.sample_action(state_var.to(device).unsqueeze(0)).to(device_cpu).detach().numpy()

        if args.il_method == "vild":    # Add noise from Sigma_k to action (noise_t = sqrt(Sigma_k)) 
            action_u = action + args.noise_t * np.random.normal( np.zeros(action.shape), np.ones(action.shape) )
            next_state, reward, done, _ = env.step(action_u)
        else:
            next_state, reward, done, _ = env.step(action)

        if step + 1 == args.t_max:
            done = 1
        memory.push(state, action, int(not done), next_state, reward, latent_code)
        state = next_state
        step = step + 1        

        """ reset env """
        if done :  # reset
            state = env.reset()
            step = 0
            latent_code = None 
                        
        """ Update policy """
        if update_type == "on_policy":
            if memory.size() >= args.big_batch_size and done :
                batch = memory.sample()

                if method_type == "IL":
                    for i_d in range(0, args.d_step):
                        index = discriminator_updater.index_sampler()   # should be inside update_discriminator for cleaner code...
                        t0_d = time.time()
                        discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step) 
                        tt_d += time.time() - t0_d

                t0_g = time.time()
                update_params_g(batch=batch) 
                tt_g += time.time() - t0_g
                memory.reset() 

        elif update_type == "off_policy":
            if total_step >= args.big_batch_size:       

                if method_type == "IL":
                    index = discriminator_updater.index_sampler()
                    batch = memory.sample(args.mini_batch_size)    
                    t0_d = time.time()
                    discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step) 
                    tt_d += time.time() - t0_d  
                elif method_type == "RL":
                    batch = memory.sample(args.mini_batch_size)    
                    
                t0_g = time.time()
                update_params_g(batch=batch)  
                tt_g += time.time() - t0_g
                       
        """ Print out result to stdout and save it to a text file for plotting """
        if total_step % log_interval == 0:
        
            result_text = t_format("Step %7d " % (total_step), 0) 
            if method_type == "RL":
                result_text += t_format("(g%2.2f)s" % (tt_g), 1)  
            elif method_type == "IL":
                c_reward_list = d_rewards.to(device_cpu).detach().numpy()
                result_text += t_format("(g%2.1f+d%2.1f)s" % (tt_g, tt_d), 1) 
                result_text += " | [D] " + t_format("min: %.2f" % np.amin(c_reward_list), 0.5) + t_format(" max: %.2f" % np.amax(c_reward_list), 0.5)

            result_text += " | [R_te] "
            if not info_method:
                result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format("max: %.2f" % log_test['max_reward'], 1) \
                    + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
            else:        
                result_text += "Avg " 
                for i_k in range(0, encode_dim):
                    result_text += t_format("%d: %.2f (%.2f)" % (i_k, log_test[i_k]['avg_reward'], log_test[i_k]['std_reward']), 2)
        
            if (args.rl_method == "sac"):
                result_text += ("| ent %0.3f" % (policy_updater.entropy_coef))

            if args.il_method == "vild":
                ## check estimated worker noise
                estimated_worker_noise = discriminator_updater.worker_net.get_worker_cov().to(device_cpu).detach().numpy().squeeze()
                if action_dim > 1:
                    estimated_worker_noise = estimated_worker_noise.mean(axis=0)  #average across action dim
                result_text += " | w_noise: %s" % (np.array2string(estimated_worker_noise, formatter={'float_kind':lambda x: "%.5f" % x}).replace('\n', '') )
                    
            tt_g = 0
            tt_d = 0

            print(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f) 

if __name__ == "__main__":
    args = args_parser()
    main(args)

