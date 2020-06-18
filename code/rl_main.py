from my_utils import *
from args_parser import * 
from core.agent import Agent
from core.dqn import *
from core.ac import *

""" The main entry function for RL """
def main(args):
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = False      # True to avoid moving gym's state to gpu tensor every step during testing.

    """ Create environment and get environment's info. """
    if args.env_atari:
        from my_utils.atari_wrappers import Task 
        env = Task(args.env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
        env_test = Task(args.env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
    elif args.env_bullet:
        import pybullet 
        import pybullet_envs 
        pybullet.connect(pybullet.DIRECT)
        env = gym.make(args.env_name)
        env.seed(args.seed)  
        env_test = env        
        if args.render:
            env_test.render(mode="human")
    else: 
        env = gym.make(args.env_name)     
        env_test = gym.make(args.env_name)
        env.seed(args.seed)  
        env_test.seed(args.seed)  

    state_dim = env.observation_space.shape[0]
    is_disc_action = args.env_discrete
    action_dim = (0 if is_disc_action else env.action_space.shape[0])
    if is_disc_action:
        a_bound = 1
        action_num = env.action_space.n 
        print("State dim: %d, action num: %d" % (state_dim, action_num))
    else:
        """ always normalize env. """ 
        from my_utils.my_gym_utils import NormalizeGymWrapper
        env = NormalizeGymWrapper(env)
        env_test = NormalizeGymWrapper(env_test)
        a_bound = np.asscalar(env.action_space.high[0])
        a_low = np.asscalar(env.action_space.low[0])
        assert a_bound == -a_low 
        print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

    """ Set method and hyper parameter in file name"""
    method_name = args.rl_method.upper()
    hypers = rl_hypers_parser(args)     
    exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)

    """ Set path for result and model files """
    result_path = "./RL_results/%s/%s/%s-%s" % (method_name, args.env_name, args.env_name, exp_name)
    model_path = "./RL_results/%s_models/%s/%s-%s" % (args.rl_method.upper(), args.env_name, args.env_name, exp_name) 
    pathlib.Path("./RL_results/%s/%s" % (method_name, args.env_name)).mkdir(parents=True, exist_ok=True) 
    if platform.system() != "Windows":
        pathlib.Path("./RL_results/%s_models/%s" % (method_name, args.env_name)).mkdir(parents=True, exist_ok=True) 
    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))

    """define actor and critic"""
    if is_disc_action:
        if args.rl_method == "dqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=False, cnn=args.cnn)
        if args.rl_method == "ddqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=True, cnn=args.cnn)
        if args.rl_method == "qr_dqn":
            policy_updater = QR_DQN(state_dim=state_dim, action_num=action_num, args=args, cnn=args.cnn)
        if args.rl_method == "clipped_ddqn":
            policy_updater = Clipped_DDQN(state_dim=state_dim, action_num=action_num, args=args, cnn=args.cnn)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=action_num, is_discrete=True, cnn=args.cnn)
    else:
        if args.rl_method == "ac":
            policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
        if args.rl_method == "sac":
            policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
        if args.rl_method == "td3":
            policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
        if args.rl_method == "trpo":
            policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    update_type = policy_updater.update_type  # "on_policy" or "off_policy"
    if args.max_step is None:
        if update_type == "on_policy":
            args.max_step = 5000000
        elif update_type == "off_policy":
            args.max_step = 1000000
        if args.env_atari:
            args.max_step = args.max_step * 10 
        
    """ Function to update the parameters of value and policy networks"""
    def update_params_g(batch):
        states = torch.FloatTensor(np.stack(batch.state)).to(device)
        next_states = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        masks = torch.FloatTensor(np.stack(batch.mask)).to(device).unsqueeze(-1)
        rewards = torch.FloatTensor(np.stack(batch.reward)).to(device).unsqueeze(-1)
        actions = torch.LongTensor(np.stack(batch.action)) if is_disc_action else torch.FloatTensor(np.stack(batch.action))

        policy_updater.update_policy(states, actions.to(device), next_states, rewards, masks)
    
    """ Storage and counters """
    memory = Memory(capacity=1000000)   # Memory buffer with 1 million max size.
    step, i_iter, tt_g = 0, 0, 0
    perform_test = 0    
    log_interval = args.max_step // 1000     # 1000 lines in the text files
    save_model_interval = (log_interval * 10) * (platform.system() != "Windows")  # do not save on my windows laptop
    print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
         (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

    """ Reset seed again """  
    if use_gpu:
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Agent for testing in a separated environemnt """
    agent_test = Agent(env_test, render=args.render, t_max=args.t_max, test_cpu=test_cpu)
    if args.env_bullet: 
        log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)

    state = env.reset()
    """ The actual learning loop"""
    for total_step in range(0, args.max_step + 1):

        """ Save the learned policy model """
        if save_model_interval > 0 and total_step % save_model_interval == 0: 
            policy_updater.save_model("%s_policy_T%d.pt" % (model_path, total_step))

        """ Test the policy before update """
        if total_step % log_interval == 0:
            perform_test = 1
         
        if perform_test:
            if args.env_bullet: 
                if done:
                    log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                    perform_test = 0
            else:
                log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                perform_test = 0

        """ take env step """
        if total_step <= args.random_action and update_type == "off_policy":
            action = env.action_space.sample()
        else:
            action = policy_updater.sample_action(torch.FloatTensor(state).to(device).unsqueeze(0)).to(device_cpu).detach().numpy()

        next_state, reward, done, _ = env.step(action)

        if step + 1 == args.t_max:
            done = 1
        memory.push(state, action, int(not done), next_state, reward, 0)
        state = next_state
        step = step + 1        

        """ reset env """
        if done :  # reset
            state = env.reset()
            step = 0
                          
        """ Update policy """
        if update_type == "on_policy":
            if memory.size() >= args.big_batch_size and done :
                t0_g = time.time()
                batch = memory.sample()
                update_params_g(batch=batch) 
                memory.reset() 
                tt_g += time.time() - t0_g

        elif update_type == "off_policy":
            if total_step >= args.big_batch_size:         
                t0_g = time.time()
                batch = memory.sample(args.mini_batch_size)     
                update_params_g(batch=batch)  
                tt_g += time.time() - t0_g
                     
        """ Print out result to stdout and save it to a text file for plotting """
        if total_step % log_interval == 0:
        
            result_text = t_format("Step %7d " % (total_step), 0) \
                        + t_format("(g%2.2f)s" % (tt_g), 1) 
            result_text += " | [R_te] " + t_format("min: %.2f" % log_test['min_reward'], 1) + t_format("max: %.2f" % log_test['max_reward'], 1) \
                            + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
            
            if (args.rl_method == "sac" or args.rl_method == "vac"):
                result_text += ("| ent %0.3f" % (policy_updater.entropy_coef))

            tt_g = 0
            print(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f) 

if __name__ == "__main__":
    args = args_parser()
    main(args)

