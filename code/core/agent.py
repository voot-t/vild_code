from my_utils import *

class Agent:

    def __init__(self, env, render=0, clip=False, t_max=1000, test_cpu=True):
        self.env = env
        self.render = render
        self.test_cpu = test_cpu
        self.t_max = t_max      
        self.is_disc_action = len(env.action_space.shape) == 0

    def collect_samples_test(self, policy, max_num_episodes, latent_code_onehot=None ):
        log = dict()
        min_reward = 1e6
        max_reward = -1e6
        total_reward_list = []

        if self.test_cpu:
            policy.policy_to_device(device_cpu)
            device_x = device_cpu 
        else:
            device_x = device 

        for _ in range(0, max_num_episodes):
            reward_episode = 0
            state = self.env.reset()

            step = 0
            while True: # run an episode

                state_var = torch.FloatTensor(state)
                if latent_code_onehot is not None:
                    state_var = torch.cat((state_var, latent_code_onehot), 0)  

                action = policy.greedy_action(state_var.to(device_x).unsqueeze(0)).to(device_cpu).detach().numpy()

                next_state, reward, done, _ = self.env.step(action)    

                if step + 1 == self.t_max:
                    done = 1    

                reward_episode += reward   

                if self.render:
                    self.env.render(mode="human")
                    time.sleep(0.001)
                    
                if done:
                    break
                                 
                state = next_state
                step = step + 1    

            # log stats
            total_reward_list += [reward_episode]
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

        if self.test_cpu:
            policy.policy_to_device(device)

        log['avg_reward'] = np.mean(np.array(total_reward_list))   
        log['std_reward'] = np.std(np.array(total_reward_list)) / np.sqrt(max_num_episodes) 
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

        return log
