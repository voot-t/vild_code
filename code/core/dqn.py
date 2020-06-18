from my_utils import *
from core_nn.nn_ac import *

""" DQN and Double DQN """
class DQN():
    def __init__(self, state_dim, action_num, args, double_q=False, encode_dim=0): 
        self.state_dim = state_dim + encode_dim 
        self.action_num = action_num
        self.gamma = args.gamma

        self.double_q = double_q  
        self.epsilon_greedy = args.epsilon_greedy
        self.dqn_gradient_clip = args.dqn_gradient_clip
        self.tau_soft = args.tau_soft 
        self.target_soft = args.target_soft 

        self.cnn = args.cnn
        self.initilize_nets(args)
        self.update_type = "off_policy"
        self.update_counter = 0

    def initilize_nets(self, args):
        self.value_net = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation, cnn=self.cnn).to(device)
        self.value_net_target = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation, cnn=self.cnn).to(device)
        
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate_pv)  
        self.hard_update(self.value_net_target, self.value_net)

    def sample_action(self, x):
        if self.epsilon_greedy < 0: # Boltzmann exploration based on Q-values
            action =  torch.distributions.Categorical(torch.softmax(self.value_net.get_value(x), dim=1)).sample()
        else:
            if np.random.rand() < self.epsilon_greedy:
                action = torch.randint(low=0, high=self.action_num, size=(1,))
            else:
                action = self.value_net.get_value(x).max(1)[1]
        return action.data.squeeze()
                           
    def greedy_action(self, x):
        return self.value_net.get_value(x).max(1)[1].data.squeeze() 
        
    def policy_to_device(self, device):
        self.value_net = self.value_net.to(device) 

    def save_model(self, path):
        torch.save( self.value_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau_soft) + param.data * self.tau_soft)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update_policy(self, states, actions, next_states, rewards, masks): 
        q_next_target = self.value_net_target.get_value(next_states)
        if self.double_q:
            a_next_max = self.value_net.get_value(next_states).max(1)[1].detach() 
            q_next_max = q_next_target.gather(1, a_next_max.unsqueeze(-1))
        else:
            q_next_max = q_next_target.max(1)[0].unsqueeze(-1) 

        q_target = rewards + masks * self.gamma * q_next_max
        q_current = self.value_net.get_value(states).gather(1, actions.unsqueeze(-1))    # make sure this is correct    

        value_loss = F.mse_loss(q_current, q_target) 
            
        self.optimizer_value.zero_grad()
        value_loss.backward()
        if self.dqn_gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.dqn_gradient_clip)
        self.optimizer_value.step()

        if self.target_soft:
            self.soft_update(self.value_net_target, self.value_net)
        else:
            self.update_counter += 1
            if self.update_counter % (1/self.tau_soft):
                self.hard_update(self.value_net_target, self.value_net)
                self.update_counter = 0

""" Clipped DDQN. Better than DQN + double_q, but doubles the memory due to additional networks """      
class Clipped_DDQN(DQN):
    def __init__(self, state_dim, action_num, args, encode_dim=0): 
        super().__init__(state_dim, action_num, args, encode_dim)

    def initilize_nets(self, args):
        self.value_net = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_2 = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.value_net_target = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_target_2 = Value(self.state_dim, output_dim=self.action_num, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_value = torch.optim.Adam( list(self.value_net.parameters()) + list(self.value_net_2.parameters()), lr=args.learning_rate_pv)  
        self.hard_update(self.value_net_target, self.value_net)
        self.hard_update(self.value_net_target_2, self.value_net_2)

    def update_policy(self, states, actions, next_states, rewards, masks): 
        q_current, q_current_2 = self.value_net.get_value(states).gather(1, actions.unsqueeze(-1)), self.value_net_2(states).gather(1, actions.unsqueeze(-1))
        q_next_max, q_next_max_2 = self.value_net_target(next_states).max(1)[0].unsqueeze(-1), self.value_net_target_2(next_states).max(1)[0].unsqueeze(-1)  
        q_next_max = torch.min(q_next_max, q_next_max_2)
        q_target = rewards + masks * self.gamma * q_next_max

        value_loss = F.mse_loss(q_current, q_target) + F.mse_loss(q_current_2, q_target) 
            
        self.optimizer_value.zero_grad()
        value_loss.backward()
        if self.dqn_gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.dqn_gradient_clip)
        self.optimizer_value.step()

        if self.target_soft:
            self.soft_update(self.value_net_target, self.value_net)
            self.soft_update(self.value_net_target_2, self.value_net_2)
        else:
            self.update_counter += 1
            if self.update_counter % (1/self.tau_soft):
                self.hard_update(self.value_net_target, self.value_net)
                self.hard_update(self.value_net_target_2, self.value_net_2)
                self.update_counter = 0

""" Quantile regression DQN. Code is based on github.com/ShangtongZhang/DeepRL """
class QR_DQN(DQN):
    def __init__(self, state_dim, action_num, args, encode_dim=0):  
        self.quantile_num = args.quantile_num 
        self.cumulative_density = torch.from_numpy((2 * np.arange(self.quantile_num) + 1) / (2.0 * self.quantile_num)).view(1, -1).to(device)   # each tau_i in the QR-DQN paper
        super().__init__(state_dim, action_num, args, encode_dim)

    def initilize_nets(self, args):
        self.value_net = QR_Value(self.state_dim, action_num=self.action_num, quantile_num=self.quantile_num, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_target = QR_Value(self.state_dim, action_num=self.action_num, quantile_num=self.quantile_num, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate_pv)  
        self.hard_update(self.value_net_target, self.value_net)

    def huber(self, x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    def update_policy(self, states, actions, next_states, rewards, masks): 
        quantile_next_target = self.value_net_target(next_states)
        if self.double_q:
            a_max_next = self.value_net(next_states).mean(dim=2).max(dim=1)[1].view(-1,1,1).repeat(1,1,self.quantile_num)    # [batch_size, 1, quantile_num] tensor
        else:
            a_max_next = quantile_next_target.mean(dim=2).max(dim=1)[1].view(-1,1,1).repeat(1,1,self.quantile_num)   # [batch_size, 1, quantile_num] tensor
        quantile_next_max = quantile_next_target.gather(1, a_max_next)[:,0,:]

        quantile_target = rewards + masks * self.gamma * quantile_next_max.detach()
        quantile_current = self.value_net(states).gather(1, actions.view(-1,1,1).repeat(1,1,self.quantile_num))[:,0,:]

        # value_loss = F.smooth_l1_loss(quantile_current, quantile_target)    # Built-in Huber loss cannot be used since it gives averaged value.
        # value_loss = (self.huber((quantile_current-q_target)) * (self.cumulative_density - ((quantile_current-quantile_target).detach() < 0).float()).abs()).mean() # * self.quantile_num

        diff = quantile_target - quantile_current
        # value_loss = self.huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs() 
        value_loss = self.huber(diff) * (self.cumulative_density - (diff < 0).float()).abs() .detach() 
        value_loss = value_loss.mean() * self.quantile_num

        # value_loss = F.mse_loss(quantile_current, quantile_target) 

        # print(value_loss)

        self.optimizer_value.zero_grad()
        value_loss.backward()
        if self.dqn_gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.dqn_gradient_clip)
        self.optimizer_value.step()

        if self.target_soft:
            self.soft_update(self.value_net_target, self.value_net)
        else:
            self.update_counter += 1
            if self.update_counter % (1/self.tau_soft):
                self.hard_update(self.value_net_target, self.value_net)
                self.update_counter = 0
