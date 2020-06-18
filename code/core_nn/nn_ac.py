from my_utils import *

class Policy_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh', param_std=1, log_std=0, a_bound=1, tanh_mean=1, squash_action=1):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.tanh_mean = tanh_mean
        self.squash_action = squash_action
        self.param_std = param_std 
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        if self.param_std == 1:
            self.log_std_out = nn.Linear(last_dim, action_dim)  # diagonal gaussian
            self.log_std_out.weight.data.mul_(0.1)
            self.log_std_out.bias.data.mul_(0.0)
        elif self.param_std == 0:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
            self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)
        elif self.param_std == -1:  # policy with fixed variance
            self.action_log_std = torch.ones(1, action_dim).to(device) * np.log(0.1)
            self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)

        self.a_bound = a_bound
        # assert self.a_bound == 1

        self.is_disc_action = False

        self.zero_mean = torch.FloatTensor(1, action_dim).fill_(0).to(device) 
        self.unit_var = torch.FloatTensor(1, action_dim).fill_(1).to(device)

        self.logprob_holder = torch.FloatTensor(1).to(device_cpu)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        if self.tanh_mean:
            action_mean = torch.tanh(action_mean) * self.a_bound

        if self.param_std == 1:
            action_log_std = self.log_std_out(x) 
            action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        else:
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def sample_full(self, states, symmetric=0):
        action_mean, action_log_std, action_std = self.forward(states)

        epsilon = torch.FloatTensor(action_mean.size()).data.normal_(0, 1).to(device)
        
        action_raw = action_mean + action_std * epsilon
        log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
        
        if self.squash_action == 1:
            action = torch.tanh(action_raw) * self.a_bound 
            log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True) 
        elif self.squash_action == -1:
            action = torch.clamp(action_raw, min=-self.a_bound, max=self.a_bound)
        elif self.squash_action == 0: 
            action = action_raw

        if symmetric:
            action_sym_raw = action_mean - action_std * epsilon
            log_prob_sym = normal_log_density(action_sym_raw, action_mean, action_log_std, action_std) 

            if self.squash_action == 1:
                action_sym = torch.tanh(action_sym_raw) * self.a_bound
                log_prob_sym -= torch.log(1 - torch.tanh(action_sym_raw).pow(2) + 1e-8).sum(1, keepdim=True)
            elif self.squash_action == -1:
                action_sym = torch.clamp(action_sym_raw, min=-self.a_bound, max=self.a_bound)
            elif self.squash_action == 0:
                action_sym = action_sym_raw 

            ## concat them along batch dimension, return tensors with double batch size
            action = torch.cat( (action, action_sym), 0 )
            log_prob = torch.cat( (log_prob, log_prob_sym), 0 )

        return action, log_prob, action_mean, action_log_std

    def sample_action(self, x, get_log_prob=False):
        action_mean, action_log_std, action_std = self.forward(x)
        action_raw = torch.normal(action_mean, action_std.to(action_mean.device))
        if self.squash_action == 1:
            action = torch.tanh(action_raw) * self.a_bound
        elif self.squash_action == -1:
            action = torch.clamp(action_raw, min=-self.a_bound, max=self.a_bound)
        elif self.squash_action == 0:
            action = action_raw 

        log_prob =  self.logprob_holder.data

        if get_log_prob:
            log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
            if self.squash_action == 1:
                log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True)

        return action.data.view(-1) 
        
    def greedy_action(self, x):
        action_mean, _, _ = self.forward(x)
        if self.squash_action == 1:
            action_mean = torch.tanh(action_mean) * self.a_bound
        return action_mean.data.view(-1)

    def get_log_prob(self, x, actions_raw, get_log_std=False): 
        action_mean, action_log_std, action_std = self.forward(x)
        log_prob = normal_log_density(actions_raw, action_mean, action_log_std, action_std)
        if self.squash_action:
            log_prob -= torch.log(1 - torch.tanh(actions_raw).pow(2) + 1e-8).sum(1, keepdim=True)
        if get_log_std:
            return log_prob, action_log_std
        else:
            return log_prob

    """ Uses for TRPO update with param_std = 0 """
    def compute_entropy(self):
        entropy = self.entropy_const + self.action_log_std.sum()
        return entropy

    """ Only works when param_std = 0 """
    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}

class Value(nn.Module):
    def __init__(self, input_dim, num_outputs=1, hidden_size=(256, 256), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, num_outputs)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.value_head(x)
        return x

    def get_value(self, x):
        return self.forward(x) 

""" Two networks in one object"""
class Value_2(nn.Module):
    def __init__(self, input_dim, num_outputs=1, hidden_size=(256, 256), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers_1 = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers_1.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.value_head_1 = nn.Linear(last_dim, num_outputs)
        self.value_head_1.weight.data.mul_(0.1)
        self.value_head_1.bias.data.mul_(0.0)

        self.affine_layers_2 = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers_2.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.value_head_2 = nn.Linear(last_dim, num_outputs)
        self.value_head_2.weight.data.mul_(0.1)
        self.value_head_2.bias.data.mul_(0.0)

    def forward(self, s, a):
        x_1 = torch.cat([s, a], 1)
        for affine in self.affine_layers_1:
            x_1 = self.activation(affine(x_1))
        x_1 = self.value_head_1(x_1)


        x_2 = torch.cat([s, a], 1)
        for affine in self.affine_layers_2:
            x_2 = self.activation(affine(x_2))
        x_2 = self.value_head_2(x_2)

        return x_1, x_2

    def get_q1(self, s, a):
        x_1 = torch.cat([s, a], 1)
        for affine in self.affine_layers_1:
            x_1 = self.activation(affine(x_1))
        x_1 = self.value_head_1(x_1)

        return x_1 

