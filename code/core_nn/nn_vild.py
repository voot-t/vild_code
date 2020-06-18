from my_utils import *

class Policy_psi(nn.Module):
    def __init__(self, state_dim, action_dim, worker_num=1, hidden_size=(100, 100), activation='tanh', param_std=1, log_std=0, a_bound=1, tanh_mean=1, squash_action=1):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        num_worker = worker_num
        self.tanh_mean = tanh_mean
        self.squash_action = squash_action
        self.num_worker = num_worker

        self.action_dim = action_dim

        self.param_std = param_std 
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.affine_layers = nn.ModuleList()

        last_dim = state_dim + action_dim         

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        
        self.action_mean_k = []
        for i in range(0, num_worker):
            self.action_mean_k += [nn.Linear(last_dim, action_dim)]
            self.action_mean_k[i].weight.data.mul_(0.1)
            self.action_mean_k[i].bias.data.mul_(0.0)
        self.action_mean_k = nn.ModuleList(self.action_mean_k)

        if self.param_std == 1:
            self.log_std_out_k = []
            for i in range(0, num_worker):
                self.log_std_out_k += [nn.Linear(last_dim, action_dim)]
                self.log_std_out_k[i].weight.data.mul_(0.1)
                self.log_std_out_k[i].bias.data.mul_(0.0)        
            self.log_std_out_k = nn.ModuleList(self.log_std_out_k)
        elif self.param_std == 0:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim, num_worker) * log_std)
    
        self.a_bound = a_bound
        # assert self.a_bound == 1

        self.is_disc_action = False

        self.zero_mean = torch.FloatTensor(1, action_dim).fill_(0).to(device) 
        self.unit_var = torch.FloatTensor(1, action_dim).fill_(1).to(device)

        self.logprob_holder = torch.FloatTensor(1).to(device_cpu)

    def forward(self, state, noisy_action, worker_ids):
        x = torch.cat( (state, noisy_action), 1)

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = []
        for i in range(0, self.num_worker):
            action_mean += [self.action_mean_k[i](x)]
            if self.tanh_mean:
                action_mean[i] = torch.tanh(action_mean[i]) * self.a_bound

        ## gather action_mean according to worker_ids
        action_mean = torch.stack(action_mean, dim=2) # tensor [b_size, a_dim, num_k]
        worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.num_worker) 
        action_mean = action_mean.gather(2, worker_ids_e)[:,:,0]

        if self.param_std == 1:
            action_log_std = []
            action_std = []

            for i in range(0, self.num_worker):
                action_log_std += [self.log_std_out_k[i](x) ]
                action_log_std[i] = torch.clamp(action_log_std[i], self.log_std_min, self.log_std_max)

            ## gather action_mean according to worker_ids
            action_log_std = torch.stack(action_log_std, dim=2) # tensor [b_size, a_dim, num_k]
            action_log_std = action_log_std.gather(2, worker_ids_e)[:,:,0]
            
        elif self.param_std == 0:
            worker_ids = worker_ids.type(torch.LongTensor).to(device)
            action_log_std_e = self.action_log_std.expand( state.size(0), -1, -1 ) # [b_size, a_dim, num_worker]
            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.num_worker) ## expanded [batch_size, action_dim, num_worker]
            action_log_std = action_log_std_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, noise_dim, num_worker] --> [batch_size, noise_dim]
        
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std
               

    def sample_full(self, states, noisy_action, worker_id, symmetric=1):

        action_mean, action_log_std, action_std = self.forward(states, noisy_action, worker_id)

        # This line uses one epsilon to sample actions for all states samples.
        epsilon = torch.FloatTensor(action_mean.size()).data.normal_(0, 1).to(device)
        
        action_raw = action_mean + action_std * epsilon
        log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
        
        if self.squash_action:
            action = torch.tanh(action_raw) * self.a_bound     
            log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True)    # ** this correction is only for a_bound = 1 !
        else: 
            action = action_raw

        if symmetric:
            action_sym_raw = action_mean - action_std * epsilon
            log_prob_sym = normal_log_density(action_sym_raw, action_mean, action_log_std, action_std) 

            if self.squash_action:
                action_sym = torch.tanh(action_sym_raw) * self.a_bound
                log_prob_sym -= torch.log(1 - torch.tanh(action_sym_raw).pow(2) + 1e-8).sum(1, keepdim=True)
            else:
                action_sym = action_sym_raw 

            ## concat them along batch dimension, return tensors with double batch size
            action = torch.cat( (action, action_sym), 0 )
            log_prob = torch.cat( (log_prob, log_prob_sym), 0 )

        return action, log_prob, action_mean, action_log_std

class Worker_Noise(nn.Module):
    def __init__(self, state_dim, action_dim, worker_num=1, worker_model=1, hidden_size=(256, 256), activation='relu', normalization=None):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.worker_num = worker_num

        self.a_bound = 1    # make this an input argument later. 

        """
        worker_model:
            1 = state independent covariance. Only model worker noise covariance. C(k) = diag[c_k]. (Used in the paper)
            2 = State independent covariance with shiftable Gaussian mean. u = a + mu(s). (WIP) 
            3 = State independent covariance with shiftable Gaussian mean k-dependent. u = a + mu(s,k). (WIP)  
            4 = Factored noise. Model worker noise vector and state difficulty. C(s, k) = diag[c_k * nu(s)]. (WIP) 
        """
        self.worker_model = worker_model
        self.action_dim = action_dim

        """ (log of) noise parameter for each worker. Taking log avoids using relu/clamp for non-negativity """
        self.worker_cov = nn.Parameter(torch.ones(self.action_dim, worker_num) * -1)

        if self.worker_model == 1:
            self.worker_mu_zero =  torch.zeros(1, self.action_dim).to(device) 
        
        elif self.worker_model == 2:
            self.affine_layers = nn.ModuleList()
            last_dim = state_dim
            for nh in hidden_size:
                self.affine_layers.append(nn.Linear(last_dim, nh))
                last_dim = nh

            self.worker_mu = nn.Linear(last_dim, action_dim)
            self.worker_mu.weight.data.mul_(0.1)
            self.worker_mu.bias.data.mul_(0.0)
        
        elif self.worker_model == 3:
            self.affine_layers = nn.ModuleList()
            last_dim = state_dim
            for nh in hidden_size:
                self.affine_layers.append(nn.Linear(last_dim, nh))
                last_dim = nh

            self.worker_mu_k = []
            for i in range(0, self.worker_num):
                self.worker_mu_k += [nn.Linear(last_dim, action_dim)]
                self.worker_mu_k[i].weight.data.mul_(0.1)
                self.worker_mu_k[i].bias.data.mul_(0.0)        
            self.worker_mu_k = nn.ModuleList(self.worker_mu_k)
            
        elif self.worker_model == 4:
            self.affine_layers = nn.ModuleList()
            last_dim = state_dim
            for nh in hidden_size:
                self.affine_layers.append(nn.Linear(last_dim, nh))
                last_dim = nh

            self.nu = nn.Linear(last_dim, action_dim)
            self.nu.weight.data.mul_(0.1)
            self.nu.bias.data.mul_(0.0)
        
            self.worker_mu_zero = torch.zeros(1, self.action_dim)

    def forward(self, states, worker_ids):
        """ 
        This function return estimated noise covariance and mean-shift which are tensors of size [batch_size, action_dim].
        states size = [batch_size_1, state_dim]
        worker_ids size = [batch_size_2] Long tensor
        Depend on the worker_model, two batch sizes could be different.
        """
        worker_ids = worker_ids.type(torch.LongTensor).to(device)

        ## state independent. 
        if self.worker_model == 1:

            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) ## expanded [batch_size, action_dim, worker_num]
            worker_cov_e = self.worker_cov.unsqueeze(0).expand(worker_ids.size(0), -1, -1)   ## [batch_size, action_dim, worker_num]
            worker_cov = worker_cov_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, action_dim, worker_num] --> [batch_size, action_dim]

            return  torch.exp(worker_cov)  + 1e-8, self.worker_mu_zero.repeat(states.size(0), 1)

        if self.worker_model == 2:

            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) ## expanded [batch_size, action_dim, worker_num]
            worker_cov_e = self.worker_cov.unsqueeze(0).expand(worker_ids.size(0), -1, -1)   ## [batch_size, action_dim, worker_num]
            worker_cov = worker_cov_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, action_dim, worker_num] --> [batch_size, action_dim]

            ## mean shift worker_mu(s,k) 
            x = states 
            for affine in self.affine_layers:
                x = self.activation(affine(x))
            worker_mu = torch.tanh(self.worker_mu(x) ) * self.a_bound 

            return torch.exp(worker_cov) + 1e-8, worker_mu

        if self.worker_model == 3:
            
            ## covariance C(k)
            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) ## expanded [batch_size, action_dim, worker_num]
            worker_cov_e = self.worker_cov.unsqueeze(0).expand(worker_ids.size(0), -1, -1)   ## [batch_size, action_dim, worker_num]
            worker_cov = worker_cov_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, action_dim, worker_num] --> [batch_size, action_dim]

            ## mean shift worker_mu(s,k) 
            x = states 
            for affine in self.affine_layers:
                x = self.activation(affine(x))

            worker_mu = []
            for i in range(0, self.worker_num):
                worker_mu += [torch.tanh(self.worker_mu_k[i](x)) * self.a_bound]  # bound in [-1, 1]

            ## gather worker_mu according to worker_ids
            worker_mu = torch.stack(worker_mu, dim=2) # tensor [b_size, a_dim, num_k]
            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) 
            worker_mu = worker_mu.gather(2, worker_ids_e)[:,:,0]

            return torch.exp(worker_cov) + 1e-8    , worker_mu

        ## Factored noise. 
        if self.worker_model == 4:
            for affine in self.affine_layers:
                states = self.activation(affine(states))
            state_diff = self.nu(states)
            state_diff = torch.sigmoid(state_diff)   

            worker_cov_e = self.worker_cov.unsqueeze(0).expand(worker_ids.size(0), -1, -1)   ## [batch_size, action_dim, worker_num]
            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) ## expanded [batch_size, action_dim, worker_num]
            worker_cov = worker_cov_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, action_dim, worker_num] --> [batch_size, action_dim]
            worker_cov = torch.exp(worker_cov)       

            return  (worker_cov * state_diff) + 1e-8, self.worker_mu_zero.repeat(states.size(0), 1)

    def get_worker_cov(self, mean=False):
        if mean:
            return torch.exp(self.worker_cov.mean(dim=0))    # mean across action dim.  return tensor size worker_num.
        else:
            return torch.exp(self.worker_cov)

