from my_utils import *
from core_nn.nn_ac import Policy_Gaussian
from core_nn.nn_irl import * 
from core_nn.nn_vild import *

from core.irl import IRL    #
import h5py 

""" Behavior cloning. Standard least-square regression gradient steps.  """
class BC(IRL):  # extend IRL to get demonstrations-related functions
    def __init__(self, state_dim, action_dim, args, a_bound=1, initialize_net=True):
        self.update_type = "off_policy"
        self.a_bound = a_bound 
        self.weight_decay = 1e-5    # This seems to help a bit. 
        super().__init__(state_dim, action_dim, args)
            
    def initilize_nets(self, args):   
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv, weight_decay=self.weight_decay) 

    def update_policy(self, total_step=0):
        index = self.index_sampler()
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        action_mean, _, _ = self.policy_net( s_real )
        loss = 0.5 * ((a_real - action_mean) ** 2 ).mean()    ##

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()
        return 

    def evaluate_train_accuray(self):
        ## Compute accuracy on the whole TRAINING dataset. 
        s_real = self.real_state_tensor[:, :].to(device)
        a_real = self.real_action_tensor[:, :].to(device)
        with torch.no_grad():   # This makes evaluation faster. 
            action_mean, _, _ = self.policy_net( s_real )
        return 0.5 * ((a_real - action_mean.data) ** 2 ).mean()    ##

    # functions from AC class 
    def sample_action(self, x):
        return self.policy_net.sample_action(x)

    def greedy_action(self, x):
        return self.policy_net.greedy_action(x)

    def policy_to_device(self, device):
        self.policy_net = self.policy_net.to(device) 

    def save_model(self, path):
        torch.save( self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

""" Co-teaching. based on https://github.com/bhanML/Co-teaching """
class COBC(BC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, initialize_net=True):
        super().__init__(state_dim, action_dim, args, a_bound)
        self.step_per_epoch = self.data_size / self.mini_batch_size 
        n_epoch = math.ceil(args.max_step / self.step_per_epoch )
        exponent = 1
        num_gradual = 10
        forget_rate = 0.4   # optimistic noise rate of 0.4 (40% of data are noise, and 60% are clean)
        
        self.rate_schedule = np.ones(n_epoch+1)*forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
            
    def initilize_nets(self, args):   
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv, weight_decay=self.weight_decay)  

        self.policy_net_2 = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)
        self.optimizer_policy_2 = torch.optim.Adam(self.policy_net_2.parameters(), lr=args.learning_rate_pv, weight_decay=self.weight_decay)

    def update_policy(self, total_step=0):
        index = self.index_sampler()
        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)

        action_mean_1, _, _ = self.policy_net( s_real )
        action_mean_2, _, _ = self.policy_net_2( s_real )

        loss_1 = 0.5 * ((action_mean_1 - a_noise) ** 2 ).mean(dim=1)    ##
        loss_2 = 0.5 * ((action_mean_2 - a_noise) ** 2 ).mean(dim=1)    ##

        ind_1_sorted = torch.argsort(loss_1.data)
        loss_1_sorted = loss_1[ind_1_sorted]

        ind_2_sorted = torch.argsort(loss_2.data)
        loss_2_sorted = loss_2[ind_2_sorted]

        epoch = round( (total_step) // self.step_per_epoch)   #
        
        remember_rate = 1 - self.rate_schedule[epoch]
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]

        # exchange
        loss_1_update = 0.5 * ((a_noise[ind_2_update] - action_mean_1[ind_2_update]) ** 2 ).mean()
        loss_2_update = 0.5 * ((a_noise[ind_1_update] - action_mean_2[ind_1_update]) ** 2 ).mean()

        self.optimizer_policy.zero_grad()
        loss_1_update.backward()
        self.optimizer_policy.step()
        
        self.optimizer_policy_2.zero_grad()
        loss_2_update.backward()
        self.optimizer_policy_2.step()
        
        return 

""" Behavior cloning with diverse-quality data """
class DBC(BC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, initialize_net=True):
        self.worker_model = args.worker_model
        self.symmetric = args.symmetric
        self.squash_action = 1 #
        if self.worker_model >= 2:
            raise NotImplementedError
        super().__init__(state_dim, action_dim, args, a_bound)

        self.behavior_cloning(bc_step=args.bc_step)   #pre-train q_psi. Policy is not involved. 
            
    def initilize_nets(self, args):   
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv, weight_decay=self.weight_decay)  

        self.worker_net = Worker_Noise(self.state_dim, self.action_dim, worker_num=self.worker_num, worker_model=args.worker_model, hidden_size=(32,), activation="tanh").to(device)
        self.optimizer_worker = torch.optim.Adam(self.worker_net.parameters(), lr=args.learning_rate_d)    

        self.policy_psi_net = Policy_psi(self.state_dim, self.action_dim, worker_num=self.worker_num, \
            param_std=0, hidden_size=args.hidden_size, activation=args.activation, squash_action=1).to(device)
        self.optimizer_policy_psi = torch.optim.Adam(self.policy_psi_net.parameters(), lr=args.learning_rate_pv)  

    def update_policy(self, total_step=0):

        index = self.index_sampler()
        self.update_inference(index=index)    # update policy psi  
        self.update_worker_noise(index=index) 

        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device)   

        sample_actions, _, _, _ = self.policy_psi_net.sample_full(s_real, a_noise, worker_id, symmetric=False)
        action_mean, _, _ = self.policy_net( s_real )
        loss = 0.5 * ((sample_actions - action_mean) ** 2 ).mean()    ##
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

    def update_inference(self, index):
        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device)   

        worker_cov, worker_mu = self.worker_net(s_real, worker_id)        # each tensor size [batch_size, action_dim]
        worker_cov = worker_cov.data.detach()
        if self.worker_model ==  1:
            worker_mu = 0

        ent_coef = 0.001    

        sym = True 
        if self.worker_model >= 2:
            sym = False 
            
        sample_actions, log_probs, action_mean, action_log_std = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=sym)
        action_mean, _, _ = self.policy_net( s_real )
        
        if sym:
            worker_cov = worker_cov.repeat(2, 1)
            s_real = s_real.repeat(2, 1)
            a_noise = a_noise.repeat(2, 1)
            action_mean = action_mean.repeat(2, 1) 
            if self.worker_model !=  1:
                worker_mu = worker_mu.repeat(2, 1)
        
        ## Basically gaussian likelihood between policy output and sample action 
        rwd = (0.5 * (action_mean.detach() - sample_actions) ** 2 + ent_coef * log_probs ).mean() # [batch_size, action_dim] -> [1] 
        error = (0.5 * (a_noise - sample_actions - worker_mu) ** 2 / worker_cov ).mean() # [batch_size, action_dim] -> [1] 
        psi_loss = rwd + error 
        psi_loss = psi_loss * torch.min(worker_cov) 

        self.optimizer_policy_psi.zero_grad()
        psi_loss.backward()       
        self.optimizer_policy_psi.step()

    def update_worker_noise(self, index):
        """ update the parameters omega of worker_net  """

        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device) 

        worker_cov, worker_mu = self.worker_net(s_real, worker_id)        # each tensor size [batch_size, action_dim]
        if self.worker_model ==  1:
            worker_mu = 0

        sample_actions, _, _, _ = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=False)

        w_loss = 0.5 * ( (a_noise - sample_actions.data.detach() - worker_mu) ** 2 / worker_cov ).mean()
        w_loss += 0.5 * torch.log(worker_cov).mean() 

        self.optimizer_worker.zero_grad()
        w_loss.backward()
        self.optimizer_worker.step()

    # for pre-trianing q_psi
    def behavior_cloning(self, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0:
            return
        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device), self.real_worker_tensor.unsqueeze(-1).to(device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)

        optimizer_psi_bc = torch.optim.Adam(self.policy_psi_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                action_mean_psi, _, _ = self.policy_psi_net( s_batch, a_batch, w_batch)
                loss_psi = 0.5 * ((action_mean_psi - a_batch) ** 2 ).mean()    ##
                optimizer_psi_bc.zero_grad()
                loss_psi.backward()
                optimizer_psi_bc.step()

        t1 = time.time()
        print("Psi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss_psi.item()))
