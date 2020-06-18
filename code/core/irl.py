from my_utils import *
from core_nn.nn_irl import * 
# from core_nn.nn_old import *
import h5py 

""" MaxEnt-IRL. I.e., Adversarial IL with linear loss function. """
class IRL(): 
    def __init__(self, state_dim, action_dim, args, initialize_net=True):
        self.mini_batch_size = args.mini_batch_size 
        self.gp_lambda = args.gp_lambda  
        self.gp_alpha = args.gp_alpha  
        self.gp_center = args.gp_center 
        self.gp_lp = args.gp_lp 
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.gamma = args.gamma

        self.load_demo_list(args, verbose=initialize_net)  
        if initialize_net:
            self.initilize_nets(args) 
            
    def initilize_nets(self, args):   
        self.discrim_net = Discriminator(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)  

    def load_demo_list(self, args, verbose=True):

        index_worker = []
        self.index_worker_idx = [] 
        self.m_return_list = []
        index_start = 0
        expert_state_list, expert_action_list, expert_reward_list, expert_mask_list, worker_id_list = [],[],[],[],[]

        if not args.env_robosuite:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
            if args.demo_split_k:
                self.traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)
                
            if args.noise_type != "normal":
                self.traj_name += ("_%s" % args.noise_type)
            if args.c_data == 1:
                self.traj_name += ("_%0.2f" % k_spec_list[0])
                
        else:
            ## Hard code for now. 
            robo_data = 10
            env_name = args.env_name + "_reach"
            filename = "../imitation_data/TRAJ_h5/%s/%s_chosen.txt" % (env_name, env_name)
            traj_path = "../imitation_data/TRAJ_h5/%s" % (env_name) ## We load demonstrations from pre-extracted h5py files to reduce processing time. Otherwise, we need to load the original demonstrations ( from hdf5 in RoboTurkPilot) and then clip them in every experiment trials.

            demo_idx = -1
            sort_list = np.arange(0, robo_data)   # demos sort/chosen 
            demo_list = []
            with open(filename, 'r') as ff:
                i = 0
                for line in ff:
                    line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
                    if demo_idx == -1:
                        demo_idx = line.index("demo") + 1
                    if np.any(sort_list == i):  # A very bad way to loop through 10 text lines. Should be simplified later...
                        demo_list += [int(line[demo_idx])]
                    i = i + 1            
            k_spec_list = demo_list 
            self.traj_name = "traj10" 
            
        worker_id = 0
        for k in range(0, len(k_spec_list)):
            k_spec = k_spec_list[k]

            if not args.env_robosuite:
                traj_filename = traj_path + ("/%s_TRAJ-N%d_%s%0.2f" % (args.env_name, args.demo_file_size, args.noise_type, k_spec))
            else:
                traj_filename = traj_path + ("/%s_TRAJ-ID%d" % (env_name, k_spec))


            if args.traj_deterministic:
                traj_filename += "_det"
            hf = h5py.File(traj_filename + ".h5", 'r')
            
            expert_mask = hf.get('expert_masks')[:]
            expert_mask_list += [expert_mask][:]
            expert_state_list += [hf.get('expert_states')[:]]
            expert_action_list += [hf.get('expert_actions')[:]]
            expert_reward_list += [hf.get('expert_rewards')[:]]
            reward_array = hf.get('expert_rewards')[:]
            step_num = expert_mask.shape[0]

            ## Set k=n and K=N. Work and pendulum and lunarlander. The results are not included in the paper.
            if not args.demo_split_k:
                worker_id = k 
                worker_id_list += [ np.ones(expert_mask.shape) * worker_id ]
                self.index_worker_idx += [ index_start + np.arange(0, step_num ) ] 
                index_start += step_num 

            else:
                
                ## need to loop through demo until mask = 0, then increase the k counter. 
                ## find index in expert_mask where value is 0
                zero_mask_idx = np.where(expert_mask == 0)[0]
                prev_idx = -1
                for i in range(0, len(zero_mask_idx)):
                    worker_id_list += [ np.ones(zero_mask_idx[i] - prev_idx) * worker_id ]
                    self.index_worker_idx += [ index_start + np.arange(0, zero_mask_idx[i] - prev_idx ) ] 
                    index_start += zero_mask_idx[i] - prev_idx

                    worker_id = worker_id + 1
                    prev_idx = zero_mask_idx[i]

            traj_num = step_num - np.sum(expert_mask)   
            m_return = np.sum(reward_array) / traj_num

            self.m_return_list += [ m_return ]

            if verbose:
                print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                    (colored(traj_filename, p_color), colored(traj_num, p_color), colored(expert_mask.shape[0] , p_color), \
                    colored( "%.2f" % (m_return), p_color )))

        expert_states = np.concatenate(expert_state_list, axis=0)
        expert_actions = np.concatenate(expert_action_list, axis=0)
        expert_rewards = np.concatenate(expert_reward_list, axis=0)
        expert_masks = np.concatenate(expert_mask_list, axis=0)
        expert_ids = np.concatenate(worker_id_list, axis=0)

        self.real_state_tensor = torch.FloatTensor(expert_states).to(device_cpu) 
        self.real_action_tensor = torch.FloatTensor(expert_actions).to(device_cpu) 
        self.real_mask_tensor = torch.FloatTensor(expert_masks).to(device_cpu) 
        self.real_worker_tensor = torch.LongTensor(expert_ids).to(device_cpu) 
        self.data_size = self.real_state_tensor.size(0) 
        # self.worker_num = worker_id + 1 # worker_id start at 0?
        self.worker_num = torch.unique(self.real_worker_tensor).size(0) # much cleaner code

        if verbose:
            print("Total data pairs: %s, K %s, state dim %s, action dim %s, a min %s, a_max %s" % \
                (colored(self.real_state_tensor.size(0), p_color), colored(self.worker_num, p_color), \
                colored(self.real_state_tensor.size(1), p_color), colored(self.real_action_tensor.size(1), p_color), 
                colored(torch.min(self.real_action_tensor).numpy(), p_color), colored(torch.max(self.real_action_tensor).numpy(), p_color)   \
                ))

    def compute_reward(self, states, actions, next_states=None, masks=None):
        return self.discrim_net.get_reward(states, actions)

    def index_sampler(self, offset=0):
        return torch.randperm(self.data_size-offset)[0:self.mini_batch_size].to(device_cpu)

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)    
        x_fake = self.discrim_net.get_reward(s_fake, a_fake) 
                
        loss_real = x_real.mean()
        loss_fake = x_fake.mean() 
        loss = -(loss_real - loss_fake)
    
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step() 
            
    def gp_regularizer(self, sa_real, sa_fake):
        if self.gp_lambda == 0:
            return 0

        real_data = sa_real.data
        fake_data = sa_fake.data
                                       
        if real_data.size(0) < fake_data.size(0):
            idx = np.random.permutation(fake_data.size(0))[0: real_data.size(0)]
            fake_data = fake_data[idx, :]
        else: 
            idx = np.random.permutation(real_data.size(0))[0: fake_data.size(0)]
            real_data = real_data[idx, :]
            
        if self.gp_alpha == "mix":
            alpha = torch.rand(real_data.size(0), 1).expand(real_data.size()).to(device)
            x_hat = alpha * real_data + (1 - alpha) * fake_data
        elif self.gp_alpha == "real":
            x_hat = real_data
        elif self.gp_alpha == "fake":
            x_hat = fake_data 

        x_hat_out = self.discrim_net(x_hat.to(device).requires_grad_())
        gradients = torch.autograd.grad(outputs=x_hat_out, inputs=x_hat, \
                        grad_outputs=torch.ones(x_hat_out.size()).to(device), \
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        if self.gp_lp:
            return ( torch.max(0, gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda
        else:
            return ((gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda

    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None :
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device), self.real_worker_tensor.unsqueeze(-1).to(device))

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                action_mean, _, _ = policy_net( s_batch )
                loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()    ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
        

""" GAIL """
class GAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                   
    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions)) # minimize agent label score. 
        
    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   
                
        adversarial_loss = torch.nn.BCEWithLogitsLoss() 
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake
        
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" VAIL """
class VAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args) 
        self.vdb_ic = 0.5   
        self.vdb_beta = 0    
        self.vdb_alpha_beta = 1e-5  
        self.bce_negative = args.bce_negative   # Code should be cleaner if we just extend GAIL class.
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                       
    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions)) # minimize agent label score. 
        
    def initilize_nets(self, args):   
        self.discrim_net = VDB_discriminator(self.state_dim, self.action_dim, encode_dim=128, hidden_size=args.hidden_size, activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)  

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real, z_real_mean, z_real_logstd = self.discrim_net.get_full(s_real, a_real)  
        x_fake, z_fake_mean, z_fake_logstd = self.discrim_net.get_full(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss() 
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake
    
        ## compute KL from E(z|x) = N(z_mean, z_std) to N(0,I). #sum across dim z, then average across batch size.
        kl_real = 0.5 * ( -z_real_logstd + torch.exp(z_real_logstd) + z_real_mean**2 - 1).sum(dim=1).mean()  
        kl_fake = 0.5 * ( -z_fake_logstd + torch.exp(z_fake_logstd) + z_fake_mean**2 - 1).sum(dim=1).mean()  
        bottleneck_reg = 0.5 * (kl_real + kl_fake) - self.vdb_ic

        loss += self.vdb_beta * bottleneck_reg
        self.vdb_beta = max(0, self.vdb_beta + self.vdb_alpha_beta * bottleneck_reg.detach().cpu().numpy())

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" AIRL """
class AIRL(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater=None):
        super().__init__(state_dim, action_dim, args)
        self.policy_updater = policy_updater
        self.label_real = 1 
        self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
              
    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   
            
        ent_coef = self.policy_updater.entropy_coef.detach()   
        log_probs_real = self.policy_updater.policy_net.get_log_prob(s_real, a_real).detach()
        log_probs_fake = self.policy_updater.policy_net.get_log_prob(s_fake, a_fake).detach()

        adversarial_loss = torch.nn.CrossEntropyLoss() 
        label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
        loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
        loss = loss_real + loss_fake
        
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" InfoGAIL """
class InfoGAIL(IRL):
    def __init__(self, state_dim, action_dim, args, encode_dim=None, policy_updater=None):
        super().__init__(state_dim, action_dim, args)
        if encode_dim is not None:
            self.encode_dim = encode_dim 
        else:
            self.encode_dim = self.worker_num
        self.info_coef = args.info_coef 
        self.loss_type = args.info_loss_type.lower() 
        self.policy_updater = policy_updater

        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                       
        self.initilize_posterior_nets(args) 

    def initilize_posterior_nets(self, args):
        self.posterior_net = Posterior(input_dim=self.state_dim + self.action_dim, encode_dim=self.encode_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.optimizer_posterior = torch.optim.Adam(self.posterior_net.parameters(), lr=args.learning_rate_d)  

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.loss_type == "bce":    # binary_cross entropy. corresponding to standard InfoGAIL
            if self.bce_negative:
                rwd =  F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
            else:
                rwd =  -F.logsigmoid(-self.discrim_net.get_reward(states, actions))
        else:   # Wasserstein variant of InfoGAIL. 
            rwd =  self.discrim_net.get_reward(states, actions)
        return rwd

    def compute_posterior_reward(self, states, actions, latent_codes, next_states=None, masks=None):
        reward_p = self.posterior_net.get_logposterior(states, actions, latent_codes) 
        return self.info_coef * reward_p

    def sample_code(self):
        return self.posterior_net.sample_code() 

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   
                
        if self.loss_type == "linear":
            loss_real = x_real.mean()
            loss_fake = x_fake.mean() 
            loss = -(loss_real - loss_fake)

        elif self.loss_type == "bce":    # Binary_cross entropy for GAIL-like variant.
            adversarial_loss = torch.nn.BCEWithLogitsLoss() 
            label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss(x_real, label_real)
            loss_fake = adversarial_loss(x_fake, label_fake)
            loss = loss_real + loss_fake
            
        elif self.loss_type == "ace":    # AIRL cross entropy for AIRL-like variant.
            ent_coef = self.policy_updater.entropy_coef.detach()   
            log_probs_real = self.policy_updater.policy_net.get_log_prob(s_real, a_real).detach()
            log_probs_fake = self.policy_updater.policy_net.get_log_prob(s_fake, a_fake).detach()

            adversarial_loss = torch.nn.CrossEntropyLoss() 
            label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
            loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
            loss = loss_real + loss_fake
            
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
        """ update posterior of infogail """
        latent_codes_fake = torch.LongTensor(np.stack(batch.latent_code)).to(device)    # Label: scalar value in range [0, code_dim-1] 
        latent_score = self.posterior_net( torch.cat((s_fake, a_fake), 1))

        posterior_loss = torch.nn.CrossEntropyLoss() 
        p_loss = posterior_loss(latent_score, latent_codes_fake.squeeze())

        self.optimizer_posterior.zero_grad()
        p_loss.backward()
        self.optimizer_posterior.step()
        
    ## Re-define BC function because it needs context variable as input. 
    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None :
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device), self.real_worker_tensor.unsqueeze(-1).to(device))

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                # We use k as context variable here. 
                latent_codes_onehot = torch.FloatTensor(s_batch.size(0), self.worker_num).to(device)
                latent_codes_onehot.zero_()
                latent_codes_onehot.scatter_(1, w_batch, 1)  #should have size [batch_size, num_worker]
                s_batch = torch.cat((s_batch, latent_codes_onehot), 1)  # input of the policy function. 

                action_mean, _, _ = policy_net( s_batch )
                loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()    ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
        