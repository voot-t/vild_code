from my_utils import * 
from core.irl import IRL 
from core_nn.nn_irl import *
from core_nn.nn_vild import *

class VILD(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater):
        super().__init__(state_dim, action_dim, args)
        self.mc_num = args.mc_num
        self.symmetric = args.symmetric
        self.noise_t = args.noise_t 
        self.per_alpha = args.per_alpha 
        self.worker_model = args.worker_model
        self.policy_updater = policy_updater
        self.loss_type = args.vild_loss_type.lower()
        self.psi_coef = args.psi_coef
        self.psi_param_std = args.psi_param_std

        self.worker_reg_coef = args.worker_reg_coef
        self.worker_reward = args.worker_reward
        self.clip_discriminator = args.clip_discriminator
        self.q_step = args.q_step 
        self.max_step = args.max_step 
        self.learning_rate_pv = args.learning_rate_pv
        self.squash_action = 1 #

        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
    
        self.initilize_worker_nets(args)

    def initilize_worker_nets(self, args):   
        """ worker noise p_omega. The network hidden size is used when worker_model >= 2 which is the work in progress. """
        self.worker_net = Worker_Noise(self.state_dim, self.action_dim, worker_num=self.worker_num, worker_model=args.worker_model, hidden_size=(32,), activation="tanh").to(device)
        self.optimizer_worker = torch.optim.Adam(self.worker_net.parameters(), lr=args.learning_rate_d)     
        """ worker policy q_psi """
        self.policy_psi_net = Policy_psi(self.state_dim, self.action_dim, worker_num=self.worker_num, \
            param_std=self.psi_param_std, hidden_size=args.hidden_size, activation=args.activation, squash_action=self.squash_action).to(device)
        self.optimizer_policy_psi = torch.optim.Adam(self.policy_psi_net.parameters(), lr=args.learning_rate_pv)  

    def compute_reward(self, states, actions, next_states=None, masks=None, worker=True):
        if self.loss_type == "bce":    # binary_cross entropy. corresponding to GAIL-like variant.
            if self.bce_negative:
                rwd =  F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
            else:
                rwd =  -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  # minimize agent label score. 
        else:
            rwd =  self.discrim_net.get_reward(states, actions)

        ## Work in progress.
        if worker and self.worker_model >= 2 and self.worker_reward:
            b_size = states.size(0) 
            states_pair = torch.repeat_interleave(states, repeats=self.worker_num, dim=0)
            worker_id_pair = torch.arange(0, self.worker_num, dtype=torch.long).unsqueeze(-1).repeat(states.size(0), 1)

            worker_cov_fake, worker_mu_fake = self.worker_net(states_pair, worker_id_pair)
            worker_cov_fake = worker_cov_fake.reshape(b_size, -1, self.action_dim)
            worker_mu_fake = worker_mu_fake.reshape(b_size, -1, self.action_dim) 
            shift = ((worker_mu_fake ** 2) / worker_cov_fake).mean().unsqueeze(-1)

            return rwd - 0.5 * torch.clamp(shift, max=self.clip_discriminator) 
        return rwd 

    ## reward when updating q_psi. Same as above except WIP stuffs. 
    def compute_inference_reward(self, states, actions):
        if self.loss_type == "bce":    # binary_cross entropy. corresponding to GAIL-like variant.
            if self.bce_negative:
                rwd =  F.logsigmoid(-self.discrim_net.get_reward(states, actions))   
            else:
                rwd =  -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  
        else:
            rwd =  self.discrim_net.get_reward(states, actions)
        return rwd 

    ## sample data index based on covariance and compute importance weight. 
    def importance_sampling(self):
        worker_cov_k = self.worker_net.get_worker_cov(mean=True).to(device_cpu).detach().numpy()
        prob_k = 1 / (worker_cov_k)
        prob_k = prob_k / prob_k.sum()
        index, iw = [], [] 
        index_worker_idx_tmp = self.index_worker_idx.copy()
        for i in range(0, self.mini_batch_size):
            ## sample k from per_prob.
            choice_k = np.random.choice(self.worker_num, size=1, p=prob_k )[0]  ## use [0] to make it a scalar, not array.
            ## Sample i data from index_worker_idx_tmp[k]
            index_i = np.random.randint(0, len(index_worker_idx_tmp[choice_k]))       
            
            index += [ index_i + index_worker_idx_tmp[choice_k][0] ]   # the sample index (start at 0) + start index of that k.   

            ## remove the i-th sample to not re-sample it.
            index_worker_idx_tmp[choice_k] = np.delete(index_worker_idx_tmp[choice_k], index_i)

            iw_tmp = 1 / (self.worker_num * prob_k[choice_k])
            if self.per_alpha == 2: ## Truncate 
                iw_tmp = np.minimum(1, iw_tmp) 
            iw += [ iw_tmp ]   #iw of that sample is w(k) = p(k)/prob_k(k) = 1/ (K*prob_k(k))
        index = torch.LongTensor( np.array(index)) 
        iw = torch.FloatTensor(iw).to(device).unsqueeze(-1)
        return index, iw 

    def update_discriminator(self, batch, index, total_step=0):
        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)
                
        self.update_inference(index=index)    # update policy psi  
        self.update_worker_noise(s_fake=s_fake, index=index)    # update worker noise 
        q_step = self.q_step
        if total_step > self.max_step // 4: # reduce the number of q-update for faster computation. 
            q_step = 1
        for _ in range(0, q_step - 1):
            index_q = self.index_sampler()   #so that we use index for discriminator update
            self.update_inference(index=index_q)   
            self.update_worker_noise(s_fake=s_fake, index=index_q)   
        
        iw = 1
        if self.per_alpha > 0:  
            index, iw = self.importance_sampling()
            if self.per_alpha == 3: # no IW. 
                iw = 1

        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device) 
        
        sample_actions, _, _, _ = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=False)
        sample_actions = sample_actions.detach() 
                
        x_real = self.discrim_net.get_reward(s_real, sample_actions)    # [batch_size, 1]
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   # [(big?) batch_size, 1]

        if self.loss_type == "linear":
            loss_real = (x_real * iw).mean()
            loss_fake = x_fake.mean() 
            loss = -(loss_real - loss_fake)

        elif self.loss_type == "bce":    # Binary_cross entropy for GAIL-like variant.
            if self.per_alpha > 0 and self.per_alpha != 3:
                adversarial_loss_real = torch.nn.BCEWithLogitsLoss(weight=iw) 
            else:
                adversarial_loss_real = torch.nn.BCEWithLogitsLoss() 
            adversarial_loss_fake = torch.nn.BCEWithLogitsLoss() 
            label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss_real(x_real, label_real)
            loss_fake = adversarial_loss_fake(x_fake, label_fake)
            loss = loss_real + loss_fake

            # cat_real = torch.cat((torch.FloatTensor(x_real.size(0), 1).fill_(0).to(device), x_real), 1)
            # cat_fake = torch.cat((torch.FloatTensor(x_fake.size(0), 1).fill_(0).to(device), x_fake), 1)
            # loss_real = -(F.log_softmax(cat_real, 1)[:, self.label_real] * iw).mean()
            # loss_fake = -(F.log_softmax(cat_fake, 1)[:, self.label_fake]).mean()
            # loss = loss_real + loss_fake


        elif self.loss_type == "ace":    # Not tested. AIRL cross entropy for AIRL-like variant.

            ent_coef = self.policy_updater.entropy_coef.detach()   
            log_probs_real = self.policy_updater.policy_net.get_log_prob(s_real, sample_actions).detach()
            log_probs_fake = self.policy_updater.policy_net.get_log_prob(s_fake, a_fake).detach()

            ## without IW. 
            adversarial_loss = torch.nn.CrossEntropyLoss() 
            label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
            loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
            loss = loss_real + loss_fake

            # ## Equivalent to using the CrossEntropyLoss like in the AIRL class, but we can use IW for each sample here. 
            # ## Should change to CrossEntropyLoss loss when it suppports weight for each sample like BCEWithLogitsLoss.
            # cat_real = torch.cat((ent_coef * log_probs_real, x_real), 1)
            # cat_fake = torch.cat((ent_coef * log_probs_fake, x_fake), 1)
            # loss_real = -(F.log_softmax(cat_real, 1)[:, self.label_real] * iw).mean()
            # loss_fake = -(F.log_softmax(cat_fake, 1)[:, self.label_fake]).mean()
            # loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat( (s_real, sample_actions), 1 ), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step() 

    """ Function to update the parameters of q_psi """
    def update_inference(self, index):
        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device)   # worker ID, LongTensor with value in [0, worker_num-1]
 
        worker_cov, worker_mu = self.worker_net(s_real, worker_id)        # each tensor size [batch_size, action_dim]
        worker_cov = worker_cov.data.detach()
        if self.worker_model ==  1:
            worker_mu = 0
            
        ent_coef = self.policy_updater.entropy_coef.detach() 

        sym = True 
        if self.worker_model >= 2:
            sym = False 

        """ sample action from q_psi """
        sample_actions, log_probs, action_mean, action_log_std = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=sym)

        if sym:
            worker_cov = worker_cov.repeat(2, 1)
            s_real = s_real.repeat(2, 1)
            a_noise = a_noise.repeat(2, 1)
            if self.worker_model !=  1:
                worker_mu = worker_mu.repeat(2, 1)
        
        error = (0.5 * (a_noise - sample_actions - worker_mu) ** 2 / worker_cov ).mean()    # [batch_size, action_dim] -> [1]. 
        rwd = self.compute_inference_reward( s_real, sample_actions).mean() - (ent_coef * log_probs).mean()     # [batch_size, 1] -> [1] 

        psi_loss = -(rwd / self.action_dim / self.psi_coef - error)    # divide rwd by action_dim because we use mean across action_dim to compute the error. psi_coef is there to further rescale reward. Default is 1 and there is no further rescaling. 
        psi_loss = psi_loss * torch.min(worker_cov)     # rescale gradients so that error have constant magnitude across iterations. 
        psi_loss += 0.001 * ((action_mean ** 2).mean() + (action_log_std ** 2).mean())  # Policy regularizer used in original SAC. Not sure if this is necessary.

        self.optimizer_policy_psi.zero_grad()
        psi_loss.backward()       
        self.optimizer_policy_psi.step()
        
    """ Function to update the parameters of worker net """
    def update_worker_noise(self, s_fake, index):

        s_real = self.real_state_tensor[index, :].to(device)
        a_noise = self.real_action_tensor[index, :].to(device)
        worker_id = self.real_worker_tensor[index].to(device) 

        worker_cov, worker_mu = self.worker_net(s_real, worker_id)
        if self.worker_model ==  1:
            worker_mu = 0

        sample_actions, _, _, _ = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=False)
    
        w_loss = 0.5 * ( (a_noise - sample_actions.data.detach() - worker_mu) ** 2 / worker_cov ).mean()
        w_loss += -0.5 * (self.noise_t**2) * (( 1 / worker_cov)).mean()    # the trace term    small and negigible. 
        w_loss += 0.5 * torch.log(worker_cov).mean()  # regularization term 
        
        ## WIP 
        if self.worker_model >= 2:
            """ Pair s to all possible k's. for each s_fake, repeat it for K times. 
            (If this takes too much forward pass computation, then use sampling random k value for each s) """
            s_fake_pair = s_fake.repeat(self.worker_num, 1)
            worker_id_pair = torch.arange(0, self.worker_num, dtype=torch.long).repeat_interleave(s_fake.size(0))
            worker_cov_fake, worker_mu_fake = self.worker_net(s_fake_pair, worker_id_pair)
            
            # worker_id_pair = torch.randint(0, self.worker_num, (s_fake.size(0),1 ))
            # worker_cov_fake, worker_mu_fake = self.worker_net(s_fake, worker_id_pair)   

            w_loss += -0.5 * ( worker_mu_fake**2 / worker_cov_fake.detach()).mean() # stop gradient for covariance. Only update covariance based on demonstration data.  

            w_loss += self.worker_reg_coef * (worker_mu_fake**2).mean()  # regularize mean
            w_loss += self.worker_reg_coef * (worker_mu**2).mean()  # regularize mean    

            # w_loss += 0.5 * torch.log(worker_cov_fake).mean()  # regularization term 

        self.optimizer_worker.zero_grad()
        w_loss.backward()
        self.optimizer_worker.step()

    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0:
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device), self.real_worker_tensor.unsqueeze(-1).to(device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)

        if policy_net is not None:
            optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)  
        optimizer_psi_bc = torch.optim.Adam(self.policy_psi_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                if policy_net is not None:
                    action_mean, _, _ = policy_net( s_batch )
                    loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()    ##
                    optimizer_pi_bc.zero_grad()
                    loss.backward()
                    optimizer_pi_bc.step()

                action_mean_psi, _, _ = self.policy_psi_net( s_batch, a_batch, w_batch)
                loss_psi = 0.5 * ((action_mean_psi - a_batch) ** 2 ).mean()    ##
                optimizer_psi_bc.zero_grad()
                loss_psi.backward()
                optimizer_psi_bc.step()

        t1 = time.time()
        if policy_net is not None:
            print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
        print("Psi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss_psi.item()))
