from my_utils import *

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, num_outputs=1, hidden_size=(100, 100), activation='tanh', normalization=None, clip=0):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
    
        self.clip = clip 
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim + action_dim 
        for nh in hidden_size:
            if normalization == "spectral":
                self.affine_layers.append(nn.utils.spectral_norm(nn.Linear(last_dim, nh)))
            elif normalization == "weight":
                self.affine_layers.append(nn.utils.weight_norm(nn.Linear(last_dim, nh)))
            else:
                self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.score_out = nn.Linear(last_dim, num_outputs)
        self.score_out.weight.data.mul_(0.1)
        self.score_out.bias.data.mul_(0.0)

        if normalization == "spectral":
            self.score_out = nn.utils.spectral_norm(self.score_out)
        elif normalization == "weight_norm":
            self.score_out = nn.utils.weight_norm(self.score_out)

    ## use by GP regularization code. Take x as (s,a) or s. 
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.score_out(x) 

        if self.clip > 0:   # if clip, we use sigmoid to bound in (0, clip) (positive)
            x = torch.sigmoid(x) * self.clip 
            
        if self.clip < 0:   # tanh to bound in (clip, -clip)
            x = torch.tanh(x) * -self.clip 
            
        return x

    ## used for reward. 
    def get_reward(self, s, a=None):
        x =  torch.cat((s,a), 1) if a is not None else s
        score = self.forward(x)
        return score

class VDB_discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, encode_dim=128, num_outputs=1, hidden_size=(100, 100), activation='tanh', normalization=None, clip=0):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
        
        self.clip = clip 
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim + action_dim
        for nh in hidden_size:
            if normalization == "spectral":
                self.affine_layers.append(nn.utils.spectral_norm(nn.Linear(last_dim, nh)))
            elif normalization == "weight":
                self.affine_layers.append(nn.utils.weight_norm(nn.Linear(last_dim, nh)))
            else:
                self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.encoder_mean = nn.Linear(last_dim, encode_dim)
        self.encoder_mean.weight.data.mul_(0.1)
        self.encoder_mean.bias.data.mul_(0.0)

        self.encoder_logstd = nn.Linear(last_dim, encode_dim)
        self.encoder_logstd.weight.data.mul_(0.1)
        self.encoder_logstd.bias.data.mul_(0.0)

        self.score_out = nn.Linear(encode_dim, num_outputs)
        self.score_out.weight.data.mul_(0.1)
        self.score_out.bias.data.mul_(0.0)

        if normalization == "spectral":
            self.score_out = nn.utils.spectral_norm(self.score_out)
            self.encoder_mean = nn.utils.spectral_norm(self.encoder_mean)
            self.encoder_logstd = nn.utils.spectral_norm(self.encoder_logstd)
        elif normalization == "weight_norm":
            self.score_out = nn.utils.weight_norm(self.score_out)
            self.encoder_mean = nn.utils.weight_norm(self.encoder_mean)
            self.encoder_logstd = nn.utils.weight_norm(self.encoder_logstd)

    ## use by GP regularization code  Take x as (s,a) or s. 
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        z_mean = self.encoder_mean(x) 
        z_logstd = self.encoder_logstd(x)

        z = z_mean +  torch.exp(z_logstd) * torch.randn_like(z_logstd) 
        score = self.score_out(z) 

        if self.clip > 0:   # if clip, we use sigmoid to bound in (0, clip) (positive argument)
            score = torch.sigmoid(score) * self.clip 
            
        if self.clip < 0:   # tanh to bound in (clip, -clip)
            score = torch.tanh(score) * -self.clip 
            
        return score 

    ## used for reward. Sigmoid is applied in the main code
    def get_reward(self, s, a=None):
        x = torch.cat((s,a), 1) if a is not None else s
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        z_mean = self.encoder_mean(x) 
        score = self.score_out(z_mean) 
        return score 

    def get_full(self, s, a=None):
        x = torch.cat((s,a), 1) if a is not None else s
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        z_mean = self.encoder_mean(x) 
        z_logstd = self.encoder_logstd(x)

        score = self.score_out(z_mean +  torch.exp(z_logstd) * torch.randn_like(z_logstd) ) 
        return score, z_mean, z_logstd 

class Posterior(nn.Module):
    def __init__(self, input_dim, encode_dim, hidden_size=(256, 256), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
        
        self.encode_dim = encode_dim 

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.score_head = nn.Linear(last_dim, encode_dim)
        self.score_head.weight.data.mul_(0.1)
        self.score_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.score_head(x) 
        return x 

    def get_logposterior(self, states, actions, latent_code):
        x = torch.cat((states, actions), 1)
        p = F.log_softmax(self.forward(x), dim=1) # [batch_size, code_dim]        

        ## Gather output according to the latent code, which should have size [batch_size, 1] and have discrete value in range [0,  code_dim-1]
        latent_code_e = latent_code.view(-1,1).expand(-1, self.encode_dim ).long().to(device)    ## [batch_size, 1] to [batch_size, code_dim]
        p_gather = p.gather(1, latent_code_e)[:,0].unsqueeze(-1)
        return p_gather 
        
    def sample_code(self):
        return torch.randint(0, self.encode_dim, size=(1,1))  
        