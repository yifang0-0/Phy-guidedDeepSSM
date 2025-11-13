import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
from torchsummary import summary
from models.physical_augment.model_phy import MODEL_PHY

import roboticstoolbox as rtb
"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""


class AE_RNN_HInput(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False, ):
        super(AE_RNN_HInput, self).__init__()

        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = param.h_dim
        self.z_dim = param.z_dim
        self.x_phy_w = param.x_phy_w
        self.x_nn_w = param.x_nn_w
        self.n_layers = param.n_layers
        self.device = device
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        self.dataset = dataset
        self.epoch_counter=0
        self.phypen_wt = 10
        # self.LayerNorm = nn.LayerNorm(self.h_dim)
        
        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)
        # print("self.device", self.device)
        # print(self.param['A_prt'], self.param['B_prt'],self.param['C'],self.mpnt_wt)


        # feature-extracting transformations (phi_y, phi_u and phi_z)
        # self.phi_u = nn.Sequential(
        #     nn.Linear(self.u_dim, self.h_dim),
        #     nn.LayerNorm(self.h_dim),
        #     nn.Dropout(0.1),
            
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.h_dim),)
        self.phi_x = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            # nn.LayerNorm(self.h_dim),
            
            # nn.Dropout(0.1),
            
            # nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        

        
        # self.x_mean = nn.Sequential(
        #     nn.Linear(self.h_dim, self.h_dim),
        #     # nn.LayerNorm(self.h_dim),
        #     # nn.Dropout(0.1),
        #     # nn.ReLU(),
        #     nn.Tanh(),
            
        #     nn.Linear(self.h_dim, self.z_dim),)
        
        # self.x_logvar = nn.Sequential(
        #     nn.Linear(self.h_dim, self.z_dim),
        #     # nn.ReLU()
        #     )

        # encoder function (phi_enc) -> Inference
        self.dynn = nn.Sequential(
            nn.Linear(self.u_dim + self.h_dim, self.h_dim),
            # nn.LayerNorm(self.h_dim),
            
            # nn.Dropout(),
            # nn.Dropout(0.1),
            
            # nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            # nn.ReLU(),
            # nn.Tanh(),
           )
        

        
        self.menn = nn.Sequential(
            nn.Linear(self.z_dim+self.h_dim, self.h_dim),
            # nn.LayerNorm(self.h_dim),
            
            # nn.Dropout(0.1),
            # nn.Tanh(),
            # nn.Linear(self.h_dim, self.h_dim),
            
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(self.h_dim, self.h_dim),
            
            # nn.Tanh(),
            
            nn.Linear(self.h_dim, self.y_dim),
            # # nn.Dropout(0.1),
            # # nn.ReLU(),
            # nn.Tanh(),
            
            # nn.Linear(self.h_dim, self.y_dim),
            # nn.ReLU(),
            )
            

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.u_dim+self.z_dim, self.h_dim, self.n_layers, bias)
        # nn.init.xavier_uniform_(self.rnn.weight)

        
        
    def dyphy(self, u, x, u_norm_dict, y_norm_dict):

        x_phy_t = self.phy_aug.dynamic_model(u,x, u_norm_dict, y_norm_dict)  

        return x_phy_t
    
    # def mephy(self, u, x, u_norm_dict, y_norm_dict):
    #     y_phy_t = self.phy_aug.measurement_model(u,x, u_norm_dict, y_norm_dict)               
    #     return y_phy_t
    
    
    def mephy(self, u, x):
        y_phy_t = self.phy_aug.measurement_model(u,x)               
        return y_phy_t
    
    def phy_penalty(self, x, u, u_norm_dict, y_norm_dict):
        loss = self.phy_aug.robo_phy_penalty(x,u, u_norm_dict, y_norm_dict)     
        return loss
    def forward(self, u, y, u_norm_dict, y_norm_dict):
        #  batch size
        torch.autograd.set_detect_anomaly(True)
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        h = torch.rand(self.n_layers, batch_size, self.h_dim, dtype=torch.float32,device=self.device)
        
        x = torch.zeros(batch_size, self.z_dim, seq_len, dtype=torch.float32, device=self.device)
        

        # for all time steps
        for t in range(seq_len):
 
            if t == 0:
                x_tm1 =  x[:,:,t].clone()
            else: 
                x_tm1 = x[:,:,t-1].clone()

            # feature extraction: u_t
            phi_u_t = u[:, :, t]
            _, h = self.rnn(torch.cat([phi_u_t.unsqueeze(0),x_tm1.unsqueeze(0)], 2), h)

            if self.mpnt_wt<=0:
                # dynn_phi = self.dynn(torch.cat([phi_u_t, x_tm1], 1))
                # x_mean_nn = self.x_mean(dynn_phi)
                # print("h.shape", h.shape)
                # print("h[-1].shape", h[-1].shape)
                x_t = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                # x_t = x_mean_nn
                
            
            #save x_t
            x[:,:,t] = x_t
            # if t==10:
            #     print("h.shape", h.shape)
            #     print("h[0]", h[0])
            #     print("h[-1]", h[-1])
            #     print("x_t.shape", x_t.shape)
            #     print("x_t", x_t)
            #     print("phy_u_t.shape", phi_u_t.shape)
            #     print("phi_u_t", phi_u_t)
            if self.mpnt_wt<=0:
                #pure nn
                # phi_x_t = self.phi_x(x_t)
                
                y_hat_t = self.menn(torch.cat([x_t, h[0]], 1))
                loss += torch.sum((y_hat_t-y[:, :, t]) ** 2)
                
                # if t == 10:
                #     print("y_hat_t-y[:, :, t]", y_hat_t-y[:, :, t])
                #     print("y_hat_t.shape", y_hat_t.shape)
                #     print("y_hat_t", y_hat_t)
                #     print("y[:, :, t].shape", y[:, :, t].shape)
                #     print("y[:, :, t]", y[:, :, t])
                
                
                
            else:       
                pass        

            if self.epoch_counter % 10 == 0 and t == 10:
                print("y_hat.shape, y.shape", y_hat_t.shape, y.shape)
                print(f"Epoch {self.epoch_counter}")
                print("y_hat_t.shape", y_hat_t.shape)
                print("y_hat_t", y_hat_t)
                print("max(y_hat), min(y_hat):", torch.max(y_hat_t).item(), torch.min(y_hat_t).item())
                print("max(y), min(y):", torch.max(y[:,:,t]).item(), torch.min(y[:,:,t]).item())
                print("loss", loss.item())
        
        self.epoch_counter += 1        
        return loss

    def generate(self, u, u_norm_dict, y_norm_dict):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]
        y_hat = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        y_hat_sigma =  torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)


        x = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)
        h = torch.rand(self.n_layers, batch_size, self.h_dim, device=self.device)

        print("mpnt_wt: ",self.mpnt_wt)
        # for all time steps
        
        for t in range(seq_len):
            # torch.autograd.set_detect_anomaly(True)
            if t == 0:
                x_tm1 =  torch.zeros(batch_size, self.z_dim, device=self.device)
            else: 
                x_tm1 = x[:,:,t-1]

            # feature extraction: u_t
            # phi_u_t = self.phi_u(u[:, :, t])
            phi_u_t = u[:, :, t]
            
            _, h = self.rnn(torch.cat([phi_u_t.unsqueeze(0),x_tm1.unsqueeze(0)], 2), h)
                
            if self.mpnt_wt<=0:
                #pure nn
                x_t = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
            
            x[:,:,t] = x_t 
            

            if self.mpnt_wt<=0:
                # pure nn
                y_hat[:, :, t]  = self.menn(torch.cat([x_t, h[-1]], 1))
            else:
                # panelty
                pass

         
        y_hat_mu = y_hat

        return y_hat, y_hat_mu, y_hat_sigma, x

