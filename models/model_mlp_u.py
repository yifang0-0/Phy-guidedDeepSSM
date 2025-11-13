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

def fun_RBF(x, label=2):
    if x.dim() == 1:
        x = x.unsqueeze(1)

    def rbf(x, c, sigma):
        return torch.exp(-((x - c) ** 2) / (sigma ** 2))

    # c1, sigma1 = -0.5, 0.1
    c2, sigma2 = -0.3, 0.2
    c3, sigma3 = 0.0, 0.2
    # c4, sigma4 = 0.1, 0.2
    # c5, sigma5 = 0.5, 0.1

    # y1 = rbf(x, c1, sigma1)
    y2 = -rbf(x, c2, sigma2)
    y3 = rbf(x, c3, sigma3)
    # y4 = -rbf(x, c4, sigma4)
    # y5 = rbf(x, c5, sigma5)


    if label == 2:
        return 0.2*(y2 + y3)
   
    else:
        raise ValueError("Invalid label for RBF.")

class MLP_U(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False, ):
        super(MLP_U, self).__init__()


        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = 20  # forced as requested
        self.z_dim = param.z_dim
        self.device = device
        self.x_phy_w = param.x_phy_w
        self.x_nn_w = param.x_nn_w
        self.n_layers = param.n_layers
        self.device = device
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        self.dataset = dataset
        self.epoch_counter=0
        self.phypen_wt = 10

        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)

        
        # self.input_dyn_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        # self.output_dyn_layer = nn.Linear(self.h_dim, self.z_dim)

        # self.input_meas_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        # self.output_meas_layer = nn.Linear(self.h_dim, self.y_dim)
        
        self.input_dyn_layer = nn.Sequential(
            nn.Linear(self.u_dim + self.z_dim, self.h_dim),
            # CustomRBF(label=2),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            # nn.ReLU()
        )
        self.output_dyn_layer = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            # nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            
        )

        # self.input_meas_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        self.input_meas_layer = nn.Sequential(
            nn.Linear(self.u_dim + self.z_dim, self.h_dim),
            # CustomRBF(label=2)
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2)
            # nn.ReLU()
        )
        self.output_meas_layer = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2)
            nn.ReLU(),
            nn.Linear(self.h_dim, self.y_dim)
        )
        
        
        
    def dyphy(self, u, x, u_norm_dict, y_norm_dict):

        x_phy_t = self.phy_aug.dynamic_model(u,x, u_norm_dict, y_norm_dict)  

        return x_phy_t

    
    def mephy(self, u, x):
        y_phy_t = self.phy_aug.measurement_model(u,x)               
        return y_phy_t
    


    def forward(self, u, y, u_norm_dict, y_norm_dict):
        #  batch size
        torch.autograd.set_detect_anomaly(True)
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        x = torch.zeros(batch_size, self.z_dim, seq_len, dtype=torch.float32, device=self.device)
        

        
        # for all time steps
        for t in range(seq_len):
            if t == 0:
                x_tm1 =  x[:,:,t].clone()
            else: 
                x_tm1 = x[:,:,t-1].clone()

            if not torch.isfinite(x_tm1).all() or x_tm1.abs().max() > 1e6:
                print(f"[DEBUG] Explosion detected at epoch={self.epoch_counter}, t={t}")
                print(f"  x_tm1 max={x_tm1.max().item():.4e}, min={x_tm1.min().item():.4e}")
                print(f"  x_tm1 = {x_tm1}")
                # raise RuntimeError("State exploded — stopping forward pass")
            if self.mpnt_wt>100:
                # pure physical
                x_t = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)

            elif  self.mpnt_wt>=10:
                #physics augmented
                dynn_phi = self.input_dyn_layer(torch.cat([u[:, :, t], x_tm1], 1))
                x_mean_nn = self.output_dyn_layer(dynn_phi)
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_nn + x_mean_phy
                
            elif self.mpnt_wt<=0:
                dynn_phi = self.input_dyn_layer(torch.cat([u[:, :, t], x_tm1], 1))
                x_mean_nn = self.output_dyn_layer(dynn_phi)
                x_t = x_mean_nn 
            
            #save x_t
            x[:,:,t] = x_t
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                print(f"NaN or Inf detected at step {self.epoch_counter}, t={t}")
                raise RuntimeError("NaN or Inf encountered in forward pass")
            if self.mpnt_wt>100:
                # pure physical constraints
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat_t = y_hat_phy


            elif self.mpnt_wt>=10:
                #physics augmented
                phi_x_t = self.input_meas_layer(torch.cat([u[:, :, t], x_t], 1))
                y_hat_nn = self.output_meas_layer(phi_x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat_t =  y_hat_nn  +  y_hat_phy

            elif self.mpnt_wt<=0:
                #pure nn
                phi_x_t = self.input_meas_layer(torch.cat([u[:, :, t], x_t], 1))
                y_hat_nn = self.output_meas_layer(phi_x_t)
                y_hat_t = y_hat_nn 


            else:       
                pass        
            loss += torch.sum((y_hat_t-y[:, :, t]) ** 2)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf detected in loss")
        
        self.epoch_counter += 1        
        return loss


    def generate(self, u, u_norm_dict, y_norm_dict):
        batch_size = u.shape[0]
        seq_len = u.shape[-1]

        # Initialize outputs
        y_hat = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        y_hat_sigma = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        x = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)

        print("mpnt_wt: ", self.mpnt_wt)

        for t in range(seq_len):
            if t == 0:
                x_tm1 = torch.zeros(batch_size, self.z_dim, device=self.device)
            else:
                x_tm1 = x[:, :, t - 1]

            # Compute next latent state x_t based on mpnt_wt mode
            if self.mpnt_wt > 100:
                # Pure physical model
                x_t = self.dyphy(u[:, :, t], x_tm1, u_norm_dict, y_norm_dict)

            elif self.mpnt_wt >= 10:
                # Physics augmented model
                # Prepare features for NN (phi_u_t, phi_xm1_t must be defined, assuming feature functions)
                dynn_phi = (self.input_dyn_layer(torch.cat([u[:, :, t], x_tm1], 1)))
                x_mean_nn = self.output_dyn_layer(dynn_phi)
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_nn + x_mean_phy


            else:
                # Pure NN model (mpnt_wt <= 0)
                dynn_phi = (self.input_dyn_layer(torch.cat([u[:, :, t], x_tm1], 1)))
                x_mean_nn = self.output_dyn_layer(dynn_phi)
                x_t = x_mean_nn
            # Save latent state
            x[:, :, t] = x_t

            # Generate output y_hat based on mpnt_wt mode
            if self.mpnt_wt > 100:
                y_hat_phy = self.mephy(u[:, :, t], x_t)
                y_hat[:, :, t] = y_hat_phy

            elif self.mpnt_wt >= 10:
                phi_x_t = self.input_meas_layer(torch.cat([u[:, :, t], x_t], 1))
                y_hat_nn = self.output_meas_layer(phi_x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat[:, :, t] = y_hat_nn  +  y_hat_phy

            else:
                phi_x_t = self.input_meas_layer(torch.cat([u[:, :, t], x_t], 1))
                y_hat_nn = self.output_meas_layer(phi_x_t)
                y_hat[:, :, t] = y_hat_nn 

        y_hat_mu = y_hat

        return y_hat, y_hat_mu, y_hat_sigma, x