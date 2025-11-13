import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
from torchsummary import summary
from models.physical_augment.model_phy import MODEL_PHY
import torch.nn.init as init
import roboticstoolbox as rtb
"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""


class AE_RNN(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False, ):
        super(AE_RNN, self).__init__()

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
        self.nx = 4


        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)
        # print("self.device", self.device)
        # print(self.param['A_prt'], self.param['B_prt'],self.param['C'],self.mpnt_wt)


        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_u = nn.Sequential(
            nn.Linear(self.u_dim * self.nx, self.h_dim),
            nn.Linear(self.h_dim, self.h_dim),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        

        
        self.x_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            # nn.LeakyReLU(),
            )
        
        # self.x_logvar = nn.Sequential(
        #     nn.Linear(self.h_dim, self.z_dim),
        #     # nn.ReLU()
        #     )

        # encoder function (phi_enc) -> Inference
        
        self.x_hybrid = nn.Sequential(
            nn.Linear(self.z_dim*2, self.z_dim),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Linear(self.z_dim, self.z_dim),
            # nn.ReLU(),
            )
        self.x_gate = nn.Sequential(
            nn.Linear(self.z_dim*2, 1),   # Learn a scalar weight
            nn.Sigmoid()                 # Ensures output ∈ (0,1)
        )
        self.y_gate = nn.Sequential(
            nn.Linear(self.y_dim*2, 1),   # Learn a scalar weight
            nn.Sigmoid()                 # Ensures output ∈ (0,1)
        )
        self.dynn = nn.Sequential(
            nn.Linear(self.u_dim * self.nx, self.h_dim),
            # nn.Dropout(),
            # nn.Tanh(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            
            nn.Linear(self.h_dim, self.h_dim),
            # nn.ReLU(),
           )
        

        
        self.menn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # nn.Dropout(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            
            nn.Linear(self.h_dim, self.y_dim),
            # nn.ReLU(),
            )
      
        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.u_dim * self.nx, self.h_dim, self.n_layers, bias)

        self.y_hybrid = nn.Sequential(
            nn.Linear(self.y_dim*2, self.y_dim),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Linear(self.z_dim, self.z_dim),
            # nn.ReLU(),
            )
                  
    def dyphy(self, u, x, u_norm_dict, y_norm_dict):
        with torch.no_grad():
            x_phy_t = self.phy_aug.dynamic_model(u,x, u_norm_dict, y_norm_dict)  
            return x_phy_t
    
    # def mephy(self, u, x, u_norm_dict, y_norm_dict):
    #     y_phy_t = self.phy_aug.measurement_model(u,x, u_norm_dict, y_norm_dict)               
    #     return y_phy_t
    
    
    def mephy(self, u, x):
        with torch.no_grad():
            y_phy_t = self.phy_aug.measurement_model(u,x)               
            return y_phy_t
    
    def phy_penalty(self, x, u, u_norm_dict, y_norm_dict):
        if self.phy_aug.model_type == "KUKA":
            # print("linear")
            loss = self.phy_aug.robo_phy_penalty_linear(x,u, u_norm_dict, y_norm_dict)
        else:
            loss = 1
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
        
        u_padded = torch.cat([
            torch.zeros(u.shape[0], u.shape[1], self.nx-1, device=u.device),  # pad with zeros
            u
        ], dim=2) 
        
        # for all time steps
        for t in range(seq_len):
            # print("seq no.: ", t)
            # print("phi_u.is_cuda()", u[:, :, t].get_device())
            if t == 0:
                x_tm1 =  x[:,:,t].clone()
            else: 
                x_tm1 = x[:,:,t-1].clone()

            # feature extraction: u_t
            # phi_u_t = self.phi_u(u[:, :, t])
            u_lagged = u_padded[:, :, t : t + self.nx]  # [batch, u_dim, nx]
            u_lagged = u_lagged.transpose(1, 2).reshape(u.shape[0], -1)  # [batch, u_dim * nx]
            # phi_u_t = self.phi_u(u_lagged)
            
            if self.mpnt_wt>100:
                # pure physical
                
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_phy

            elif  self.mpnt_wt>=10:
                #physics augmented
                
                dynn_phi = self.dynn(torch.cat([u_lagged, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                # x_t =  self.x_nn_w*x_mean_nn + self.x_phy_w*x_mean_phy
                w_x = 2*self.x_gate(torch.cat([ x_mean_nn,  x_mean_phy],1))
                x_t = w_x * x_mean_nn + (2 - w_x) * x_mean_phy
                # x_t = self.x_hybrid(torch.cat([x_mean_nn,  x_mean_phy],1))
                if self.epoch_counter % 5 == 0 and t == 50:
                    print(f"Epoch {self.epoch_counter}")
                    print("x_mean_nn:", x_mean_nn)
                    print("max(x_mean_nn_0), min(x_mean_nn_0):", torch.max(x_mean_nn[:,0]).item(), torch.min(x_mean_nn[:,0]).item())
                    print("max(x_mean_phy_0), min(x_mean_phy_0):", torch.max(x_mean_phy[:,0]).item(), torch.min(x_mean_phy[:,0]).item())
                    
                    print("max(x_mean_nn_1), min(x_mean_nn_1):", torch.max(x_mean_nn[:,1]).item(), torch.min(x_mean_nn[:,1]).item())
                    print("max(x_mean_phy_1), min(x_mean_phy_1):", torch.max(x_mean_phy[:,1]).item(), torch.min(x_mean_phy[:,1]).item())
                    
                    print("x_gate:", w_x)
                    # linear_layer = self.x_hybrid[0]

                    # # Get the weights and biases
                    # weights = linear_layer.weight.data
                    # biases = linear_layer.bias.data

                    # # Print or return them
                    # print("x_phy, x_nn,Weights:\n", weights)
                    # print("x_phy, x_nn,Biases:\n", biases)
                # print("w1,w2: ", self.x_phy_w, self.x_nn_w)

            elif self.mpnt_wt<=0:
                dynn_phi = self.dynn(torch.cat([u_lagged, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
            
            #save x_t
            x[:,:,t] = x_t
            # recurrence: u_t+2 -> h_t+1
            _, h = self.rnn(u_lagged.unsqeezw(0), h)

            # if self.epoch_counter % 5 == 0 and t == 100:
            #     print(f"Epoch {self.epoch_counter}")
            #     print("max(x_0), min(x_0):", torch.max(x_t[:,0]).item(), torch.min(x_t[:,0]).item())
            #     print("max(x_1), min(x_1):", torch.max(x_t[:,1]).item(), torch.min(x_t[:,1]).item())
            # if self.epoch_counter % 5 == 0 and t == 100:
            #     print(f"Epoch {self.epoch_counter}")
            #     print("x_mean_nn:", x_mean_nn)
            #     print("max(x_mean_nn_0), min(x_mean_nn_0):", torch.max(x_mean_nn[:,0]).item(), torch.min(x_mean_nn[:,0]).item())
            #     print("max(x_mean_nn_1), min(x_mean_nn_1):", torch.max(x_mean_nn[:,1]).item(), torch.min(x_mean_nn[:,1]).item())
                
                # print("max(y), min(y):", torch.max(y).item(), torch.min(y).item())
            
            # print("all_loss", loss)

            if self.mpnt_wt>100:
                # pure physical constraints
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat_t = y_hat_phy
                loss += torch.sum((y_hat_t-y[:, :, t]) ** 2)



            elif self.mpnt_wt>=10:
                #physics augmented
                # phi_x_t = self.phi_x(torch.cat([x_t, x_logvar], 1))
                phi_x_t = self.phi_x(x_t)
                y_hat_nn = self.menn(phi_x_t)
                
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                w_y = 2*self.y_gate(torch.cat([y_hat_nn, y_hat_phy],1))
                y_hat_t = w_y * y_hat_nn + (2 - w_y) * y_hat_phy
                # y_hat_t = self.y_hybrid(torch.cat([y_hat_nn,  y_hat_phy],1))
                

                # phy_loss = self.phy_penalty(x_tm1, u[:,:,t] , u_norm_dict, y_norm_dict)
                # print("torch.sum((y_hat-y[:, :, t]) ** 2)", (torch.sum((y_hat-y[:, :, t]) ** 2)))
                # print("weighted phy_loss", self.phypen_wt*phy_loss)
                # loss += (torch.sum((y_hat-y[:, :, t]) ** 2)+ self.phypen_wt*phy_loss)
                loss += (torch.sum((y_hat_t-y[:, :, t]) ** 2))
                
                # print("loss now,   y_hat,y\n", loss, y_hat[0],y[0, :, t],(y_hat[0]-y[0, :, t]) ** 2,torch.sum((y_hat[0]-y[0, :, t]) ** 2), y_hat.shape)
            elif self.mpnt_wt<=-10:
                #physics guided
                phi_x_t = self.phi_x(x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                if y_hat_phy.dtype != self.y_phi_phy[0].weight.dtype:
                    y_hat_phy_f32 = y_hat_phy.to(self.y_phi_phy[0].weight.dtype)
                    y_phy_phi = self.y_phi_phy(y_hat_phy_f32)
                else:
                    y_phy_phi = self.y_phi_phy(y_hat_phy)
                    
                y_hat = self.menn_phy(torch.cat([phi_x_t,y_phy_phi],1))
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)

            elif self.mpnt_wt<=0:
                #pure nn
                phi_x_t = self.phi_x(x_t)
                # y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat_nn = self.menn(phi_x_t)
                y_hat_t = y_hat_nn
                loss += torch.sum((y_hat_t-y[:, :, t]) ** 2)
                # if t == 10:
                #     print("y_hat_t-y[:, :, t]", y_hat_t-y[:, :, t])
                #     print("y_hat_t.shape", y_hat_t.shape)
                #     print("y_hat_t", y_hat_t)
                #     print("y[:, :, t].shape", y[:, :, t].shape)
                #     print("y[:, :, t]", y[:, :, t])
                
            else:       
                pass        
                # ## add measurement known panalty
                # phi_x = self.phi_x(x_t)

                # y_hat = self.menn(phi_x)
                # measure_mean_t, measure_var_t = self.mephy(x_mean, x_logvar )
                # pred_measurepanalty_dist = tdist.Normal(measure_mean_t, measure_var_t.sqrt())
                # loss_panelty = torch.sum(pred_measurepanalty_dist.log_prob(y[:, :, t]))
                # loss += (torch.sum((y_hat-y[:, :, t]) ** 2) - self.mpnt_wt*loss_panelty)
                # Print max/min values every 50th epoch
                
            if self.epoch_counter % 5 == 0 and t == 50:
                print(f"Epoch {self.epoch_counter}")
                if self.mpnt_wt>=10:
                    print("max(y_hat_phy), min(y_hat_phy):", torch.max(y_hat_phy).item(), torch.min(y_hat_phy).item())
                if self.mpnt_wt<=10:
                    print("max(y_hat_nn), min(y_hat_nn):", torch.max(y_hat_nn).item(), torch.min(y_hat_nn).item())
                if self.mpnt_wt==10:
                    print("y_gate:", w_y)
            # print("phi_x_t", phi_x_t)
            # print("u_t", u[:, :, t])
            # print("x_t", x_t)
            # print("x_mean_nn:", x_mean_nn)
            # print("x_mean_phy:", x_mean_phy)
            
            # print(y_hat_t.shape, y[:, :, t].shape)
            # print("y_hat_t", y_hat_t)
            # print("y[:,:,t]", y[:, :, t])
            if self.epoch_counter % 5 == 0 and t == 100:

                print("max(y_hat), min(y_hat):", torch.max(y_hat_t).item(), torch.min(y_hat_t).item())
                print("max(y), min(y):", torch.max(y[:,:,t]).item(), torch.min(y[:,:,t]).item())
        
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
        # u: [batch, u_dim, seq_len]
        u_padded = torch.cat([
            torch.zeros(u.shape[0], u.shape[1], self.nx-1, device=u.device),  # pad with zeros
            u
        ], dim=2)  # [batch, u_dim, seq_len + nx - 1]
        print("mpnt_wt: ",self.mpnt_wt)
        # for all time steps
        
        for t in range(seq_len):
            # torch.autograd.set_detect_anomaly(True)
            if t == 0:
                x_tm1 =  torch.zeros(batch_size, self.z_dim, device=self.device)
                # x_tm1[:, 1] = 4.9728
                # x_tm1[:,0] = 6.4158
                
                print("x_tm1", x_tm1)
            else: 
                x_tm1 = x[:,:,t-1]

            # feature extraction: u_t
            u_lagged = u_padded[:, :, t : t + self.nx]  # [batch, u_dim, nx]
            u_lagged = u_lagged.transpose(1, 2).reshape(u.shape[0], -1)  # [batch, u_dim * nx]
            phi_u_t = self.phi_u(u_lagged)
            # phi_u_t = self.phi_u(u[:, :, t])

            if self.mpnt_wt>100:
                # pure physical
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_phy
            elif  self.mpnt_wt>=10:
                # physical augmentation CX
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)

                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                # x_t = x_mean_nn + x_mean_phy
                w_x = 2*self.x_gate(torch.cat([ x_mean_nn,  x_mean_phy],1))
                x_t = w_x * x_mean_nn + (2 - w_x) * x_mean_phy
                # x_t = self.x_hybrid(torch.cat([x_mean_nn,  x_mean_phy],1))
                
                # x_t =  self.x_nn_w*x_mean_nn + self.x_phy_w*x_mean_phy
                
                    
            elif self.mpnt_wt<=-10:
                #physics guided
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                if x_mean_phy.dtype != self.x_phi_phy[0].weight.dtype:
                    x_mean_phy_f32 = x_mean_phy.to(self.x_phi_phy[0].weight.dtype)
                    x_phy_phi = self.x_phi_phy(x_mean_phy_f32)
                else:
                    x_phy_phi = self.x_phi_phy(x_mean_phy)
                dynn_phi = self.dynn_phy(torch.cat([phi_u_t, h[-1],x_phy_phi], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
                
            elif self.mpnt_wt<=0:
                #pure nn
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn

            # phi_x = self.phi_x(x[:,:,t])
            # y_hat[:, :, t] = self.menn(phi_x)
            
            x[:,:,t] = x_t 
            # recurrence: u_t+2 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)
                        # print("all_loss", loss)
                        
            if self.mpnt_wt>100:
                # pure physical constraints
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat[:, :, t] = y_hat_phy
                
                        
            elif self.mpnt_wt>=10:
                
                # physical augmentation CX
                phi_x_t = self.phi_x(x_t)
                y_hat_nn = self.menn(phi_x_t)
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                w_y = 2*self.y_gate(torch.cat([y_hat_nn, y_hat_phy],1))

                y_hat[:, :, t] = w_y * y_hat_nn + (2 - w_y) * y_hat_phy
                # y_hat[:, :, t] = self.y_hybrid(torch.cat([y_hat_nn,  y_hat_phy],1))
                # print(y_hat.shape)
                # print(" y_hat,y\n", y_hat[3, :, t] )
                
            elif self.mpnt_wt<=-10:
                # physics guided 
                phi_x_t = self.phi_x(x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_phy_phi = self.y_phi_phy(y_hat_phy)
                y_hat[:, :, t] = self.menn_phy(torch.cat([phi_x_t,y_phy_phi],1))

            elif self.mpnt_wt<=0:
                # pure nn
                # remember to add y_var (maybe not)
                phi_x = self.phi_x(x_mean_nn)
                y_hat[:, :, t] = self.menn(phi_x)
                
            else:
                # panelty
                pass

         
        y_hat_mu = y_hat

        return y_hat, y_hat_mu, y_hat_sigma, x

