import torch
import torch.nn as nn
from torch.nn import functional as F
from models.physical_augment.model_phy import MODEL_PHY

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class Transformer_AE(nn.Module):
    def __init__(self, param, device, sys_param={}, dataset="toy_lgssm", bias=False, k_peek=3):
        super(Transformer_AE, self).__init__()
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
        self.k_peek =606
        self.k_peek_true = 30
        self.epoch_counter = 0
        self.window_size = 606
        self.saved_phy = {} 
        
        

        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)


# python main_single.py --n_epoch=1000 --model='AE-TRANSFORMER' --h_dim=32  --dataset='cascaded_tank' --logdir='0706_sim_known_1'  --mpnt_wt=-1   --do_test  --z_dim=8  --u_dim=1 --y_dim=1 --init_lr=1e-2
        print("initialize the transformer", dataset)
        print("layer_number", self.n_layers)

        # Update input projection to account for u + peeked y
        self.input_proj = nn.Sequential(
            nn.Linear(self.u_dim + self.y_dim + self.y_dim, self.h_dim),
        )
        self.positional_encoding = PositionalEncoding(self.h_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=4,
            dim_feedforward=4*self.h_dim,
            activation='gelu',
            batch_first=True,
            # dropout=0.3,   
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.n_layers + 1,
            norm=nn.LayerNorm(self.h_dim)
        )

        self.x_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(self.h_dim, self.z_dim)
        )


        self.menn = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            # nn.Tanh(),
            nn.GELU(),

            # nn.Dropout(0.3),
            nn.Linear(self.h_dim, self.h_dim),
            nn.GELU(),
            # nn.Tanh(),
            # nn.Dropout(0.3),
            nn.Linear(self.h_dim, self.y_dim),
        )

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"- Transformer encoder: {sum(p.numel() for p in self.transformer_encoder.parameters() if p.requires_grad):,}")
        print(f"- State estimator (x_mean): {sum(p.numel() for p in self.x_mean.parameters() if p.requires_grad):,}")
        print(f"- Measurement network: {sum(p.numel() for p in self.menn.parameters() if p.requires_grad):,}")

    def generate_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

    def dyphy(self, u, x, u_norm_dict, y_norm_dict):
        return self.phy_aug.dynamic_model(u, x, u_norm_dict, y_norm_dict)

    def mephy(self, u, x):
        return self.phy_aug.measurement_model(u, x)
    
    def simphy(self, u, x, u_norm_dict, y_norm_dict):
        xt = self.phy_aug.dynamic_model(u, x, u_norm_dict, y_norm_dict)
        return self.phy_aug.measurement_model(u, xt)
        
    
    def generate_causal_window_mask(self, seq_len, window_size, device):
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)  # start with all masked (True)
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start:i + 1] = False  # allow attention in the window
        # print("seq_len, window_size,",seq_len, window_size)
        return mask
    
    def create_augmented_input(self, u, y_phy, y):
        if u.dim() == 3 and u.shape[1] == self.u_dim:
            u = u.transpose(1, 2)  # [B, T, u_dim]


        B, T, _ = u.shape
        k = min(self.k_peek, T)
        k_true = min(self.k_peek_true, T)
        
        # print("u.shape, y_phy.shape",u.shape, y_phy.shape)
        
        y_peek = y_phy[:, :k, :]  # [B, k, y_dim]
        y_pad = torch.zeros(B, T - k, self.y_dim, device=y.device)
        print("y.shape, y_peek.shape, y_pad.shape",y.shape,y_peek.shape, y_pad.shape)
        y_aug = torch.cat([y_peek, y_pad], dim=1)  # [B, T, y_dim]

        y_peek_true = y[:, :k_true, :]  # [B, k, y_dim]
        y_pad_true = torch.zeros(B, T - k_true, self.y_dim, device=y.device)
        y_aug_true = torch.cat([y_peek_true, y_pad_true], dim=1)  # [B, T, y_dim]
        # print("y_aug.shape",y_phy.shape)
        # print("y_aug_true.shape",y_aug_true.shape)
        u_aug = torch.cat([u, y_aug, y_aug_true], dim=2)  # [B, T, u_dim + y_dim]
        return u_aug, y

    def forward(self, u, y, u_norm_dict, y_norm_dict):
        # print("y.shape",y.shape)
        # print("u.shape",u.shape)
        # key = tuple(u[0, 0, :5].tolist())
        # print("key u is:" , key)
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        if y.dim() == 3 and y.shape[1] == self.y_dim:
            y = y.transpose(1, 2)  # [B, T, y_dim]
        x_phy = torch.zeros(batch_size, 12, seq_len+1, dtype=torch.float32, device=self.device)
        y_phy = torch.zeros(batch_size, self.y_dim, seq_len, dtype=torch.float32, device=self.device)
        if self.k_peek>0:
            print("self.k_peel. ",self.k_peek)
            for t in range(seq_len):
                x_phy[:,:,t+1] = self.dyphy(u[:, :, t],x_phy[:,:,t], u_norm_dict, y_norm_dict)
                y_phy[:,:,t] = self.mephy(u[:,:,t], x_phy[:,:,t])
                # self.saved_phy[key] = {'x': x_phy, 'y': y_phy}
        else: 
            x_phy = torch.zeros(batch_size, 12, seq_len+1, dtype=torch.float32, device=self.device)
            y_phy = torch.zeros(batch_size, self.y_dim, seq_len, dtype=torch.float32, device=self.device)
        
        # self.simphy(self, u, x, u_norm_dict, y_norm_dict)
        # print(y_phy)
        if y_phy.dim() == 3 and y_phy.shape[1] == self.y_dim:
            y_phy = y_phy.transpose(1, 2)  # [B, T, y_dim]
        u_aug, _ = self.create_augmented_input(u, y_phy, y)

        phi_u = self.positional_encoding(self.input_proj(u_aug))  # [B, T, h_dim]
        seq_len = phi_u.size(1)
        # causal_mask = self.generate_causal_mask(seq_len, phi_u.device)
        causal_mask = self.generate_causal_window_mask(seq_len, self.window_size, device=u.device)
        encoded = self.transformer_encoder(phi_u, mask=causal_mask)  # [B, T, h_dim]
        # encoded = self.transformer_encoder(phi_u)
        x_mean = self.x_mean(encoded)  # [B, T, z_dim]
        y_hat = self.menn(x_mean)      # [B, T, y_dim]
        lamba_smooth = 0
        smoothness_loss = torch.mean((y_hat[:, 1:] - y_hat[:, :-1])**2)
        loss_rmse =  torch.sum((y_hat - y) ** 2)
        print("rmse traning/val:", torch.sqrt(torch.mean((y_hat - y) ** 2)))
        # print("loss_rmse, lamba_smooth*smoothness_loss, lamba_smooth", loss_rmse, lamba_smooth*smoothness_loss,lamba_smooth)
        loss = loss_rmse +  lamba_smooth* smoothness_loss
        # loss = torch.sum((y_hat - y) ** 2)

        if self.epoch_counter % 50 == 0:
            print(f"Epoch {self.epoch_counter}")
            print("max(y_hat), min(y_hat):",y_hat.shape, torch.max(y_hat).item(), torch.min(y_hat).item())
            print("max(y), min(y):", y.shape,torch.max(y).item(), torch.min(y).item())
        self.epoch_counter += 1

        return loss

    def generate(self, u, y, u_norm_dict, y_norm_dict):
        # if u.dim() == 3 and u.shape[1] == self.u_dim:
        #     u = u.transpose(1, 2)
        batch_size = y.shape[0]
        seq_len = y.shape[2]
        
        if y.dim() == 3 and y.shape[1] == self.y_dim:
            y = y.transpose(1, 2)  # [B, T, y_dim]
        x_phy = torch.zeros(batch_size, 12, seq_len+1, dtype=torch.float32, device=self.device)
        y_phy = torch.zeros(batch_size, self.y_dim, seq_len, dtype=torch.float32, device=self.device)
        if self.k_peek>0:
            for t in range(seq_len):
                # print("calculate")
                x_phy[:,:,t+1] = self.dyphy(u[:, :, t],x_phy[:,:,t], u_norm_dict, y_norm_dict)
                y_phy[:,:,t] = self.mephy(u[:,:,t], x_phy[:,:,t])
                
        if y_phy.dim() == 3 and y_phy.shape[1] == self.y_dim:
            y_phy = y_phy.transpose(1, 2)  # [B, T, y_dim]
        u_aug, _ = self.create_augmented_input(u, y_phy, y)

        phi_u = self.positional_encoding(self.input_proj(u_aug))
        seq_len = phi_u.size(1)
        # causal_mask = self.generate_causal_mask(seq_len, u.device)
        causal_mask = self.generate_causal_window_mask(seq_len, self.window_size, device=u.device)
    
        encoded = self.transformer_encoder(phi_u, mask=causal_mask)
        # encoded = self.transformer_encoder(phi_u)
        
        x_mean = self.x_mean(encoded)
        y_hat = self.menn(x_mean)

        x_mean = x_mean.transpose(1, 2)  # [B, z_dim, T]
        y_hat = y_hat.transpose(1, 2)    # [B, y_dim, T]
        y_hat_sigma = torch.zeros_like(y_hat)

        # print("y_hat.shape", y_hat.shape)
        # print("u.shape", u.shape)
        # print("u_aug.shape", u_aug.shape)
        
        
        # print("x_mean.shape", x_mean.shape)

        return y_hat, y_hat, y_hat_sigma, x_mean
