import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import AE_RNN, Transformer_AE, AE_RNN_HInput, MLP, AE_RNN_U, MLP_U, Liu, Liu_U
from . import AE_RNN, Transformer_AE,  Liu_U, MLP_U, AE_RNN_XU, AE_RNN_U, AE_RNN_U_SGM

from torchsummary import summary

class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, options, normalizer_input=None, normalizer_output=None,
                 *args, **kwargs):
        super(DynamicModel, self).__init__()
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.args = args
        self.kwargs = kwargs
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output
        self.model_name = model
        # self.normalizer
        self.zero_initial_state = False
    
        model_options = options['model_options']
        if "system_options" in options:
            system_options = options['system_options']
            # if self.normalizer_input is not None:
            #     system_options['x0'] = self.normalizer_output.normalize(system_options['x0'])
            #     print(system_options['x0'])
        # initialize the model
        print("model is: ", model)
        if model == 'AE-RNN':
            self.m = AE_RNN(model_options, options['device'],system_options,options['dataset'])
        elif model == 'AE-RNN-U':
            self.m = AE_RNN_U(model_options, options['device'],system_options,options['dataset'])
        elif model == 'AE-RNN-U-SGM':
            self.m = AE_RNN_U_SGM(model_options, options['device'],system_options,options['dataset'])
        elif model == 'AE-RNN-XU':
            self.m = AE_RNN_XU(model_options, options['device'],system_options,options['dataset'])
        elif model == 'MLP':
            self.m = MLP(model_options, options['device'],system_options,options['dataset'])
        elif model == 'MLP-U':
            self.m = MLP_U(model_options, options['device'],system_options,options['dataset'])
        elif model == 'LIU':
            self.m = Liu(model_options, options['device'],system_options,options['dataset'])
        elif model == 'LIU-U':
            self.m = Liu_U(model_options, options['device'],system_options,options['dataset'])
        elif model == 'AE-TRANSFORMER':
            self.m = Transformer_AE(model_options, options['device'],system_options,options['dataset'])
        elif model == 'AE-RNN-HInput':
            self.m = AE_RNN_HInput(model_options, options['device'],system_options,options['dataset'])
        else:
            raise Exception("Unimplemented model")
        
        #%% if on cpu, running summary could be problemetic
        # summary(self.m.to(options["device"]), input_size = [(model_options.u_dim,1),(model_options.y_dim,1)])



    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def forward(self, u, y=None):

        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
            # print("forward: input has been normalized")
            # normalizer_dict_u = self.normalizer_input.to_dict()
            
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)
            # print("forward: output has been normalized")
            # normalizer_dict_y = self.normalizer_input.to_dict()

        loss = self.m(u, y, self.normalizer_input, self.normalizer_output)

        return loss

    def generate(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
            normalizer_dict_u = self.normalizer_input.to_dict()
            # print("generate: input has been normalized")
            
            
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)
            # print("generate: output has been normalized")
            normalizer_dict_y = self.normalizer_input.to_dict()
        try:
            
            if self.model_name == 'AE-TRANSFORMER':
                y_sample, y_sample_mu, y_sample_sigma, z = self.m.generate(u, y, self.normalizer_input, self.normalizer_output)
            else: 
                y_sample, y_sample_mu, y_sample_sigma, z = self.m.generate(u, self.normalizer_input, self.normalizer_output)
            # %% I don't understand this part???
            if self.normalizer_output is not None:
                y_sample = self.normalizer_output.unnormalize(y_sample)
                y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
                y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)
                # print("generate: output has been unnormalized")
            return y_sample, y_sample_mu, y_sample_sigma,z
        
        except ValueError:
            y_sample, y_sample_mu, y_sample_sigma = self.m.generate(u)
            # %% I don't understand this part???
            if self.normalizer_output is not None:
                y_sample = self.normalizer_output.unnormalize(y_sample)
                y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
                y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)
                # print("generate: output has been unnormalized")
                
            return y_sample, y_sample_mu, y_sample_sigma