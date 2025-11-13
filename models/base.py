import torch
import torch.nn as nn
from enum import Enum
import numpy as np


class Normalizer1D(nn.Module):
    _epsilon = 1e-16

    def __init__(self, scale, offset):
        super(Normalizer1D, self).__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32) + self._epsilon)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def to_dict(self):
        """
        Save the normalization parameters in a dictionary.
        Returns:
            dict: A dictionary containing 'scale' and 'offset' tensors.
        """
        # print("scale, offset: ", self.scale, self.offset)
        # exit()
        return {
            'scale': self.scale,  # Convert to NumPy array if needed
            'offset': self.offset  # Convert to NumPy array if needed
        }
    
    def normalize(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.offset) / self.scale
        return x.permute(0, 2, 1)

    def unnormalize(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.scale + self.offset
        return x.permute(0, 2, 1)

    def unnormalize_mean(self, x_mu):
        x_mu = x_mu.permute(0, 2, 1)
        x_mu = x_mu * self.scale + self.offset
        return x_mu.permute(0, 2, 1)

    def unnormalize_sigma(self, x_sigma):
        x_sigma = x_sigma.permute(0, 2, 1)
        x_sigma = x_sigma * self.scale
        return x_sigma.permute(0, 2, 1)


# class Normalizer1D(nn.Module):
#     _epsilon = 1e-16

#     def __init__(self, x_min, x_max):
#         super(Normalizer1D, self).__init__()
#         self.register_buffer('x_min', torch.tensor(x_min, dtype=torch.float32))
#         self.register_buffer('x_max', torch.tensor(x_max, dtype=torch.float32) + self._epsilon)
#         self.register_buffer('scale', self.x_max - self.x_min + self._epsilon)
#         self.register_buffer('offset', torch.tensor(x_min, dtype=torch.float32))
        

#     def normalize(self, x):
#         x = x.permute(0, 2, 1)
#         x = (x - self.x_min) / self.scale
#         return x.permute(0, 2, 1)

#     def unnormalize(self, x):
#         x = x.permute(0, 2, 1)
#         x = x * self.scale + self.x_min
#         return x.permute(0, 2, 1)

#     def to_dict(self):
#         """
#         Save the normalization parameters in a dictionary.
#         Returns:
#             dict: A dictionary containing 'scale' and 'offset' tensors.
#         """
#         return {
#             'scale': self.scale,  # Convert to NumPy array if needed
#             'offset': self.offset  # Convert to NumPy array if needed
#         }


class DynamicModule(nn.Module):
    def __init__(self):
        super(DynamicModule, self).__init__()
        self.has_internal_state = False

    def get_requested_input(self, requested_output='internal'):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def init_hidden(self, batch_size, device=None):
        if self.has_internal_state:
            raise NotImplementedError

