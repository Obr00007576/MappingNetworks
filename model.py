import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from collections import OrderedDict
import torch.nn.functional as F
from utils import easy_orthogonal, hard_orthogonal, harder_orthogonal, check_nan, direct_orthogonal
from torch.nn.init import orthogonal_


class MappingNetworks(nn.Module):
    
    def __init__(self, z_dim, param_shapes, is_plus_z = True, is_W_orthogonal = True):
        super().__init__()
        
        self.z = nn.Parameter(torch.randn(z_dim))
        self.param_shapes = param_shapes
        self.param_sizes = OrderedDict()
        total_p = 0
        for name, shape in param_shapes.items():
            size = reduce(mul, shape, 1)
            self.param_sizes[name] = size
            total_p += size
        self.total_p = total_p
        if is_W_orthogonal:
            W = direct_orthogonal(z_dim, total_p)
        else:
            W = torch.nn.init.normal_(torch.empty(z_dim, total_p))
            W = F.normalize(W, dim=1)
        self.register_buffer("W", W)
        self.alpha = 1e-7
        self.is_plus_z = is_plus_z

    def forward(self):
        W_n = self.W  
        if self.is_plus_z:
            W_n = self.W + self.z.unsqueeze(1)*self.alpha
        theta_flat = self.z @ W_n
        params = OrderedDict()
        start = 0
        for name, shape in self.param_shapes.items():
            size = self.param_sizes[name]
            end = start + size
            piece = theta_flat[start:end]
            params[name] = piece.view(shape)
            start = end
        return params

class LatentClassifier(nn.Module):
    
    def __init__(self, z_dim=1024, is_plus_z = True, is_W_orthogonal = True):
        super().__init__()
        
        param_shapes = OrderedDict({
            "fc1_weight": (128, 784),
            "fc1_bias": (128,),
            "fc2_weight": (10, 128),
            "fc2_bias": (10,),
        })
        self.mapping_networks = MappingNetworks(z_dim = z_dim, 
                                                param_shapes = param_shapes, 
                                                is_plus_z = is_plus_z, 
                                                is_W_orthogonal = is_W_orthogonal)

    def forward(self, x):
        p = self.mapping_networks()
        h = F.linear(
            x,
            p["fc1_weight"],
            p["fc1_bias"]
        )
        h = F.relu(h)
        logits = F.linear(
            h,
            p["fc2_weight"],
            p["fc2_bias"]
        )
        return logits
    
class Classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        param_shapes = OrderedDict({
            "fc1_weight": (128, 784),
            "fc1_bias": (128,),
            "fc2_weight": (10, 128),
            "fc2_bias": (10,),
        })
        self.params = nn.ParameterDict({name: nn.Parameter(torch.empty(shape))for name, shape in param_shapes.items()})
        for name, p in self.params.items():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            else:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        params = self.params
        h = F.linear(
            x,
            params["fc1_weight"],
            params["fc1_bias"]
        )
        h = F.relu(h)
        logits = F.linear(
            h,
            params["fc2_weight"],
            params["fc2_bias"]
        )
        return logits

class LatentConVAE(nn.Module):
    
    def __init__(self,
        input_size=28,
        in_channels=1,
        hidden_dims=[16, 32, 64],
        kernel_size=3,
        stride=2,
        padding=1,
        z_dim=128,
        latent_size = 128):
        super().__init__()
        
        current_size = input_size
        for i in range(len(hidden_dims)):
            current_size = (current_size+2*padding-kernel_size)//stride + 1
        last_feature_size = current_size
        
        self.last_feature_size = last_feature_size
        param_shapes = {
            "fc_mu_weight": (latent_size, hidden_dims[-1]*last_feature_size*last_feature_size),
            "fc_mu_bias": (latent_size,),

            "fc_logvar_weight": (latent_size, hidden_dims[-1]*last_feature_size*last_feature_size),
            "fc_logvar_bias": (latent_size,),
            
            "dec_fc_weight": (hidden_dims[-1]*last_feature_size*last_feature_size, latent_size),
            "dec_fc_bias": (hidden_dims[-1]*last_feature_size*last_feature_size,),
        }
        hidden_dims = [in_channels] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[i]
            next_hidden_dim = hidden_dims[i+1]
            param_shapes[f"enc{i+1}_weight"] = (next_hidden_dim, hidden_dim, kernel_size, kernel_size)
            param_shapes[f"enc{i+1}_bias"] = (next_hidden_dim,)
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[len(hidden_dims) - i - 1]
            next_hidden_dim = hidden_dims[len(hidden_dims) - i - 2]
            param_shapes[f"dec{i+1}_weight"] = (hidden_dim, next_hidden_dim, kernel_size, kernel_size)
            param_shapes[f"dec{i+1}_bias"] = (next_hidden_dim,)
        self.stride = stride
        self.padding = padding
        self.hidden_dims = hidden_dims
        self.mapping_networks = MappingNetworks(z_dim = z_dim, param_shapes = param_shapes)
        
    def forward(self, x):
        params = self.mapping_networks()
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, params)
        return {"recon": recon, "mu": mu, "logvar": logvar}
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, params):
        for i in range(len(self.hidden_dims) - 1):     
            x = F.conv2d(
                x,
                params[f"enc{i+1}_weight"],
                params[f"enc{i+1}_bias"],
                stride=self.stride,
                padding=self.padding,
            )
            x = F.leaky_relu(x, 0.2)

        x = torch.flatten(x, 1)
        mu = F.linear(
            x,
            params["fc_mu_weight"],
            params["fc_mu_bias"],
        )
        logvar = F.linear(
            x,
            params["fc_logvar_weight"],
            params["fc_logvar_bias"],
        )
        return mu, logvar
    
    def decode(self, z, params):
        x = F.linear(
            z,
            params["dec_fc_weight"],
            params["dec_fc_bias"],
        )
        x = x.view(-1, self.hidden_dims[-1], self.last_feature_size, self.last_feature_size)
        for i in range(len(self.hidden_dims) - 1):
            output_padding = 0 if i==0 else 1
            x = F.conv_transpose2d(
                x,
                params[f"dec{i+1}_weight"],
                params[f"dec{i+1}_bias"],
                stride=self.stride,
                padding=self.padding,
                output_padding=output_padding
            )
            if i < len(self.hidden_dims) - 2:
                x = F.leaky_relu(x, 0.2)
        x = torch.sigmoid(x)
        return x
    
class ConVAE(nn.Module):
    
    def __init__(self,
        input_size=28,
        in_channels=1,
        hidden_dims=[16, 32, 64],
        kernel_size=3,
        stride=2,
        padding=1,
        latent_size = 128):
        super().__init__()
        
        current_size = input_size
        for i in range(len(hidden_dims)):
            current_size = (current_size+2*padding-kernel_size)//stride + 1
        last_feature_size = current_size
        
        self.last_feature_size = last_feature_size
        param_shapes = {
            "fc_mu_weight": (latent_size, hidden_dims[-1]*last_feature_size*last_feature_size),
            "fc_mu_bias": (latent_size,),

            "fc_logvar_weight": (latent_size, hidden_dims[-1]*last_feature_size*last_feature_size),
            "fc_logvar_bias": (latent_size,),
            
            "dec_fc_weight": (hidden_dims[-1]*last_feature_size*last_feature_size, latent_size),
            "dec_fc_bias": (hidden_dims[-1]*last_feature_size*last_feature_size,),
        }
        hidden_dims = [in_channels] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[i]
            next_hidden_dim = hidden_dims[i+1]
            param_shapes[f"enc{i+1}_weight"] = (next_hidden_dim, hidden_dim, kernel_size, kernel_size)
            param_shapes[f"enc{i+1}_bias"] = (next_hidden_dim,)
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[len(hidden_dims) - i - 1]
            next_hidden_dim = hidden_dims[len(hidden_dims) - i - 2]
            param_shapes[f"dec{i+1}_weight"] = (hidden_dim, next_hidden_dim, kernel_size, kernel_size)
            param_shapes[f"dec{i+1}_bias"] = (next_hidden_dim,)
        self.params = nn.ParameterDict({name: nn.Parameter(torch.empty(shape))for name, shape in param_shapes.items()})
        for name, p in self.params.items():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "logvar" in name:
                if "bias" in name:
                    nn.init.constant_(p, -4.0)
                else:
                    nn.init.normal_(p, 0, 0.02)
            elif "mu" in name:
                if "bias" in name:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, 0, 0.02)
            else:
                nn.init.xavier_normal_(p)
        self.stride = stride
        self.padding = padding
        self.hidden_dims = hidden_dims
        
        
    def forward(self, x):
        params = self.params
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, params)
        return {"recon": recon, "mu": mu, "logvar": logvar}
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, params):
        for i in range(len(self.hidden_dims) - 1):     
            x = F.conv2d(
                x,
                params[f"enc{i+1}_weight"],
                params[f"enc{i+1}_bias"],
                stride=self.stride,
                padding=self.padding,
            )
            x = F.leaky_relu(x, 0.2)

        x = torch.flatten(x, 1)
        mu = F.linear(
            x,
            params["fc_mu_weight"],
            params["fc_mu_bias"],
        )
        logvar = F.linear(
            x,
            params["fc_logvar_weight"],
            params["fc_logvar_bias"],
        )
        return mu, logvar
    
    def decode(self, z, params):
        x = F.linear(
            z,
            params["dec_fc_weight"],
            params["dec_fc_bias"],
        )
        x = x.view(-1, self.hidden_dims[-1], self.last_feature_size, self.last_feature_size)
        for i in range(len(self.hidden_dims) - 1):
            output_padding = 0 if i==0 else 1
            x = F.conv_transpose2d(
                x,
                params[f"dec{i+1}_weight"],
                params[f"dec{i+1}_bias"],
                stride=self.stride,
                padding=self.padding,
                output_padding=output_padding
            )
            if i < len(self.hidden_dims) - 2:
                x = F.leaky_relu(x, 0.2)
        x = torch.sigmoid(x)
        return x