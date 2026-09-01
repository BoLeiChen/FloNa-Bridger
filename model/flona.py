import os
import argparse
import time


import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class flona(nn.Module):
    def __init__(
                self, 
                vision_encoder, 
                noise_pred_net,
                dist_pred_net
                ):
        super(flona, self).__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
       
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["floorplan_img"],
                                         kwargs["obs_pos"], kwargs["goal_pos"], kwargs["obs_ori"],
                                         astar_local_cond=kwargs.get("astar_local_cond"),
                                         astar_global_cond=kwargs.get("astar_global_cond")) #, input_goal_mask=kwargs["input_goal_mask"]
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"],
                                         local_cond=kwargs.get("local_cond"),
                                         global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsfloorplan_cond"])
        else:
            raise NotImplementedError
        return output

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output

# class PositionPredictionDenseNet(nn.Module):
#     def __init__(self, embedding_dim):
#         super(PositionPredictionDenseNet, self).__init__()
        
#         self.embedding_dim = embedding_dim 
#         self.network = nn.Sequential(
#             nn.Linear(self.embedding_dim, self.embedding_dim//4),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim//16, 4)
#         )
    
#     def forward(self, x):
#         x = x.reshape((-1, self.embedding_dim))
#         output = self.network(x)
#         return output
