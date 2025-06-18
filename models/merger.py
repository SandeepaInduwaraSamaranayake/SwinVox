# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn # Added nn for clarity, though it's implicitly used

class Merger(nn.Module): # Inherit from nn.Module for clarity
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition (remain the same)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(36, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, raw_features, coarse_volumes):
        # Input shapes:
        # raw_features: [batch_size, n_views, 9, 32, 32, 32]
        # coarse_volumes: [batch_size, n_views, 32, 32, 32]

        batch_size, n_views, _, _, _, _ = raw_features.shape # Get batch_size and n_views

        # Reshape raw_features for batched processing by the 3D convolutions
        # [batch_size, n_views, 9, 32, 32, 32] -> [batch_size * n_views, 9, 32, 32, 32]
        raw_features_reshaped = raw_features.view(batch_size * n_views, 9, 32, 32, 32)

        # Apply layers in a batched manner
        volume_weight1 = self.layer1(raw_features_reshaped)                                   # [B*N, 9, 32, 32, 32]
        volume_weight2 = self.layer2(volume_weight1)                                          # [B*N, 9, 32, 32, 32]
        volume_weight3 = self.layer3(volume_weight2)                                          # [B*N, 9, 32, 32, 32]
        volume_weight4 = self.layer4(volume_weight3)                                          # [B*N, 9, 32, 32, 32]
        
        # Concatenate: [B*N, 9*4=36, 32, 32, 32]
        volume_weight = self.layer5(torch.cat([
            volume_weight1, volume_weight2, volume_weight3, volume_weight4
        ], dim=1))                                                                            # [B*N, 9, 32, 32, 32]
        
        volume_weight = self.layer6(volume_weight)                                            # [B*N, 1, 32, 32, 32]
        
        # Remove the channel dimension
        volume_weights_single_channel = torch.squeeze(volume_weight, dim=1)                   # [B*N, 32, 32, 32]

        # Reshape back to separate views before softmax
        # [batch_size * n_views, 32, 32, 32] -> [batch_size, n_views, 32, 32, 32]
        volume_weights = volume_weights_single_channel.view(batch_size, n_views, 32, 32, 32)
        
        # Apply softmax over the n_views dimension (dim=1)
        volume_weights = torch.softmax(volume_weights, dim=1)

        # Element-wise multiplication and summation remains the same
        # coarse_volumes: [batch_size, n_views, 32, 32, 32]
        # volume_weights: [batch_size, n_views, 32, 32, 32]
        fused_volumes = coarse_volumes * volume_weights
        merged_volume = torch.sum(fused_volumes, dim=1) # Sum over the n_views dimension

        #return torch.clamp(merged_volume, min=0, max=1) # Uncomment if you need explicit clamping
        return merged_volume