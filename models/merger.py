# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn 

class Merger(nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        # Layer 1 --> Layer 4 (Convolutional Blocks)
        # Conv3d      : standard 3D convolutional layers. primarily serve to process and refine the volumetric raw_features, 
        #               learning more robust local patterns within each individual view's feature representation.
        # BatchNorm3d : Stabilize training by normalizing the feature maps.
        # LeakyReLU   : Introduces non-linearity, allowing the network to learn complex relationships.
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
        # Layer 5 (Feature Concatenation and Reduction)
        # The outputs of (layer 1 --> Layer 4) are concatenated along the channel dimension (9*4 = 36 channels).
        # Allows the network to combine different levels of processed information about the volumetric features.
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(36, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        # Layer 6 (Final Weight Generation Layer)
        # 3D convolutional block takes the 9-channel features from layer5 and reduces them to a 1 channel.
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
        # raw_features pass sequentially through layer1 to layer4
        # Each layer further processes and refines these 3D feature maps, learning different aspects of the input
        volume_weight1 = self.layer1(raw_features_reshaped)                                   # [B*N, 9, 32, 32, 32]
        volume_weight2 = self.layer2(volume_weight1)                                          # [B*N, 9, 32, 32, 32]
        volume_weight3 = self.layer3(volume_weight2)                                          # [B*N, 9, 32, 32, 32]
        volume_weight4 = self.layer4(volume_weight3)                                          # [B*N, 9, 32, 32, 32]
        
        # Concatenate: [B*N, 9*4=36, 32, 32, 32]
        # The outputs of (layer 1 --> Layer 4) are concatenated along the channel dimension (9*4 = 36 channels).
        # Then passed through layer5 to reduce it back to 9 channels
        volume_weight = self.layer5(torch.cat([
            volume_weight1, volume_weight2, volume_weight3, volume_weight4
        ], dim=1))                                                                            # [B*N, 9, 32, 32, 32]
        
        # Converts the 9-channel feature map into a 1-channel volume_weight tensor
        volume_weight = self.layer6(volume_weight)                                            # [B*N, 1, 32, 32, 32]
        
        # Remove the channel dimension
        volume_weights_single_channel = torch.squeeze(volume_weight, dim=1)                   # [B*N, 32, 32, 32]

        # Reshape back to separate views before softmax
        # So that each view has its own learned voxel-wise weights.
        # [batch_size * n_views, 32, 32, 32] -> [batch_size, n_views, 32, 32, 32]
        volume_weights = volume_weights_single_channel.view(batch_size, n_views, 32, 32, 32)
        
        # MOST CRITICAL STEP OF MERGER : For each individual voxel (x, y, z), this operation 
        # ensures that the sum of its weights across all n_views is equal to 1.
        # merger learns how much 'confidence' or importance to assign to each view's prediction 
        # for every single voxel in the 3D grid.
        # Apply softmax over the n_views dimension (dim=1)
        volume_weights = torch.softmax(volume_weights, dim=1)

        # Element-wise multiplication and summation
        # coarse_volumes: [batch_size, n_views, 32, 32, 32]
        # volume_weights: [batch_size, n_views, 32, 32, 32]
        fused_volumes = coarse_volumes * volume_weights
        merged_volume = torch.sum(fused_volumes, dim=1) # Sum over the n_views dimension

        #return torch.clamp(merged_volume, min=0, max=1) # Uncomment if you need explicit clamping
        return merged_volume
