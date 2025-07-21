# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import logging

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # spatially downsamples the 7x7 2D features from the Encoder to a 2x2 resolution
        # preparing them for transformation into a small 3D cube.
        self.spatial_reduce = nn.AdaptiveAvgPool2d((2, 2))  # [256, 7, 7] -> [256, 2, 2]

        # Layer Definition (3D Transposed Convolutions/deconvolutions)
        # ConvTranspose3d : upsample the volumetric features, gradually increasing the spatial resolution of the 3D grid
        # BatchNorm3d  : stabilize and accelerate training by normalizing the activations of previous layers
        # ReLU : Activation function to introduce non-linearity, allowing the network to learn complex mappings
        # Stride =2 : doubles the output spatial dimension
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=(6, 4, 4), stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
        )

    def forward(self, image_features):
            # Input: [batch_size, n_views, channels, height, width] e.g.:torch.Size([64, 1, 256, 7, 7])
            batch_size, n_views, channels, height, width = image_features.shape

            # Reshape for batched processing: [batch_size * n_views, channels, height, width]
            # Allows parallel processing of each view's features
            combined_features = image_features.view(batch_size * n_views, channels, height, width)
            #logging.info(f"[Decoder] Combined features shape: {combined_features.shape}")                           # e.g., [64*24, 256, 7, 7]

            # Apply spatial dimension reduction using AdaptiveAvgPool2d
            # [batch_size * n_views, 256, 7, 7] -> [batch_size * n_views, 256, 2, 2]
            gen_volume = self.spatial_reduce(combined_features)
            #logging.info(f"[Decoder] After spatial_reduce: {gen_volume.shape}")                                     # e.g., [64*24, 256, 2, 2]

            # Replicate the 2D features along a new depth dimension to create a 2x2x2 small initial cube
            # .unsqueeze(2) adds a new dimension
            # .expand(-1, -1, 2, -1, -1) replicates the existing data twice along this new dimension
            # .contiguous() ensures the memory layout is correct for subsequent 3D operations
            # [batch_size * n_views, 256, 2, 2] -> [batch_size * n_views, 256, 2, 2, 2]
            gen_volume = gen_volume.unsqueeze(2).expand(-1, -1, 2, -1, -1).contiguous()
            # -1 in expand means "don't change this dimension's size", useful when batch_size*n_views is variable
            #logging.info(f"[Decoder] After unsqueeze and expand for 3D init: {gen_volume.shape}")                   # e.g., [64*24, 256, 2, 2, 2]


            # Process through layers
            gen_volume = self.layer1(gen_volume)                                                                     # [batch_size * n_views, 128, 4, 4, 4]
            #logging.info(f"[Decoder] After layer1: {gen_volume.shape}")
            gen_volume = self.layer2(gen_volume)                                                                     # [batch_size * n_views, 64, 8, 8, 8]
            #logging.info(f"[Decoder] After layer2: {gen_volume.shape}")
            gen_volume = self.layer3(gen_volume)                                                                     # [batch_size * n_views, 32, 16, 16, 16]
            #logging.info(f"[Decoder] After layer3: {gen_volume.shape}")
            gen_volume = self.layer4(gen_volume)                                                                     # [batch_size * n_views, 8, 32, 32, 32]
            #logging.info(f"[Decoder] After layer4: {gen_volume.shape}")

            # raw_feature : This richer, multi-channel 3D feature map is often useful for subsequent module merger.
            raw_feature = gen_volume                                                                                 # [batch_size * n_views, 8, 32, 32, 32]
            # This 1-channel volume represents the raw logits for the occupancy of each voxel
            gen_volume = self.layer5(gen_volume)                                                                     # [batch_size * n_views, 1, 32, 32, 32]
            #logging.info(f"[Decoder] After layer5: {gen_volume.shape}")

            # Concatenate raw_feature and gen_volume along channel dimension
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)                                                # [batch_size * n_views, 9, 32, 32, 32]
            #logging.info(f"[Decoder] After concatenation: {raw_feature.shape}")

            # Reshape outputs back to [batch_size, n_views, ...]
            gen_volumes = gen_volume.view(batch_size, n_views, 1, 32, 32, 32).squeeze(2)                             # [batch_size, n_views, 32, 32, 32]
            raw_features = raw_feature.view(batch_size, n_views, 9, 32, 32, 32)                                      # [batch_size, n_views, 9, 32, 32, 32]

            #logging.info(f"[Decoder] Final gen_volumes shape: {gen_volumes.shape}")
            #logging.info(f"[Decoder] Final raw_features shape: {raw_features.shape}")

            return raw_features, gen_volumes
